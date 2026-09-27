from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) Negative Log-Likelihood Loss.

    ZINB Distribution:
        P(X=x | μ, θ, π) =
            π + (1-π) * NB(0; μ, θ)          if x = 0
            (1-π) * NB(x; μ, θ)              if x > 0

    where NB(x; μ, θ) is the Negative Binomial distribution:
        NB(x; μ, θ) = Γ(x+θ) / (Γ(θ)Γ(x+1)) * (θ/(θ+μ))^θ * (μ/(θ+μ))^x

    Supports two parameterizations for π:
        - probability mode (default): pi is in [0, 1]
        - logits mode (scVI-compatible): pi is unbounded logits, more numerically stable

    Args:
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', or 'none'
        logits: If True, pi input is in logits scale (scVI-style). Default: False.
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean", logits: bool = False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.logits = logits

    def forward(
        self,
        mu: torch.Tensor,  # (B, N) - Mean of NB
        theta: torch.Tensor,  # (B, N) - Dispersion of NB (inverse overdispersion)
        pi: torch.Tensor,  # (B, N) - Dropout prob (or logits if self.logits=True)
        target: torch.Tensor,  # (B, N) - Raw counts (target)
        mask: Optional[torch.Tensor] = None,  # (B, N) - True for valid positions
        weight: Optional[torch.Tensor] = None,  # (B, N) - Optional per-entry weights
    ) -> torch.Tensor:
        """
        Compute ZINB NLL Loss.

        Returns:
            Scalar loss value (if reduction='mean' or 'sum') or (B, N) tensor (if 'none')
        """
        # Ensure positive values for numerical stability
        mu = mu.clamp(min=self.eps)
        theta = theta.clamp(min=self.eps)

        # Pre-compute common terms
        theta_mu = theta + mu
        log_theta_eps = torch.log(theta + self.eps)
        log_theta_mu_eps = torch.log(theta_mu + self.eps)

        # log NB(0; μ, θ) = θ * log(θ/(θ+μ))
        log_nb_zero = theta * (log_theta_eps - log_theta_mu_eps)

        # log NB(x; μ, θ) for x > 0
        log_nb_nonzero = (
            torch.lgamma(target + theta + self.eps)
            - torch.lgamma(theta + self.eps)
            - torch.lgamma(target + 1.0)
            + theta * (log_theta_eps - log_theta_mu_eps)
            + target * (torch.log(mu + self.eps) - log_theta_mu_eps)
        )

        # Compute ZINB log-likelihood based on parameterization
        if self.logits:
            # scVI-style: pi is zi_logits (unbounded), use softplus for stability
            zi_logits = pi  # unbounded logits

            # case x = 0:
            # log P(X=0) = softplus(log_nb_zero - zi_logits) - softplus(-zi_logits)
            # Derivation: log(sigmoid(l) + (1-sigmoid(l))*NB(0))
            #           = log(exp(l) + exp(log_nb_zero)) - log(1+exp(l))
            #           = softplus(log_nb_zero - l) - softplus(-l)  (after algebra)
            case_zero = (
                F.softplus(log_nb_zero - zi_logits) - F.softplus(-zi_logits)
            )

            # case x > 0:
            # log P(X>0) = log(1-π) + log NB(x)
            #            = log(sigmoid(-zi_logits)) + log NB(x)
            #            = -softplus(zi_logits) + log NB(x)
            case_nonzero = -F.softplus(zi_logits) + log_nb_nonzero
        else:
            # Probability mode: pi is in [0, 1]
            pi = pi.clamp(min=self.eps, max=1.0 - self.eps)
            log_pi = torch.log(pi + self.eps)
            log_one_minus_pi = torch.log(1.0 - pi + self.eps)

            # case x = 0: log P(X=0) = log(π + (1-π) * NB(0))
            case_zero = torch.logsumexp(
                torch.stack([log_pi, log_one_minus_pi + log_nb_zero], dim=-1), dim=-1
            )

            # case x > 0: log P(X>0) = log(1-π) + log NB(x)
            case_nonzero = log_one_minus_pi + log_nb_nonzero

        # Select based on target value
        is_zero = target < 0.5
        log_prob = torch.where(is_zero, case_zero, case_nonzero)

        # Negative log-likelihood
        nll = -log_prob

        # Apply mask if provided
        if mask is not None:
            nll = nll * mask.float()
        if weight is not None:
            nll = nll * weight

        # Reduction
        # Per-cell mode: sum over genes, then average over the batch.
        if self.reduction == "mean":
            if mask is not None:
                denom = mask.float()
                if weight is not None:
                    denom = denom * weight
                return nll.sum() / (denom.sum() + self.eps)
            if weight is not None:
                return nll.sum() / (weight.sum() + self.eps)
            return nll.mean()
        elif self.reduction == "per_cell":
            # Sum over genes (dim=1), then average over cells (dim=0).
            return nll.sum(dim=1).mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll


class NBLoss(nn.Module):
    """
    Negative Binomial (NB) Negative Log-Likelihood Loss.

    NB Distribution:
        P(X=x | μ, θ) = Γ(x+θ) / (Γ(θ)Γ(x+1)) × (θ/(θ+μ))^θ × (μ/(θ+μ))^x

    Unlike ZINB, NB has no zero-inflation parameter π.
    This forces the model to explain zeros through μ and θ alone,
    providing stronger pressure for accurate μ magnitude learning.

    Reference: Svensson (2020), Townes et al. (2019) — UMI-based scRNA-seq
    data does not require zero-inflation modeling.

    Args:
        eps: Small constant for numerical stability
        reduction: 'mean', 'sum', 'per_cell', or 'none'
    """

    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        mu: torch.Tensor,       # (B, N) - Mean of NB
        theta: torch.Tensor,    # (B, N) - Dispersion of NB
        target: torch.Tensor,   # (B, N) - Raw counts (target)
        mask: Optional[torch.Tensor] = None,   # (B, N) - True for valid positions
        weight: Optional[torch.Tensor] = None,  # (B, N) - Optional per-entry weights
    ) -> torch.Tensor:
        """
        Compute NB NLL Loss.

        Returns:
            Scalar loss value (if reduction='mean' or 'sum') or (B, N) tensor (if 'none')
        """
        mu = mu.clamp(min=self.eps)
        theta = theta.clamp(min=self.eps)

        # NB log-likelihood:
        # log P(X=x|μ,θ) = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
        #                  + θ*log(θ/(θ+μ)) + x*log(μ/(θ+μ))
        log_theta_mu = torch.log(theta + mu + self.eps)
        log_prob = (
            torch.lgamma(target + theta + self.eps)
            - torch.lgamma(theta + self.eps)
            - torch.lgamma(target + 1.0)
            + theta * (torch.log(theta + self.eps) - log_theta_mu)
            + target * (torch.log(mu + self.eps) - log_theta_mu)
        )

        nll = -log_prob

        if mask is not None:
            nll = nll * mask.float()
        if weight is not None:
            nll = nll * weight

        # Reduction
        if self.reduction == "mean":
            if mask is not None:
                denom = mask.float()
                if weight is not None:
                    denom = denom * weight
                return nll.sum() / (denom.sum() + self.eps)
            if weight is not None:
                return nll.sum() / (weight.sum() + self.eps)
            return nll.mean()
        elif self.reduction == "per_cell":
            # Sum over genes (dim=1), then average over cells (dim=0).
            return nll.sum(dim=1).mean()
        elif self.reduction == "sum":
            return nll.sum()
        else:
            return nll
