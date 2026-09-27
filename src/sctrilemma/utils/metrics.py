from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

if TYPE_CHECKING:
    import numpy


def compute_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "rbf",
    bandwidth: Optional[float] = None,
) -> torch.Tensor:
    x_flat = x.flatten(1)
    y_flat = y.flatten(1)

    if bandwidth is None:
        with torch.no_grad():
            all_samples = torch.cat([x_flat, y_flat], dim=0)
            pairwise_dists = torch.cdist(all_samples, all_samples)
            bandwidth = float(pairwise_dists.median().item()) + 1e-8

    def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        dists = torch.cdist(a, b).pow(2)
        return torch.exp(-dists / (2 * bandwidth**2))

    def linear_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b.T

    kernel_fn = rbf_kernel if kernel == "rbf" else linear_kernel

    k_xx = kernel_fn(x_flat, x_flat)
    k_yy = kernel_fn(y_flat, y_flat)
    k_xy = kernel_fn(x_flat, y_flat)

    n = x_flat.shape[0]
    m = y_flat.shape[0]

    mask_xx = 1.0 - torch.eye(n, device=x.device)
    mask_yy = 1.0 - torch.eye(m, device=y.device)

    mmd = (
        (k_xx * mask_xx).sum() / (n * (n - 1) + 1e-8)
        + (k_yy * mask_yy).sum() / (m * (m - 1) + 1e-8)
        - 2 * k_xy.mean()
    )

    return mmd.clamp(min=0.0)


def compute_count_weight_metrics(
    weights: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    if mask is not None:
        valid = weights[mask]
    else:
        valid = weights.flatten()

    if valid.numel() == 0:
        zero = torch.tensor(0.0, device=weights.device)
        return {
            "count_weights_mean": zero,
            "count_weights_std": zero,
        }

    return {
        "count_weights_mean": valid.mean(),
        "count_weights_std": valid.std(),
    }


def compute_combined_zero_metrics(
    zinb_mu: torch.Tensor,
    zinb_theta: torch.Tensor,
    zinb_pi: Optional[torch.Tensor],
    raw_counts: torch.Tensor,
    padding_mask: torch.Tensor,
    zi_logits: bool = False,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Compute combined ZINB zero prediction metrics.

    PRB-004: Evaluates the full ZINB zero probability
    ``P(x=0) = pi + (1-pi) * NB(0 | mu, theta)``
    against observed zeros, replacing the previous separate pi-only
    (``zero_accuracy``) and mu-only (``zero_recovery_*``) metrics.

    Note:
        This remains supervised against observed zeros in ``raw_counts``.
        It is a better model-side proxy for zero prediction than separate
        pi-only or mu-only metrics, but it does not distinguish technical
        dropout from biological zero on the target side.

    Args:
        zinb_mu: (B, N) predicted mean expression.
        zinb_theta: (B, N) predicted dispersion.
        zinb_pi: (B, N) predicted dropout probability (or logits), or
            ``None`` for NB-only mode.
        raw_counts: (B, N) ground truth counts.
        padding_mask: (B, N) True for valid gene positions.
        zi_logits: If True, ``zinb_pi`` contains logits (pre-sigmoid).
        threshold: Decision boundary for ``P(x=0)`` → binary zero prediction.
        eps: Numerical stability constant.

    Returns:
        Dict with keys ``combined_zero_{accuracy,precision,recall,f1,
        specificity,balanced_acc,expected_rate,observed_rate,
        calibration_gap,pi_fraction}``.
    """
    device = zinb_mu.device
    mask = padding_mask.float()
    n_valid = mask.sum() + eps

    # --- P(x=0) = pi + (1-pi) * NB(0 | mu, theta) ---
    mu = zinb_mu.clamp(min=eps)
    theta = zinb_theta.clamp(min=eps)
    # log NB(0 | mu, theta) = theta * log(theta / (theta + mu))  (loss.py:61)
    log_nb_zero = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    nb_zero = torch.exp(log_nb_zero)  # (B, N)

    if zinb_pi is not None:
        pi = torch.sigmoid(zinb_pi) if zi_logits else zinb_pi.clamp(0.0, 1.0)
        p_zero = pi + (1.0 - pi) * nb_zero
    else:
        pi = None
        p_zero = nb_zero

    p_zero = p_zero.clamp(0.0, 1.0)

    # --- Confusion matrix (positive class = zero) ---
    pred_zero = (p_zero > threshold).float() * mask
    actual_zero = (raw_counts < 0.5).float() * mask
    actual_nonzero = (1.0 - actual_zero) * mask

    tp = (pred_zero * actual_zero).sum()
    fp = (pred_zero * actual_nonzero).sum()
    fn = ((1.0 - pred_zero) * mask * actual_zero).sum()
    tn = ((1.0 - pred_zero) * mask * actual_nonzero).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    balanced_acc = (recall + specificity) / 2.0

    # --- Calibration ---
    expected_rate = (p_zero * mask).sum() / n_valid
    observed_rate = actual_zero.sum() / n_valid
    calibration_gap = expected_rate - observed_rate

    # --- Pi fraction: descriptive direct-pi contribution ratio ---
    # This is a diagnostic about how much of the modeled zero probability
    # comes from the explicit pi path. Low values do not, by themselves,
    # prove that the pi head failed to learn because NB(0 | mu, theta) can
    # legitimately dominate the combined zero probability.
    if pi is not None:
        pi_contrib = (pi * mask).sum()
        total_p_zero = (p_zero * mask).sum() + eps
        pi_fraction = pi_contrib / total_p_zero
    else:
        pi_fraction = torch.tensor(0.0, device=device)

    return {
        "combined_zero_accuracy": accuracy,
        "combined_zero_precision": precision,
        "combined_zero_recall": recall,
        "combined_zero_f1": f1,
        "combined_zero_specificity": specificity,
        "combined_zero_balanced_acc": balanced_acc,
        "combined_zero_expected_rate": expected_rate,
        "combined_zero_observed_rate": observed_rate,
        "combined_zero_calibration_gap": calibration_gap,
        "combined_zero_pi_fraction": pi_fraction,
    }


def compute_pearson_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Pearson correlation between predicted and target values.

    Fully vectorized implementation for GPU efficiency.

    Args:
        pred: (B, N) predicted values (e.g., ZINB mu)
        target: (B, N) target values (e.g., raw counts)
        mask: (B, N) boolean mask for valid positions (True = valid)
        eps: small constant for numerical stability

    Returns:
        Scalar tensor with mean Pearson correlation across batch
    """
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    mask_float = mask.float()
    n_valid = mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)

    # Masked mean per sample
    pred_masked = pred * mask_float
    target_masked = target * mask_float
    pred_mean = pred_masked.sum(dim=-1, keepdim=True) / n_valid  # (B, 1)
    target_mean = target_masked.sum(dim=-1, keepdim=True) / n_valid  # (B, 1)

    # Centered values (masked)
    pred_centered = (pred - pred_mean) * mask_float  # (B, N)
    target_centered = (target - target_mean) * mask_float  # (B, N)

    # Covariance and std per sample
    cov = (pred_centered * target_centered).sum(dim=-1)  # (B,)
    pred_std = (pred_centered**2).sum(dim=-1).sqrt() + eps  # (B,)
    target_std = (target_centered**2).sum(dim=-1).sqrt() + eps  # (B,)

    # Pearson per sample
    corr = cov / (pred_std * target_std)  # (B,)

    # Handle samples with too few valid points (need at least 2)
    valid_samples = n_valid.squeeze(-1) >= 2  # (B,)
    if valid_samples.any():
        return corr[valid_samples].mean()
    return torch.tensor(0.0, device=pred.device)


# =====================================================================
# Reconstruction metric primitives (GPU-accelerated, dense 2D format)
# =====================================================================
#
# The functions in this block are the canonical reusable implementations
# for reconstruction benchmarks.  They were originally written inside
# ``sctrilemma/benchmark/metrics.py`` (and are still imported
# back from there for backwards compatibility) but have been lifted here
# so training and benchmark code share a single implementation.
#
# Conventions:
#   * ``raw``, ``recon`` are (N_cells, N_genes) float32 tensors or arrays
#     in log1p(CP10K) space unless the docstring says otherwise.
#   * Spearman-based helpers use *average-tie* ranks via
#     :func:`rankdata_average_gpu`, which is bit-exact with
#     ``scipy.stats.rankdata(method='average')``.  Ordinal
#     (``argsort``-based) ranks remain available through the
#     ``tie_handling='ordinal'`` kwarg on
#     :func:`compute_spearman_correlation`.


def rankdata_average_gpu(x: torch.Tensor) -> torch.Tensor:
    """scipy.stats.rankdata(x, method='average', axis=-1) on GPU.

    Returns float32 ranks (1-based) with ties averaged.  Computed via
    sorted-position first/last averaging within each tie group — bit-exact
    match to scipy on dense float32 input with +inf sentinels.

    The implementation works on any leading dimensions (ranks are taken
    along the last axis).
    """
    assert x.ndim >= 1
    device = x.device
    n = x.shape[-1]

    sorted_vals, sort_idx = torch.sort(x, dim=-1, stable=True)

    positions = torch.arange(1, n + 1, device=device, dtype=torch.float32)
    shape = [1] * (x.ndim - 1) + [n]
    positions = positions.view(*shape).expand_as(x)

    neq_left = torch.ones_like(sorted_vals, dtype=torch.bool)
    neq_left[..., 1:] = sorted_vals[..., 1:] != sorted_vals[..., :-1]
    pos_for_first = torch.where(
        neq_left, positions, torch.tensor(-float("inf"), device=device)
    )
    first_pos = torch.cummax(pos_for_first, dim=-1).values

    neq_right = torch.ones_like(sorted_vals, dtype=torch.bool)
    neq_right[..., :-1] = sorted_vals[..., :-1] != sorted_vals[..., 1:]
    pos_for_last = torch.where(
        neq_right, positions, torch.tensor(float("inf"), device=device)
    )
    flipped = torch.flip(pos_for_last, dims=[-1])
    flipped_min = torch.cummin(flipped, dim=-1).values
    last_pos = torch.flip(flipped_min, dims=[-1])

    avg_sorted = (first_pos + last_pos) / 2.0
    avg_ranks = torch.empty_like(x, dtype=torch.float32)
    avg_ranks.scatter_(-1, sort_idx, avg_sorted)
    return avg_ranks


def compute_spearman_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    *,
    tie_handling: Literal["average", "ordinal"] = "average",
) -> torch.Tensor:
    """
    Compute Spearman rank correlation between predicted and target values.

    Spearman correlation is the Pearson correlation of the ranks.  Two
    rank conventions are supported:

    * ``tie_handling='average'`` (default): uses
      :func:`rankdata_average_gpu`, which is bit-exact with
      ``scipy.stats.rankdata(method='average')``.  Tied values receive
      the average of their ordinal positions — the field standard and
      the only convention that is numerically faithful on
      log1p(CP10K)-normalised single-cell data where ~90% of entries
      are zero-tied.  Drift vs scipy is at float32 noise level
      (≲ 1e-7) on non-degenerate inputs.

    * ``tie_handling='ordinal'`` (legacy): uses
      ``argsort(dim=-1).argsort(dim=-1)`` — tied values receive arbitrary
      ordinal ranks (0, 1, 2, ...) depending on sort stability.  This
      was the historical default; it is still available for research
      scripts that need to reproduce old Spearman numbers exactly (e.g.
      flat-sparse gene-level paths that cannot yet use average-tie
      ranking without a dedicated scatter kernel).

    The masking strategy differs slightly between the two branches so the
    resulting masked Pearson on ranks still matches a fully masked
    reference:
      - ordinal: masked positions are pushed to ``-finfo.min`` (keeping
        the historical numerics intact).
      - average: masked positions are pushed to ``+finfo.max``.  Under
        average-tie ranking, this places all invalids in a single
        tied-max group at the end of the row, which the subsequent
        ``* mask_float`` multiplier zeros out cleanly without distorting
        the rank spacing of valid entries.

    .. warning::
       The ``average`` branch assumes valid entries of ``pred`` and
       ``target`` are strictly less than ``torch.finfo(dtype).max``.
       This holds for all production callers (ZINB mu, raw counts,
       log1p-CP10K), where values sit at O(10).  If a valid entry ever
       equals ``finfo.max`` it will collide with the mask sentinel and
       be absorbed into the tied-max group, silently corrupting the
       rank of that valid position.  Use ``tie_handling='ordinal'`` if
       your input may contain extreme values; ordinal's ``-finfo.min``
       sentinel has the symmetric caveat for very-negative inputs.

    Args:
        pred: (B, N) predicted values (e.g., ZINB mu)
        target: (B, N) target values (e.g., raw counts)
        mask: (B, N) boolean mask for valid positions (True = valid)
        eps: small constant for numerical stability
        tie_handling: "average" for scipy-equivalent average-tie ranks
            (default, matches scipy.stats.rankdata(method='average')),
            "ordinal" for legacy argsort-based ranks.  The dense zero-shot
            benchmark and training validation both use the default.

    Returns:
        Scalar tensor with mean Spearman correlation across batch
    """
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)

    _B, _N = pred.shape
    device = pred.device

    if tie_handling == "ordinal":
        # Legacy path: ordinal ranks via double argsort.
        large_neg = torch.finfo(pred.dtype).min
        pred_for_rank = pred.clone()
        pred_for_rank[~mask] = large_neg
        target_for_rank = target.clone()
        target_for_rank[~mask] = large_neg
        pred_ranks = pred_for_rank.argsort(dim=-1).argsort(dim=-1).float()
        target_ranks = target_for_rank.argsort(dim=-1).argsort(dim=-1).float()
    elif tie_handling == "average":
        # Average-tie ranks via rankdata_average_gpu.  Push invalids to
        # +inf so they land in a tied-max group that cancels under the
        # subsequent mask multiply.
        large_pos = torch.finfo(pred.dtype).max
        pred_for_rank = pred.clone()
        pred_for_rank[~mask] = large_pos
        target_for_rank = target.clone()
        target_for_rank[~mask] = large_pos
        pred_ranks = rankdata_average_gpu(pred_for_rank)
        target_ranks = rankdata_average_gpu(target_for_rank)
    else:
        raise ValueError(
            f"tie_handling must be 'average' or 'ordinal', got {tie_handling!r}"
        )

    # Now compute Pearson on ranks (reusing the vectorized logic)
    mask_float = mask.float()
    n_valid = mask_float.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)

    # Masked mean per sample
    pred_masked = pred_ranks * mask_float
    target_masked = target_ranks * mask_float
    pred_mean = pred_masked.sum(dim=-1, keepdim=True) / n_valid  # (B, 1)
    target_mean = target_masked.sum(dim=-1, keepdim=True) / n_valid  # (B, 1)

    # Centered values (masked)
    pred_centered = (pred_ranks - pred_mean) * mask_float  # (B, N)
    target_centered = (target_ranks - target_mean) * mask_float  # (B, N)

    # Covariance and std per sample
    cov = (pred_centered * target_centered).sum(dim=-1)  # (B,)
    pred_std = (pred_centered**2).sum(dim=-1).sqrt() + eps  # (B,)
    target_std = (target_centered**2).sum(dim=-1).sqrt() + eps  # (B,)

    # Spearman per sample
    corr = cov / (pred_std * target_std)  # (B,)

    # Handle samples with too few valid points (need at least 2)
    valid_samples = n_valid.squeeze(-1) >= 2  # (B,)
    if valid_samples.any():
        return corr[valid_samples].mean()
    return torch.tensor(0.0, device=device)


# ---------------------------------------------------------------------
# Dense-path reconstruction metric helpers (lifted from benchmark)
# ---------------------------------------------------------------------


def normalize_to_log1p_cp10k(
    X: np.ndarray, in_place: bool = False
) -> np.ndarray:
    """Normalize expression to log1p(CP10K) space.

    Each row is independently scaled: ``log1p(x / sum(x) * 10000)``.
    This is the field-standard comparison space used by scLDM, GEARS, and
    scGen.  The transform is scale-invariant — it produces the same result
    regardless of whether the input is raw counts, library-normalized
    frequencies, or any other positive scaling.

    Args:
        X: 2D array, will be copied unless *in_place* is True.
        in_place: If True, ``X`` is modified in place (no allocation).
            Caller must own the buffer.
    """
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    if in_place and X.dtype == np.float32:
        X /= row_sums
        X *= 10_000
        np.log1p(X, out=X)
        return X
    return np.log1p(X / row_sums * 10_000).astype(np.float32)


def cell_pearson_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    min_valid_count: int = 2,
    eps: float = 1e-8,
) -> float:
    """Weighted cell-wise Pearson on dense (N, G) torch tensors.

    Masked positions are excluded from per-cell mean / covariance.  Cells
    with fewer than ``min_valid_count`` valid entries are dropped from
    the outer mean.  Returns a Python float for easy dataclass insertion.
    """
    mask_f = mask.float()
    n_valid = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    r_mean = (raw * mask_f).sum(dim=-1, keepdim=True) / n_valid
    p_mean = (recon * mask_f).sum(dim=-1, keepdim=True) / n_valid
    r_c = (raw - r_mean) * mask_f
    p_c = (recon - p_mean) * mask_f
    cov = (r_c * p_c).sum(dim=-1)
    r_std = (r_c * r_c).sum(dim=-1).sqrt().clamp_min(eps)
    p_std = (p_c * p_c).sum(dim=-1).sqrt().clamp_min(eps)
    corr = cov / (r_std * p_std)
    valid_rows = n_valid.squeeze(-1) >= min_valid_count
    if not bool(valid_rows.any()):
        return 0.0
    return float(corr[valid_rows].mean().item())


def cell_spearman_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    mask: torch.Tensor,
    min_valid_count: int = 2,
    eps: float = 1e-8,
) -> float:
    """Weighted cell-wise Spearman with average-tie ranks (scipy-match).

    Uses :func:`rankdata_average_gpu` internally.  Masked positions are
    pushed to +inf before ranking so they share a tied-max group at the
    end of every row and cancel cleanly under the subsequent masked
    Pearson step.
    """
    device = raw.device
    large = torch.finfo(raw.dtype).max
    raw_in = torch.where(mask, raw, torch.tensor(large, device=device, dtype=raw.dtype))
    recon_in = torch.where(mask, recon, torch.tensor(large, device=device, dtype=recon.dtype))
    r_ranks = rankdata_average_gpu(raw_in)
    p_ranks = rankdata_average_gpu(recon_in)
    mask_f = mask.float()
    n_valid = mask_f.sum(dim=-1, keepdim=True).clamp(min=1)
    r_mean = (r_ranks * mask_f).sum(dim=-1, keepdim=True) / n_valid
    p_mean = (p_ranks * mask_f).sum(dim=-1, keepdim=True) / n_valid
    r_c = (r_ranks - r_mean) * mask_f
    p_c = (p_ranks - p_mean) * mask_f
    cov = (r_c * p_c).sum(dim=-1)
    r_std = (r_c * r_c).sum(dim=-1).sqrt().clamp_min(eps)
    p_std = (p_c * p_c).sum(dim=-1).sqrt().clamp_min(eps)
    corr = cov / (r_std * p_std)
    valid_rows = n_valid.squeeze(-1) >= min_valid_count
    if not bool(valid_rows.any()):
        return 0.0
    return float(corr[valid_rows].mean().item())


def gene_pearson_dense(
    raw: torch.Tensor, recon: torch.Tensor, min_cells: int = 10, eps: float = 1e-8
) -> Tuple[float, torch.Tensor]:
    """Mean per-gene Pearson + biased per-gene variance of raw.

    Returns ``(mean_gene_pearson, raw_gene_var)`` where the variance is
    a biased ``sum((x - mean)^2) / n_cells`` tensor of shape (G,) so the
    caller can reuse it for HVG selection without recomputing.  Genes
    with zero std on either side are excluded from the mean.
    """
    n_cells = raw.shape[0]
    r_mean = raw.mean(dim=0, keepdim=True)
    p_mean = recon.mean(dim=0, keepdim=True)
    r_c = raw - r_mean
    p_c = recon - p_mean
    cov = (r_c * p_c).sum(dim=0)
    r_sq = (r_c * r_c).sum(dim=0)
    p_sq = (p_c * p_c).sum(dim=0)
    r_std = r_sq.sqrt().clamp_min(eps)
    p_std = p_sq.sqrt().clamp_min(eps)
    corr = cov / (r_std * p_std)
    valid = (r_std > eps) & (p_std > eps)
    if n_cells < min_cells or not bool(valid.any()):
        mean_corr = float("nan")
    else:
        mean_corr = float(corr[valid].mean().item())
    gene_var = r_sq / max(n_cells, 1)
    return mean_corr, gene_var


def gene_spearman_dense(
    raw: torch.Tensor, recon: torch.Tensor, min_cells: int = 10, eps: float = 1e-8
) -> float:
    """Mean per-gene Spearman (rank over cells per gene, average-tie).

    Only genes with non-degenerate rank std (>eps) contribute to the
    mean, matching the CPU ``_compute_gene_spearman_chunked`` filter.
    """
    if raw.shape[0] < min_cells:
        return float("nan")
    raw_t = raw.transpose(0, 1)  # (G, N) — rank along cells
    recon_t = recon.transpose(0, 1)
    r_ranks = rankdata_average_gpu(raw_t)
    p_ranks = rankdata_average_gpu(recon_t)
    r_mean = r_ranks.mean(dim=-1, keepdim=True)
    p_mean = p_ranks.mean(dim=-1, keepdim=True)
    r_c = r_ranks - r_mean
    p_c = p_ranks - p_mean
    cov = (r_c * p_c).sum(dim=-1)
    r_norm = (r_c * r_c).sum(dim=-1).sqrt()
    p_norm = (p_c * p_c).sum(dim=-1).sqrt()
    denom = r_norm * p_norm
    valid = denom > eps
    if not bool(valid.any()):
        return 0.0
    corrs = cov[valid] / denom[valid]
    finite = torch.isfinite(corrs)
    if not bool(finite.any()):
        return 0.0
    return float(corrs[finite].mean().item())


def hvg_cell_correlations_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    gene_var: torch.Tensor,
    n_top: int = 2000,
    min_cells: int = 10,
) -> Tuple[float, float]:
    """HVG-subset cell-wise Pearson + Spearman.

    Selects the top ``n_top`` genes by raw variance (expected to be the
    biased per-gene variance tensor returned by :func:`gene_pearson_dense`)
    and runs cell-wise correlations on that subset.  Each cell's
    correlation is computed over ``n_top`` genes; the returned values are
    the mean over valid cells.
    """
    n_cells = raw.shape[0]
    if n_cells < min_cells:
        return 0.0, 0.0
    valid = gene_var > 0
    k = min(n_top, int(valid.sum().item()))
    if k < 2:
        return 0.0, 0.0
    top_idx = torch.argsort(gene_var, descending=True)[:k]
    top_idx, _ = torch.sort(top_idx)  # sorted ascending, matches CPU convention
    raw_sub = raw[:, top_idx]
    recon_sub = recon[:, top_idx]
    mask_sub = torch.ones_like(raw_sub, dtype=torch.bool)
    pearson = cell_pearson_dense(raw_sub, recon_sub, mask_sub)
    spearman = cell_spearman_dense(raw_sub, recon_sub, mask_sub)
    return pearson, spearman


def gene_mean_correlation_dense(
    raw: torch.Tensor, recon: torch.Tensor, min_genes: int = 3
) -> Tuple[float, float]:
    """Pearson + Spearman between per-gene mean vectors.

    Measures whether the *population-level* gene expression profile is
    preserved in distribution-matching evaluations. Spearman uses
    :func:`rankdata_average_gpu`.
    """
    if raw.shape[1] < min_genes:
        return 0.0, 0.0
    raw_mean = raw.mean(dim=0)
    rec_mean = recon.mean(dim=0)
    a = raw_mean - raw_mean.mean()
    b = rec_mean - rec_mean.mean()
    denom = a.norm() * b.norm() + 1e-8
    pearson = float(((a * b).sum() / denom).item())
    raw_r = rankdata_average_gpu(raw_mean.unsqueeze(0)).squeeze(0)
    rec_r = rankdata_average_gpu(rec_mean.unsqueeze(0)).squeeze(0)
    a = raw_r - raw_r.mean()
    b = rec_r - rec_r.mean()
    denom = a.norm() * b.norm() + 1e-8
    spearman = float(((a * b).sum() / denom).item())
    return (
        pearson if math.isfinite(pearson) else 0.0,
        spearman if math.isfinite(spearman) else 0.0,
    )


def mmd_pca_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    n_components: int = 30,
    max_cells: int = 2000,
    rng_seed: int = 0,
) -> float:
    """RBF-MMD² on PCA-projected cells.

    Matches benchmark ``_compute_mmd_pca`` + ``compute_mmd`` semantics:
      * Deterministic numpy-seeded subsample without replacement.
      * PCA via torch SVD fit on raw (subsample), project both through
        the same loadings.
      * RBF bandwidth = median pairwise distance across the combined
        (x, y) samples.
      * Returns MMD² with diagonal excluded for kxx/kyy, clamp_min=0.

    The subsample seed matches the CPU implementation so results line up
    across paths.
    """
    n_cells, n_features = raw.shape
    n_comps = min(n_components, n_cells - 1, n_features)
    if n_comps < 2:
        return float("nan")
    if n_cells > max_cells:
        rng = np.random.default_rng(seed=rng_seed)
        idx_np = rng.choice(n_cells, max_cells, replace=False)
        idx = torch.from_numpy(np.sort(idx_np)).to(raw.device)
        raw_s = raw[idx]
        recon_s = recon[idx]
    else:
        raw_s = raw
        recon_s = recon

    mean = raw_s.mean(dim=0, keepdim=True)
    raw_c = raw_s - mean
    recon_c = recon_s - mean
    _U, _S, Vh = torch.linalg.svd(raw_c, full_matrices=False)
    axes = Vh[:n_comps]
    raw_p = raw_c @ axes.T
    recon_p = recon_c @ axes.T

    all_samples = torch.cat([raw_p, recon_p], dim=0)
    with torch.no_grad():
        d = torch.cdist(all_samples, all_samples)
        bandwidth = float(d.median().item()) + 1e-8

    def _rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(a, b).pow(2) / (2 * bandwidth * bandwidth))

    k_xx = _rbf(raw_p, raw_p)
    k_yy = _rbf(recon_p, recon_p)
    k_xy = _rbf(raw_p, recon_p)
    n = raw_p.shape[0]
    m = recon_p.shape[0]
    mask_xx = 1.0 - torch.eye(n, device=raw_p.device)
    mask_yy = 1.0 - torch.eye(m, device=raw_p.device)
    mmd2 = (
        (k_xx * mask_xx).sum() / (n * (n - 1) + 1e-8)
        + (k_yy * mask_yy).sum() / (m * (m - 1) + 1e-8)
        - 2 * k_xy.mean()
    )
    return float(mmd2.clamp_min(0).item())


def gene_corr_matrix_pearson_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    top_k_genes: int = 100,
    min_cells: int = 10,
) -> float:
    """scDesign3 gene-gene correlation matrix Pearson (top-K by expression).

    Takes the top ``top_k_genes`` genes by mean raw expression, builds a
    ``(k, k)`` gene-gene correlation matrix on raw and recon, and returns
    the Pearson correlation between their flattened upper-triangular
    entries.
    """
    n_cells, n_genes = raw.shape
    k = min(top_k_genes, n_genes)
    if k < 3 or n_cells < max(3, 3 * k // 10):
        return float("nan")
    gene_mean = raw.mean(dim=0)
    top_idx = torch.argsort(gene_mean, descending=True)[:k]
    raw_sub = raw[:, top_idx]
    recon_sub = recon[:, top_idx]
    raw_corr = torch.corrcoef(raw_sub.transpose(0, 1))
    rec_corr = torch.corrcoef(recon_sub.transpose(0, 1))
    mask = torch.triu(torch.ones(k, k, device=raw.device, dtype=torch.bool), diagonal=1)
    a = raw_corr[mask]
    b = rec_corr[mask]
    valid = torch.isfinite(a) & torch.isfinite(b)
    if int(valid.sum().item()) < 3:
        return float("nan")
    a = a[valid]
    b = b[valid]
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = a_c.norm() * b_c.norm() + 1e-8
    r = float(((a_c * b_c).sum() / denom).item())
    return r if math.isfinite(r) else 0.0


def cell_corr_ks_dense(
    raw: torch.Tensor,
    recon: torch.Tensor,
    max_cells: int = 2000,
    rng_seed: int = 1,
) -> float:
    """KS distance between pairwise cell-cell correlation distributions.

    GPU builds the correlation matrices via :func:`torch.corrcoef`; the
    final KS statistic runs on CPU via ``scipy.stats.ks_2samp`` because
    KS is O(N log N) sort + linear scan and vectorises poorly.  With
    ``max_cells<=2000`` the upper triangle is <= ~2 million values, cheap.
    """
    from scipy.stats import ks_2samp

    n_cells = raw.shape[0]
    if n_cells > max_cells:
        rng = np.random.default_rng(seed=rng_seed)
        idx_np = rng.choice(n_cells, max_cells, replace=False)
        idx = torch.from_numpy(np.sort(idx_np)).to(raw.device)
        raw_s = raw[idx]
        recon_s = recon[idx]
    else:
        raw_s = raw
        recon_s = recon
    n_sub = raw_s.shape[0]
    if n_sub < 3:
        return float("nan")
    real_corr = torch.corrcoef(raw_s)
    recon_corr = torch.corrcoef(recon_s)
    tri_mask = torch.triu(
        torch.ones(n_sub, n_sub, device=raw.device, dtype=torch.bool), diagonal=1
    )
    real_cc = real_corr[tri_mask].detach().cpu().numpy()
    recon_cc = recon_corr[tri_mask].detach().cpu().numpy()
    valid = np.isfinite(real_cc) & np.isfinite(recon_cc)
    if valid.sum() < 3:
        return float("nan")
    return float(ks_2samp(real_cc[valid], recon_cc[valid]).statistic)


def rank_calibrated_pred_zero(
    raw_counts: np.ndarray, reconstructed: np.ndarray
) -> np.ndarray:
    """Per-cell rank-calibrated predicted-zero mask.

    For each cell, marks the ``k`` lowest reconstruction values as
    predicted-zero, where ``k`` matches the actual number of zeros in the
    raw counts for that cell.  Because log1p is monotonic and
    zero-preserving, operating on log1p(CP10K) inputs gives the same
    mask as operating on raw counts — callers are free to pass either
    space.

    Returns:
        Boolean ``(n_cells, n_genes)`` numpy array — True where the cell's
        bottom-k reconstructed entries correspond to the actual zero
        count.
    """
    n_cells, n_genes = reconstructed.shape
    actual_zero = raw_counts == 0
    pred_zero = np.zeros_like(actual_zero)
    n_zero_per_cell = actual_zero.sum(axis=1)
    for i in range(n_cells):
        k = int(n_zero_per_cell[i])
        if k == 0:
            continue
        if k >= n_genes:
            pred_zero[i] = True
            continue
        bottom_k_idx = np.argpartition(reconstructed[i], k - 1)[:k]
        pred_zero[i, bottom_k_idx] = True
    return pred_zero


def _matrix_chunk_to_numpy(matrix, start: int, end: int) -> np.ndarray:
    """Slice rows [start:end] of a sparse-or-dense matrix and return dense
    float32 numpy.  Private helper used by the CPU chunked paths below."""
    chunk = matrix[start:end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    elif hasattr(chunk, "toarray"):
        chunk = chunk.toarray()
    return np.asarray(chunk, dtype=np.float32)


def compute_mae_and_zero_stats(
    raw_norm: np.ndarray,
    recon_norm: np.ndarray,
    *,
    cell_chunk_size: int = 1024,
) -> Dict[str, float]:
    """Compute MAE and zero/nonzero recovery stats in log1p(CP10K) space.

    Both inputs are expected to already be in log1p(CP10K) space.  Because
    log1p is zero-preserving and monotonic, the zero mask and the bottom-k
    rank ordering are identical to their raw-count-space counterparts; the
    helper therefore operates directly on the normalized buffers, which
    lets the caller normalise in place and skip allocating separate raw
    vs normalized copies of each (N, G) dense matrix.
    """
    abs_error_sum = 0.0
    total_entries = 0
    zero_correct = 0

    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for start in range(0, recon_norm.shape[0], cell_chunk_size):
        end = min(start + cell_chunk_size, recon_norm.shape[0])
        raw_norm_chunk = _matrix_chunk_to_numpy(raw_norm, start, end)
        recon_norm_chunk = np.asarray(recon_norm[start:end], dtype=np.float32)

        abs_error_sum += float(
            np.abs(recon_norm_chunk - raw_norm_chunk).sum(dtype=np.float64)
        )
        total_entries += int(raw_norm_chunk.size)

        actual_zero = raw_norm_chunk == 0.0
        n_cells_chunk = recon_norm_chunk.shape[0]
        pred_zero = np.zeros_like(actual_zero)
        n_zero_per_cell = actual_zero.sum(axis=1)
        for i in range(n_cells_chunk):
            k = int(n_zero_per_cell[i])
            if k == 0:
                continue
            if k >= recon_norm_chunk.shape[1]:
                pred_zero[i] = True
                continue
            bottom_k_idx = np.argpartition(recon_norm_chunk[i], k - 1)[:k]
            pred_zero[i, bottom_k_idx] = True

        zero_correct += int((pred_zero == actual_zero).sum())

        pred_expressed = ~pred_zero
        gt_expressed = ~actual_zero
        gt_zero = actual_zero

        tp += float(np.logical_and(pred_expressed, gt_expressed).sum())
        fp += float(np.logical_and(pred_expressed, gt_zero).sum())
        fn += float(np.logical_and(~pred_expressed, gt_expressed).sum())
        tn += float(np.logical_and(~pred_expressed, gt_zero).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "recon_mae": abs_error_sum / max(total_entries, 1),
        "zero_accuracy": zero_correct / max(total_entries, 1),
        "zero_recovery_precision": precision,
        "zero_recovery_recall": recall,
        "zero_recovery_f1": f1,
        "zero_recovery_specificity": specificity,
        "zero_recovery_accuracy": accuracy,
        "zero_recovery_balanced_acc": (recall + specificity) / 2.0,
    }


def compute_marginal_ks(
    raw_norm: np.ndarray, recon_norm: np.ndarray
) -> Dict[str, float]:
    """scDesign2/3-style marginal-distribution KS distances.

    Both inputs are in log1p(CP10K) space.  Returns gene_mean_ks,
    gene_var_ks, and gene_detect_ks (via rank-calibrated pred_zero); the
    per-cell detection metric is intentionally omitted because under
    rank-calibration real_cell_detect == recon_cell_detect row-wise.
    """
    from scipy.stats import ks_2samp

    out: Dict[str, float] = {}
    n_cells = raw_norm.shape[0]

    if n_cells < 10:
        return {
            "marginal_gene_mean_ks": float("nan"),
            "marginal_gene_var_ks": float("nan"),
            "marginal_gene_detect_ks": float("nan"),
        }

    try:
        real_gene_mean = raw_norm.mean(axis=0)
        recon_gene_mean = recon_norm.mean(axis=0)
        out["marginal_gene_mean_ks"] = float(
            ks_2samp(real_gene_mean, recon_gene_mean).statistic
        )
    except Exception:
        out["marginal_gene_mean_ks"] = float("nan")

    try:
        real_gene_var = raw_norm.var(axis=0)
        recon_gene_var = recon_norm.var(axis=0)
        out["marginal_gene_var_ks"] = float(
            ks_2samp(real_gene_var, recon_gene_var).statistic
        )
    except Exception:
        out["marginal_gene_var_ks"] = float("nan")

    try:
        pred_zero = rank_calibrated_pred_zero(raw_norm, recon_norm)
        real_gene_detect = (raw_norm > 0.0).mean(axis=0)
        recon_gene_detect = (~pred_zero).mean(axis=0)
        out["marginal_gene_detect_ks"] = float(
            ks_2samp(real_gene_detect, recon_gene_detect).statistic
        )
    except Exception:
        out["marginal_gene_detect_ks"] = float("nan")

    return out


# ---------------------------------------------------------------------
# CPU fallback paths (chunked scipy/numpy implementations)
# ---------------------------------------------------------------------
# These used to live in sctrilemma/benchmark/metrics.py as
# private underscore helpers.  They are now hosted here so the dense
# reconstruction-metric pipeline has a single source of truth on both
# the CPU (numpy/scipy/torch-on-CPU) and GPU (torch CUDA) paths.  Callers
# can continue to import the underscore-prefixed names via the
# backwards-compatible aliased imports in
# ``sctrilemma/benchmark/metrics.py``.


def _compute_spearman_correlation_tieaware(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Cell-wise Spearman with average-tie ranks and masking (CPU, scipy)."""
    from scipy.stats import rankdata

    if pred.size == 0:
        return 0.0

    mask_bool = np.asarray(mask, dtype=bool)
    valid_rows = mask_bool.sum(axis=1) >= 2
    if not valid_rows.any():
        return 0.0

    pred_valid = np.asarray(pred[valid_rows], dtype=np.float32)
    target_valid = np.asarray(target[valid_rows], dtype=np.float32)
    mask_valid = mask_bool[valid_rows]

    # Invalid positions are sent to +inf so valid entries keep the same
    # relative ranks they would have under valid-only ranking.
    pred_rank_in = np.where(mask_valid, pred_valid, np.inf)
    target_rank_in = np.where(mask_valid, target_valid, np.inf)
    pred_ranks = rankdata(pred_rank_in, method="average", axis=1).astype(np.float64)
    target_ranks = rankdata(target_rank_in, method="average", axis=1).astype(np.float64)

    mask_float = mask_valid.astype(np.float64)
    n_valid = np.clip(mask_float.sum(axis=1, keepdims=True), 1.0, None)
    pred_mean = (pred_ranks * mask_float).sum(axis=1, keepdims=True) / n_valid
    target_mean = (target_ranks * mask_float).sum(axis=1, keepdims=True) / n_valid

    pred_centered = (pred_ranks - pred_mean) * mask_float
    target_centered = (target_ranks - target_mean) * mask_float
    numer = (pred_centered * target_centered).sum(axis=1)
    denom = np.sqrt((pred_centered ** 2).sum(axis=1)) * np.sqrt(
        (target_centered ** 2).sum(axis=1)
    )

    valid = np.isfinite(numer) & np.isfinite(denom) & (denom > eps)
    if not valid.any():
        return 0.0

    corrs = numer[valid] / denom[valid]
    finite = np.isfinite(corrs)
    if not finite.any():
        return 0.0
    return float(corrs[finite].mean())


def _weighted_chunk_correlation(
    raw_matrix,
    reconstructed: np.ndarray,
    mask_builder,
    corr_fn,
    *,
    min_valid_count: int,
    cell_chunk_size: int,
    raw_transform=None,
) -> float:
    """Cell-chunked weighted Pearson via ``corr_fn`` (torch on CPU)."""
    weighted_sum = 0.0
    valid_total = 0

    for start in range(0, reconstructed.shape[0], cell_chunk_size):
        end = min(start + cell_chunk_size, reconstructed.shape[0])
        raw_chunk = _matrix_chunk_to_numpy(raw_matrix, start, end)
        if raw_transform is not None:
            raw_chunk = np.asarray(raw_transform(raw_chunk), dtype=np.float32)
        recon_chunk = np.asarray(reconstructed[start:end], dtype=np.float32)
        mask_np = np.asarray(mask_builder(raw_chunk), dtype=bool)
        valid_rows = mask_np.sum(axis=1) >= min_valid_count
        if not valid_rows.any():
            continue

        raw_t = torch.from_numpy(raw_chunk)
        recon_t = torch.from_numpy(recon_chunk)
        mask_t = torch.from_numpy(mask_np)
        corr = float(corr_fn(recon_t, raw_t, mask_t))
        valid_count = int(valid_rows.sum())
        weighted_sum += corr * valid_count
        valid_total += valid_count

    if valid_total == 0:
        return 0.0
    return weighted_sum / valid_total


def _weighted_chunk_spearman_correlation(
    raw_matrix,
    reconstructed: np.ndarray,
    mask_builder,
    *,
    min_valid_count: int,
    cell_chunk_size: int,
    raw_transform=None,
) -> float:
    """Cell-chunked weighted Spearman via scipy average-tie ranks."""
    weighted_sum = 0.0
    valid_total = 0

    for start in range(0, reconstructed.shape[0], cell_chunk_size):
        end = min(start + cell_chunk_size, reconstructed.shape[0])
        raw_chunk = _matrix_chunk_to_numpy(raw_matrix, start, end)
        if raw_transform is not None:
            raw_chunk = np.asarray(raw_transform(raw_chunk), dtype=np.float32)
        recon_chunk = np.asarray(reconstructed[start:end], dtype=np.float32)
        mask_np = np.asarray(mask_builder(raw_chunk), dtype=bool)
        valid_rows = mask_np.sum(axis=1) >= min_valid_count
        if not valid_rows.any():
            continue

        corr = _compute_spearman_correlation_tieaware(
            recon_chunk[valid_rows],
            raw_chunk[valid_rows],
            mask_np[valid_rows],
        )
        valid_count = int(valid_rows.sum())
        weighted_sum += corr * valid_count
        valid_total += valid_count

    if valid_total == 0:
        return 0.0
    return weighted_sum / valid_total


def _pearson_1d(
    x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Pearson correlation between two 1-D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom < eps:
        return torch.tensor(float("nan"), device=x.device)
    return (x * y).sum() / denom


def _compute_gene_pearson_chunked(
    raw_matrix,
    reconstructed: np.ndarray,
    cell_chunk_size: int = 10_000,
    min_cells_per_gene: int = 10,
    eps: float = 1e-8,
) -> Tuple[float, np.ndarray]:
    """Gene-level Pearson via cell-chunked online accumulation (CPU).

    Returns (gene_pearson, per_gene_variance_of_raw) for HVG reuse.
    """
    n_cells, n_genes = reconstructed.shape

    sum_x = np.zeros(n_genes, dtype=np.float64)
    sum_y = np.zeros(n_genes, dtype=np.float64)
    sum_xx = np.zeros(n_genes, dtype=np.float64)
    sum_yy = np.zeros(n_genes, dtype=np.float64)
    sum_xy = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_cells, cell_chunk_size):
        end = min(start + cell_chunk_size, n_cells)
        raw_chunk = _matrix_chunk_to_numpy(raw_matrix, start, end).astype(np.float64)
        recon_chunk = reconstructed[start:end].astype(np.float64)

        sum_x += recon_chunk.sum(axis=0)
        sum_y += raw_chunk.sum(axis=0)
        sum_xx += (recon_chunk**2).sum(axis=0)
        sum_yy += (raw_chunk**2).sum(axis=0)
        sum_xy += (recon_chunk * raw_chunk).sum(axis=0)

    mean_x = sum_x / n_cells
    mean_y = sum_y / n_cells
    cov = sum_xy / n_cells - mean_x * mean_y
    var_x = np.clip(sum_xx / n_cells - mean_x**2, 0, None)
    var_y = np.clip(sum_yy / n_cells - mean_y**2, 0, None)
    denom = np.sqrt(var_x) * np.sqrt(var_y)

    if n_cells < min_cells_per_gene:
        return 0.0, var_y

    valid = denom > eps
    if not valid.any():
        return 0.0, var_y

    gene_pearson = float(np.mean(cov[valid] / denom[valid]))
    return gene_pearson, var_y


def _compute_gene_spearman_chunked(
    raw_matrix,
    reconstructed: np.ndarray,
    gene_chunk_size: int = 500,
    min_cells_per_gene: int = 10,
    eps: float = 1e-8,
) -> float:
    """Gene-level Spearman with proper tie handling via scipy (CPU)."""
    from scipy.stats import rankdata

    n_cells, n_genes = reconstructed.shape

    if n_cells < min_cells_per_gene:
        return 0.0

    spearman_sum = 0.0
    n_valid = 0

    for g_start in range(0, n_genes, gene_chunk_size):
        g_end = min(g_start + gene_chunk_size, n_genes)

        raw_slice = raw_matrix[:, g_start:g_end]
        if sp.issparse(raw_slice):
            raw_cols = raw_slice.toarray().astype(np.float32)
        else:
            raw_cols = np.asarray(raw_slice, dtype=np.float32)
        recon_cols = np.asarray(reconstructed[:, g_start:g_end], dtype=np.float32)

        raw_ranks = rankdata(raw_cols, method="average", axis=0).astype(np.float32)
        recon_ranks = rankdata(recon_cols, method="average", axis=0).astype(np.float32)
        del raw_cols, recon_cols

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_t = torch.from_numpy(raw_ranks).to(device)
        recon_t = torch.from_numpy(recon_ranks).to(device)
        del raw_ranks, recon_ranks

        raw_ranks = raw_t
        recon_ranks = recon_t
        del raw_t, recon_t

        raw_ranks -= raw_ranks.mean(dim=0, keepdim=True)
        recon_ranks -= recon_ranks.mean(dim=0, keepdim=True)
        numer = (raw_ranks * recon_ranks).sum(dim=0)
        denom = raw_ranks.norm(dim=0) * recon_ranks.norm(dim=0)
        del raw_ranks, recon_ranks

        valid_mask = denom > eps
        if valid_mask.any():
            corrs = numer[valid_mask] / denom[valid_mask]
            finite_mask = torch.isfinite(corrs)
            spearman_sum += corrs[finite_mask].sum().item()
            n_valid += int(finite_mask.sum().item())

        del numer, denom

    if n_valid == 0:
        return 0.0
    return spearman_sum / n_valid


def _compute_hvg_metrics_chunked(
    raw_matrix,
    reconstructed: np.ndarray,
    gene_var: np.ndarray,
    *,
    top_k_hvg: int = 2000,
    min_cells_per_gene: int = 10,
    cell_chunk_size: int = 10_000,
) -> Dict[str, float]:
    """Cell-level Pearson/Spearman on top-K HVG genes, chunked by cells (CPU)."""
    n_cells, n_genes = reconstructed.shape

    if n_cells < min_cells_per_gene:
        return {"recon_pearson_hvg": 0.0, "recon_spearman_hvg": 0.0}

    gene_var_filtered = gene_var.copy()
    k = min(top_k_hvg, int((gene_var_filtered > 0).sum()))
    if k < 2:
        return {"recon_pearson_hvg": 0.0, "recon_spearman_hvg": 0.0}

    hvg_indices = np.argsort(gene_var_filtered)[-k:]
    hvg_indices.sort()

    weighted_pearson = 0.0
    weighted_spearman = 0.0
    total = 0

    for start in range(0, n_cells, cell_chunk_size):
        end = min(start + cell_chunk_size, n_cells)

        raw_slice = raw_matrix[start:end][:, hvg_indices]
        if sp.issparse(raw_slice):
            raw_hvg = raw_slice.toarray().astype(np.float32)
        else:
            raw_hvg = np.asarray(raw_slice, dtype=np.float32)
        recon_hvg = np.asarray(
            reconstructed[start:end][:, hvg_indices], dtype=np.float32
        )

        n_chunk = end - start
        raw_t = torch.from_numpy(raw_hvg)
        recon_t = torch.from_numpy(recon_hvg)
        mask_t = torch.ones_like(raw_t, dtype=torch.bool)
        p = float(compute_pearson_correlation(recon_t, raw_t, mask_t))
        s = _compute_spearman_correlation_tieaware(
            recon_hvg,
            raw_hvg,
            np.ones_like(raw_hvg, dtype=bool),
        )

        weighted_pearson += p * n_chunk
        weighted_spearman += s * n_chunk
        total += n_chunk

        del raw_hvg, recon_hvg, raw_t, recon_t, mask_t

    if total == 0:
        return {"recon_pearson_hvg": 0.0, "recon_spearman_hvg": 0.0}

    return {
        "recon_pearson_hvg": weighted_pearson / total,
        "recon_spearman_hvg": weighted_spearman / total,
    }


def _compute_gene_and_hvg_metrics(
    raw_matrix,
    reconstructed: np.ndarray,
    *,
    cell_chunk_size: int = 10_000,
    gene_chunk_size: int = 200,
) -> Tuple[Dict[str, float], Optional[str]]:
    """Compute gene-level and HVG metrics (CPU path).

    Caller must pass already-dense ``log1p(CP10K)`` arrays.  HVG selection
    uses variance in log1p(CP10K) space (matches scanpy seurat_v3 style).
    """
    try:
        gene_pearson, gene_var = _compute_gene_pearson_chunked(
            raw_matrix,
            reconstructed,
            cell_chunk_size=cell_chunk_size,
        )
        gene_spearman = _compute_gene_spearman_chunked(
            raw_matrix,
            reconstructed,
            gene_chunk_size=gene_chunk_size,
        )
        hvg_metrics = _compute_hvg_metrics_chunked(
            raw_matrix,
            reconstructed,
            gene_var,
            cell_chunk_size=cell_chunk_size,
        )

        return {
            "gene_pearson": gene_pearson,
            "gene_spearman": gene_spearman,
            "recon_pearson_hvg": hvg_metrics["recon_pearson_hvg"],
            "recon_spearman_hvg": hvg_metrics["recon_spearman_hvg"],
        }, None

    except Exception as exc:
        print(f"Warning: gene/HVG metric computation failed: {exc}")
        return {
            "gene_pearson": math.nan,
            "gene_spearman": math.nan,
            "recon_pearson_hvg": math.nan,
            "recon_spearman_hvg": math.nan,
        }, "computation_failed"


def _compute_gene_mean_correlation(
    raw_norm: np.ndarray,
    recon_norm: np.ndarray,
) -> Tuple[float, float]:
    """Per-gene mean expression correlation (scGen/scLDM style, scipy CPU).

    For each gene, computes the mean across all cells and returns Pearson
    and Spearman correlation between the real and reconstructed gene-mean
    vectors.  Measures whether the *population-level* gene expression
    profile is preserved.  Inputs are expected in log1p(CP10K) space.
    """
    from scipy.stats import pearsonr, spearmanr

    real_mean = raw_norm.mean(axis=0).astype(np.float64)
    gen_mean = recon_norm.mean(axis=0).astype(np.float64)

    valid = np.isfinite(real_mean) & np.isfinite(gen_mean)
    if valid.sum() < 3:
        return 0.0, 0.0

    try:
        p = float(pearsonr(real_mean[valid], gen_mean[valid])[0])
    except Exception:
        p = 0.0
    try:
        s = float(spearmanr(real_mean[valid], gen_mean[valid])[0])
    except Exception:
        s = 0.0

    return (p if math.isfinite(p) else 0.0, s if math.isfinite(s) else 0.0)


def _compute_mmd_pca(
    raw_norm: np.ndarray,
    recon_norm: np.ndarray,
    n_comps: int = 30,
    max_cells: int = 2000,
) -> float:
    """MMD² on PCA-projected log1p(CP10K) space via sklearn PCA (CPU).

    PCA is fit on the real data and the reconstructed data is projected
    via the same loadings.  MMD² is computed with an RBF kernel and
    median-heuristic bandwidth through :func:`compute_mmd`.
    """
    from sklearn.decomposition import PCA

    n_cells = raw_norm.shape[0]
    n_features = raw_norm.shape[1]
    n_comps_actual = min(n_comps, n_cells - 1, n_features)
    if n_comps_actual < 2:
        return float("nan")

    if n_cells > max_cells:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(n_cells, max_cells, replace=False)
        raw_sub = raw_norm[idx]
        recon_sub = recon_norm[idx]
    else:
        raw_sub = raw_norm
        recon_sub = recon_norm

    pca = PCA(n_components=n_comps_actual)
    raw_pca = pca.fit_transform(raw_sub).astype(np.float32)
    recon_pca = pca.transform(recon_sub).astype(np.float32)

    mmd = compute_mmd(
        torch.from_numpy(raw_pca),
        torch.from_numpy(recon_pca),
        kernel="rbf",
    )
    return float(mmd.item())


def _compute_joint_distribution_metrics(
    raw_norm: np.ndarray,
    recon_norm: np.ndarray,
    top_k_genes: int = 100,
    max_cells: int = 2000,
) -> Dict[str, float]:
    """scDesign3-style joint distribution metrics (CPU path).

    Returns both gene_corr_matrix_pearson (top-K highly-expressed genes)
    and cell_corr_ks (pairwise cell-cell correlation KS distance), in
    a single pass.  Each metric has its own try/except so one failure
    does not propagate.
    """
    from scipy.stats import ks_2samp

    out: Dict[str, float] = {}
    n_cells, n_genes = raw_norm.shape

    # --- Gene-gene correlation matrix Pearson r ---
    try:
        k = min(top_k_genes, n_genes)
        if k < 3 or n_cells < max(3, 3 * k // 10):
            out["gene_corr_matrix_pearson"] = float("nan")
        else:
            gene_mean = raw_norm.mean(axis=0)
            top_idx = np.argsort(gene_mean)[-k:]
            real_sub = raw_norm[:, top_idx]
            recon_sub = recon_norm[:, top_idx]

            real_corr = np.corrcoef(real_sub.T)
            recon_corr = np.corrcoef(recon_sub.T)

            iu = np.triu_indices(k, k=1)
            real_flat = real_corr[iu]
            recon_flat = recon_corr[iu]

            valid = np.isfinite(real_flat) & np.isfinite(recon_flat)
            if valid.sum() < 3:
                out["gene_corr_matrix_pearson"] = float("nan")
            else:
                from scipy.stats import pearsonr

                r = float(pearsonr(real_flat[valid], recon_flat[valid])[0])
                out["gene_corr_matrix_pearson"] = r if math.isfinite(r) else 0.0
    except Exception:
        out["gene_corr_matrix_pearson"] = float("nan")

    # --- Cell-cell correlation distribution KS ---
    try:
        if n_cells > max_cells:
            rng = np.random.default_rng(seed=1)  # distinct seed from MMD PCA
            cell_idx = rng.choice(n_cells, max_cells, replace=False)
            real_cc_in = raw_norm[cell_idx]
            recon_cc_in = recon_norm[cell_idx]
        else:
            real_cc_in = raw_norm
            recon_cc_in = recon_norm

        n_sub = real_cc_in.shape[0]
        if n_sub < 3:
            out["cell_corr_ks"] = float("nan")
        else:
            real_cell_corr = np.corrcoef(real_cc_in)
            recon_cell_corr = np.corrcoef(recon_cc_in)
            iu = np.triu_indices(n_sub, k=1)
            real_cc = real_cell_corr[iu]
            recon_cc = recon_cell_corr[iu]
            valid = np.isfinite(real_cc) & np.isfinite(recon_cc)
            if valid.sum() < 3:
                out["cell_corr_ks"] = float("nan")
            else:
                out["cell_corr_ks"] = float(
                    ks_2samp(real_cc[valid], recon_cc[valid]).statistic
                )
    except Exception:
        out["cell_corr_ks"] = float("nan")

    return out


# =====================================================================
# End reconstruction metric primitives
# =====================================================================


def _scatter_pearson_chunked(
    flat_mu: torch.Tensor,
    flat_raw: torch.Tensor,
    flat_idx: torch.Tensor,
    vocab_size: int,
    chunk_size: int = 5_000_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-gene Pearson accumulators via chunked scatter_add_.

    Processes data in chunks to avoid allocating full-size float64 copies.
    Returns (n, sum_x, sum_y, sum_xx, sum_yy, sum_xy) accumulators of shape
    (vocab_size,) in float64.
    """
    device = flat_mu.device
    n = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    sum_x = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    sum_y = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    sum_xx = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    sum_yy = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    sum_xy = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    N = flat_mu.shape[0]
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx_c = flat_idx[start:end].long()
        mu_c = flat_mu[start:end].double()
        raw_c = flat_raw[start:end].double()
        n.scatter_add_(0, idx_c, torch.ones_like(mu_c))
        sum_x.scatter_add_(0, idx_c, mu_c)
        sum_y.scatter_add_(0, idx_c, raw_c)
        sum_xx.scatter_add_(0, idx_c, mu_c * mu_c)
        sum_yy.scatter_add_(0, idx_c, raw_c * raw_c)
        sum_xy.scatter_add_(0, idx_c, mu_c * raw_c)

    return n, sum_x, sum_y, sum_xx, sum_yy, sum_xy


def _env_int(name: str, default: int) -> int:
    """Parse an int env value with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float env value with fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _estimated_dense_float32_gb(matrix) -> float:
    """Estimate dense float32 materialization size in GB."""
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2:
        return 0.0
    n_rows, n_cols = int(shape[0]), int(shape[1])
    return float(n_rows * n_cols * np.dtype(np.float32).itemsize / 1e9)


# =============================================================================
# Bio-conservation Metrics (PRD v1.1.3 Section 13)
# =============================================================================


def stratified_subsample(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    batch_ids: torch.Tensor,
    max_per_type: int = 500,
    min_cells_for_rare: int = 50,
    total_max: int = 10000,
    return_indices: bool = False,
    seed: int = 0,
):
    """
    Stratified subsampling that preserves all cell types.

    Args:
        embeddings: (N, D) embedding tensor
        cell_type_labels: (N,) cell type labels
        batch_ids: (N,) batch IDs
        max_per_type: max cells per type
        min_cells_for_rare: min threshold for including rare types
        total_max: total max cells
        return_indices: if True, also return the sampled row indices
        seed: deterministic seed for subsampling

    Returns:
        (subsampled_embeddings, subsampled_labels, subsampled_batch_ids)
        or (..., sampled_indices) if return_indices=True
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Move to CPU for processing
    embeddings_np = embeddings.detach().cpu().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()
    batch_np = batch_ids.detach().cpu().numpy()

    # Filter out invalid labels
    valid_mask = labels_np >= 0
    if not valid_mask.any():
        if return_indices:
            return embeddings, cell_type_labels, batch_ids, None
        return embeddings, cell_type_labels, batch_ids

    valid_indices = np.where(valid_mask)[0]
    valid_labels = labels_np[valid_mask]

    unique_types, counts = np.unique(valid_labels, return_counts=True)
    type_to_indices = {
        ct: valid_indices[valid_labels == ct]
        for ct in unique_types
    }

    # Keep all valid labels. Rare labels retain all cells by default so
    # isolated-label metrics remain meaningful after subsampling.
    target_counts: dict[int, int] = {}
    rare_types: set[int] = set()
    for ct, count in zip(unique_types, counts):
        ct_key = int(ct)
        ct_count = int(count)
        if ct_count < min_cells_for_rare:
            rare_types.add(ct_key)
            target_counts[ct_key] = ct_count
        else:
            target_counts[ct_key] = min(ct_count, max_per_type)

    total_target = sum(target_counts.values())
    if total_target > total_max:
        reserved_counts: dict[int, int] = {}
        for ct in unique_types:
            ct_key = int(ct)
            target = target_counts[ct_key]
            if ct_key in rare_types:
                reserved_counts[ct_key] = target
            else:
                reserved_counts[ct_key] = 1 if target > 0 else 0

        reserved_total = sum(reserved_counts.values())
        if reserved_total > total_max:
            reserved_counts = {}
            remaining = total_max
            ordered_types = sorted(unique_types, key=lambda ct: len(type_to_indices[ct]))
            for ct in ordered_types:
                if remaining <= 0:
                    break
                reserved_counts[int(ct)] = 1
                remaining -= 1
        else:
            remaining = total_max - reserved_total
            extra_caps = {
                ct_key: max(0, target_counts[ct_key] - reserved_counts[ct_key])
                for ct_key in reserved_counts
            }
            total_extra = sum(extra_caps.values())

            if remaining > 0 and total_extra > 0:
                fractional_parts: list[tuple[float, int]] = []
                for ct_key, cap in extra_caps.items():
                    if cap <= 0:
                        continue
                    raw_extra = remaining * cap / total_extra
                    extra = min(cap, int(np.floor(raw_extra)))
                    reserved_counts[ct_key] += extra
                    fractional_parts.append((raw_extra - extra, ct_key))

                leftover = total_max - sum(reserved_counts.values())
                for _, ct_key in sorted(fractional_parts, reverse=True):
                    if leftover <= 0:
                        break
                    if reserved_counts[ct_key] < target_counts[ct_key]:
                        reserved_counts[ct_key] += 1
                        leftover -= 1

        target_counts = reserved_counts

    sampled_indices = []
    for ct in unique_types:
        ct_key = int(ct)
        ct_indices = type_to_indices[ct]
        n_sample = min(len(ct_indices), target_counts.get(ct_key, 0))
        if n_sample <= 0:
            continue
        if n_sample >= len(ct_indices):
            chosen = ct_indices
        else:
            chosen = rng.choice(ct_indices, n_sample, replace=False)
        sampled_indices.extend(chosen.tolist())

    sampled_indices = np.asarray(sampled_indices, dtype=np.int64)

    sub_emb = torch.from_numpy(embeddings_np[sampled_indices])
    sub_labels = torch.from_numpy(labels_np[sampled_indices])
    sub_batch = torch.from_numpy(batch_np[sampled_indices])

    if return_indices:
        return sub_emb, sub_labels, sub_batch, sampled_indices
    return sub_emb, sub_labels, sub_batch


def _faiss_knn(
    embeddings: "numpy.ndarray",
    k: int = 30,
    metric: str = "euclidean",
) -> tuple["numpy.ndarray", "numpy.ndarray"]:
    """
    Fast k-NN using FAISS.

    Args:
        embeddings: (N, D) numpy array
        k: number of neighbors
        metric: "euclidean" or "cosine"

    Returns:
        (distances, indices) - both (N, k) arrays
        indices excludes self (first neighbor)
    """
    import numpy as np

    try:
        import faiss

        dim = embeddings.shape[1]
        embeddings = embeddings.astype(np.float32)

        if metric == "cosine":
            # Normalize for cosine similarity
            # Use IndexFlatIP (Inner Product)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            embeddings_proc = embeddings / norm
            index = faiss.IndexFlatIP(dim)
        else:
            # Euclidean distance
            # Use IndexFlatL2
            embeddings_proc = embeddings
            index = faiss.IndexFlatL2(dim)

        # Build index
        index.add(embeddings_proc)  # pyright: ignore[reportCallIssue]

        # Search (k+1 to exclude self which is at dist 0)
        D, I_nn = index.search(embeddings_proc, k + 1)  # noqa: E741  # pyright: ignore[reportCallIssue]

        # Remove self (first neighbor)
        D, I_nn = D[:, 1:], I_nn[:, 1:]

        # IndexFlatL2 returns squared L2 distances; convert to euclidean
        # for consistency with sklearn fallback.
        if metric != "cosine":
            D = np.sqrt(np.maximum(D, 0.0))

        return D, I_nn

    except ImportError:
        # Fallback to sklearn
        from sklearn.neighbors import NearestNeighbors

        if metric == "cosine":
            # sklearn uses 'cosine' distance (1 - similarity)
            nn_metric = "cosine"
        else:
            nn_metric = "euclidean"

        nn = NearestNeighbors(n_neighbors=k + 1, metric=nn_metric)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        return distances[:, 1:], indices[:, 1:]


def _get_compute_device() -> torch.device:
    """Get the best available compute device for bio metrics."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _silhouette_samples_torch(
    dist: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Vectorized silhouette using pre-computed distance matrix.

    Args:
        dist: (N, N) pairwise distance matrix (euclidean or cosine).
        labels: (N,) integer cluster labels.

    Returns:
        (N,) per-sample silhouette coefficients.
    """
    N = dist.shape[0]
    device = dist.device

    a = torch.zeros(N, device=device)
    b = torch.full((N,), float("inf"), device=device)

    unique_labels = labels.unique()

    for lbl in unique_labels:
        mask = labels == lbl  # (N,)
        n_in = mask.sum()
        if n_in <= 1:
            continue

        # dist[:, mask].sum(dim=1) = total distance from every point to this cluster
        dist_to_cluster = dist[:, mask].sum(dim=1)  # (N,)

        # a_i: mean intra-cluster distance (self-dist=0, so subtract nothing needed)
        a[mask] = dist_to_cluster[mask] / (n_in - 1).clamp(min=1)

        # b_i candidate: mean distance to this cluster for points NOT in it
        not_mask = ~mask
        if not_mask.any():
            mean_dist = dist_to_cluster[not_mask] / n_in
            b[not_mask] = torch.minimum(b[not_mask], mean_dist)

    sil = (b - a) / torch.clamp(torch.maximum(a, b), min=1e-8)
    return sil


def _silhouette_score_torch(
    dist: torch.Tensor, labels: torch.Tensor
) -> float:
    """Mean silhouette score from pre-computed distance matrix."""
    return float(_silhouette_samples_torch(dist, labels).mean().item())


def compute_nmi_ari(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    n_clusters: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute NMI and ARI using K-means clustering.

    Uses FAISS KMeans (CPU) for speed, falls back to sklearn if unavailable.

    Args:
        embeddings: (N, D) embedding tensor
        cell_type_labels: (N,) cell type labels
        n_clusters: number of clusters (defaults to number of unique labels)

    Returns:
        Dict with 'nmi' and 'ari' scores
    """
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Move to numpy
    emb_np = embeddings.detach().cpu().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()

    # Filter valid labels
    valid_mask = labels_np >= 0
    if valid_mask.sum() < 10:
        return {"nmi": 0.0, "ari": 0.0}

    emb_valid = emb_np[valid_mask]
    labels_valid = labels_np[valid_mask]

    # Determine n_clusters
    if n_clusters is None:
        n_clusters = len(np.unique(labels_valid))
    n_clusters = min(n_clusters, len(emb_valid) - 1)

    if n_clusters < 2:
        return {"nmi": 0.0, "ari": 0.0}

    # K-means clustering — prefer FAISS (faster), fall back to sklearn
    try:
        import faiss

        emb_f32 = np.ascontiguousarray(emb_valid, dtype=np.float32)
        km = faiss.Kmeans(
            d=emb_f32.shape[1], k=n_clusters,
            niter=20, nredo=1, seed=42, verbose=False,
        )
        km.train(emb_f32)
        _, assignments = km.index.search(emb_f32, 1)  # pyright: ignore[reportCallIssue]
        pred_labels = assignments.flatten()
    except ImportError:
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        pred_labels = kmeans.fit_predict(emb_valid)

    # Compute metrics (label comparison — fast on CPU)
    nmi = normalized_mutual_info_score(labels_valid, pred_labels)
    ari = adjusted_rand_score(labels_valid, pred_labels)

    return {"nmi": float(nmi), "ari": float(ari)}


def compute_asw_label(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
) -> float:
    """
    Compute Average Silhouette Width for cell type labels.
    Higher is better (cell types well separated).

    Uses torch.cdist on GPU for pairwise distances, falls back to sklearn on CPU.

    .. warning::
       **Scale change (2026-04):** this function now returns the value
       rescaled to [0, 1] via ``(asw + 1) / 2`` to match the scib-metrics
       convention.  Previously it returned the raw [-1, 1] silhouette.
       This is a **schema change** — historical WandB runs use the old
       [-1, 1] scale.  The raw value can still be obtained from
       ``_silhouette_score_torch``.

    Args:
        embeddings: (N, D) embedding tensor
        cell_type_labels: (N,) cell type labels

    Returns:
        Silhouette score in [0, 1] (float)
    """
    # Filter valid labels
    labels = cell_type_labels.detach().long()
    valid_mask = labels >= 0
    if valid_mask.sum() < 10:
        return 0.0

    emb_valid = embeddings.detach().float()[valid_mask]
    labels_valid = labels[valid_mask]

    n_unique = labels_valid.unique().numel()
    if n_unique < 2:
        return 0.0

    try:
        device = _get_compute_device()
        emb_gpu = emb_valid.to(device)
        labels_gpu = labels_valid.to(device)
        dist = torch.cdist(emb_gpu, emb_gpu)  # (N, N)
        raw_score = _silhouette_score_torch(dist, labels_gpu)
        return float((raw_score + 1.0) / 2.0)  # rescale [-1,1] → [0,1]
    except Exception:
        return 0.0


def compute_asw_batch(
    embeddings: torch.Tensor,
    batch_ids: torch.Tensor,
    cell_type_labels: torch.Tensor,
) -> float:
    """
    Compute batch ASW following scib definition.

    For each cell type, computes ASW on batch labels *within that cell type*.
    Uses absolute silhouette values scaled by subtracting from 1.
    Final score is averaged across cell types.

    Args:
        embeddings: (N, D) embedding tensor
        batch_ids: (N,) batch IDs
        cell_type_labels: (N,) cell type labels

    Returns:
        batchASW score (0-1 range):
        - 1: ideal batch mixing
        - 0: strongly separated batches
    """
    import numpy as np
    from sklearn.metrics import silhouette_samples

    # Move to numpy
    emb_np = embeddings.detach().cpu().numpy()
    batch_np = batch_ids.detach().cpu().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()

    # Filter valid cell type labels
    valid_mask = labels_np >= 0
    if valid_mask.sum() < 10:
        return 0.0

    emb_valid = emb_np[valid_mask]
    batch_valid = batch_np[valid_mask]
    labels_valid = labels_np[valid_mask]

    # Need at least 2 unique batches
    n_batches = len(np.unique(batch_valid))
    if n_batches < 2:
        return 1.0  # Only one batch = trivially "mixed"

    # Compute batchASW per cell type, then average
    unique_types = np.unique(labels_valid)
    batch_asw_per_type = []

    for ct in unique_types:
        ct_mask = labels_valid == ct
        ct_emb = emb_valid[ct_mask]
        ct_batches = batch_valid[ct_mask]

        # Need at least 2 cells and 2 batches within this cell type
        if len(ct_emb) < 2 or len(np.unique(ct_batches)) < 2:
            continue

        try:
            ct_sil = silhouette_samples(ct_emb, ct_batches)
        except Exception:
            continue

        # batchASW_j = 1 - mean(|s(i)|) for cells in cell type j
        batch_asw_j = 1.0 - np.mean(np.abs(ct_sil))  # pyright: ignore[reportCallIssue, reportArgumentType]
        batch_asw_per_type.append(batch_asw_j)

    if len(batch_asw_per_type) == 0:
        return 0.0

    # Final batchASW = average across cell types
    return float(np.mean(batch_asw_per_type))


def compute_bras(
    embeddings: torch.Tensor,
    batch_ids: torch.Tensor,
    cell_type_labels: torch.Tensor,
    between_cluster_distances: str = "mean_other",
) -> float:
    """
    Compute Batch-Removal-Adapted Silhouette (BRAS).

    BRAS fixes the 'nearest-cluster issue' of standard batch ASW by redefining
    b_i as the mean distance to ALL other clusters (not just the nearest).
    Uses cosine distance for higher discriminative power.

    Reference: Rautenstrauch & Ohler, Nature Biotechnology (2025)
    "Shortcomings of silhouette in single-cell integration benchmarking"
    Available in scib-metrics >= 0.5.5

    For each cell i in batch cluster C_k:
        a_i = mean cosine distance to all other cells in C_k
        b_i = mean cosine distance to all cells in ANY other cluster (BRAS default)
             OR mean cosine distance to FURTHEST other cluster (variant)
        s_i = (b_i - a_i) / max(a_i, b_i)

    Per cell type j: BRAS_j = mean(1 - |s_i|) for i in cell type j
    Final BRAS = mean(BRAS_j) across all cell types

    Args:
        embeddings: (N, D) embedding tensor
        batch_ids: (N,) batch IDs
        cell_type_labels: (N,) cell type labels
        between_cluster_distances: 'mean_other' (default, mean across ALL other clusters)
                                   or 'furthest' (distance to FURTHEST other cluster)

    Returns:
        BRAS score (0-1 range):
        - 1: ideal batch mixing
        - 0: strongly separated batches
    """
    import numpy as np

    # Move to numpy float32
    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    batch_np = batch_ids.detach().cpu().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()

    # Filter valid cell type labels
    valid_mask = labels_np >= 0
    if valid_mask.sum() < 10:
        return 0.0

    emb_valid = emb_np[valid_mask]
    batch_valid = batch_np[valid_mask]
    labels_valid = labels_np[valid_mask]

    # Need at least 2 unique batches
    unique_batches = np.unique(batch_valid)
    n_batches = len(unique_batches)
    if n_batches < 2:
        return 1.0

    # L2-normalize embeddings for cosine distance
    norms = np.linalg.norm(emb_valid, axis=1, keepdims=True) + 1e-8
    emb_normalized = emb_valid / norms

    # Compute BRAS per cell type, then average
    unique_types = np.unique(labels_valid)
    bras_per_type = []

    for ct in unique_types:
        ct_mask = labels_valid == ct
        ct_emb = emb_normalized[ct_mask]
        ct_batches = batch_valid[ct_mask]

        # Need at least 2 cells and 2 batches within this cell type
        ct_unique_batches = np.unique(ct_batches)
        if len(ct_emb) < 2 or len(ct_unique_batches) < 2:
            continue

        # Vectorized: compute full cosine distance matrix within this cell type
        cos_sim = ct_emb @ ct_emb.T  # (n_ct, n_ct)
        cos_dist = 1.0 - cos_sim

        n_ct = len(ct_emb)
        sil_values = np.zeros(n_ct, dtype=np.float64)
        valid_cells = np.ones(n_ct, dtype=bool)

        for i in range(n_ct):
            cell_batch = ct_batches[i]

            # a_i: mean cosine distance to same-batch cells (excluding self)
            same_mask = (ct_batches == cell_batch)
            same_mask[i] = False  # exclude self
            if same_mask.sum() == 0:
                valid_cells[i] = False
                continue

            a_i = cos_dist[i, same_mask].mean()

            # b_i: depends on between_cluster_distances mode
            if between_cluster_distances == "mean_other":
                other_mask = ct_batches != cell_batch
                if other_mask.sum() == 0:
                    valid_cells[i] = False
                    continue
                b_i = cos_dist[i, other_mask].mean()

            elif between_cluster_distances == "furthest":
                other_batches = ct_unique_batches[ct_unique_batches != cell_batch]
                batch_mean_dists = np.array([
                    cos_dist[i, ct_batches == ob].mean()
                    for ob in other_batches
                ])
                b_i = batch_mean_dists.max()

            else:
                raise ValueError(
                    f"Unknown between_cluster_distances: {between_cluster_distances}"
                )

            # Silhouette coefficient
            max_ab = max(a_i, b_i)
            if max_ab == 0:
                sil_values[i] = 0.0
            else:
                sil_values[i] = (b_i - a_i) / max_ab

        # Filter valid cells and compute BRAS_j
        valid_sil = sil_values[valid_cells]
        if len(valid_sil) == 0:
            continue

        # BRAS_j = mean(1 - |s_i|) for this cell type
        bras_j = np.mean(1.0 - np.abs(valid_sil))
        bras_per_type.append(bras_j)

    if len(bras_per_type) == 0:
        return 0.0

    return float(np.mean(bras_per_type))


def _perplexity_weighted_simpson(
    knn_dists: np.ndarray,
    knn_labels: np.ndarray,
    n_labels: int,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    """Compute perplexity-weighted Simpson index for each cell.

    Matches the scib-metrics / Korsunsky (2019) LISI definition:
    for each cell, find β via binary search so that the Shannon entropy of
    ``P = softmax(-β · dists)`` equals ``log(perplexity)``, then compute
    ``simpson = sum(bincount(labels, weights=P)²)``.

    Fully vectorized over cells (no Python loop per cell).

    Args:
        knn_dists: (N, k) distances to k nearest neighbors.
        knn_labels: (N, k) integer labels of k nearest neighbors.
        n_labels: total number of unique labels.
        perplexity: target perplexity for the soft neighbor distribution.
        tol: convergence tolerance for entropy binary search.
        max_iter: maximum binary search iterations.

    Returns:
        (N,) Simpson index per cell.  Cells where H=0 get simpson=-1.
    """
    import numpy as np

    N, K = knn_dists.shape
    log_perp = np.log(perplexity)

    # Initialize binary search variables (all cells in parallel)
    beta = np.ones(N, dtype=np.float64)
    betamin = np.full(N, -np.inf, dtype=np.float64)
    betamax = np.full(N, np.inf, dtype=np.float64)

    # Precompute one-hot for label bincount: (N, k) → (N, k, n_labels) is too large.
    # Instead, use scatter-add per iteration.

    converged_any = np.zeros(N, dtype=bool)

    for _ in range(max_iter):
        # P = exp(-beta * dist), (N, k)
        P = np.exp(-beta[:, None] * knn_dists)
        sumP = P.sum(axis=1, keepdims=True)  # (N, 1)
        sumP = np.maximum(sumP, 1e-16)

        # Entropy H = log(sumP) + beta * sum(dist * P) / sumP
        H = np.log(sumP[:, 0]) + beta * (knn_dists * P).sum(axis=1) / sumP[:, 0]

        # Check convergence (new this iteration)
        Hdiff = H - log_perp
        converged_this = np.abs(Hdiff) < tol

        # Freeze cells that have ever converged
        converged_any |= converged_this
        if converged_any.all():
            break

        # Binary search update (only for not-yet-converged cells)
        too_high = Hdiff > 0
        betamin_new = np.where(too_high, beta, betamin)
        betamax_new = np.where(too_high, betamax, beta)

        new_beta_high = np.where(
            betamax_new == np.inf, beta * 2, (beta + betamax_new) / 2
        )
        new_beta_low = np.where(
            betamin_new == -np.inf, beta / 2, (beta + betamin_new) / 2
        )
        beta_new = np.where(too_high, new_beta_high, new_beta_low)

        # Only update non-converged cells
        update_mask = ~converged_any
        beta = np.where(update_mask, beta_new, beta)
        betamin = np.where(update_mask, betamin_new, betamin)
        betamax = np.where(update_mask, betamax_new, betamax)

    # Final P with converged beta
    P = np.exp(-beta[:, None] * knn_dists)
    sumP = np.maximum(P.sum(axis=1, keepdims=True), 1e-16)
    H_final = np.log(sumP[:, 0]) + beta * (knn_dists * P).sum(axis=1) / sumP[:, 0]
    P = P / sumP

    # Simpson index: sum(bincount(labels, weights=P)²) per cell
    # Vectorized via scatter-add
    simpson = np.full(N, -1.0, dtype=np.float64)
    valid = H_final != 0

    if valid.any():
        # Build per-cell label sums: (N_valid, n_labels)
        P_valid = P[valid]  # (N_valid, k)
        labels_valid = knn_labels[valid]  # (N_valid, k)
        N_valid = P_valid.shape[0]

        # Vectorized bincount via advanced indexing
        label_sums = np.zeros((N_valid, n_labels), dtype=np.float64)
        cell_idx = np.arange(N_valid)[:, None]  # (N_valid, 1)
        np.add.at(label_sums, (cell_idx, labels_valid), P_valid)

        simpson[valid] = (label_sums ** 2).sum(axis=1)

    return simpson


def compute_ilisi(
    embeddings: torch.Tensor,
    batch_ids: torch.Tensor,
    k: int = 30,
) -> float:
    """
    Compute integration LISI (Local Inverse Simpson Index), normalized to 0-1.
    Higher is better (more batch mixing).

    Uses perplexity-weighted Simpson index matching scib-metrics / Korsunsky
    (2019).  FAISS GPU k-NN for speed, vectorized numpy binary search for
    perplexity (no JAX dependency, no per-cell Python loop).

    Args:
        embeddings: (N, D) embedding tensor
        batch_ids: (N,) batch IDs
        k: number of nearest neighbors

    Returns:
        Normalized iLISI score (0-1 range):
        - 0: no batch mixing (all neighbors from same batch)
        - 1: perfect batch mixing (equal representation of all batches)
    """
    import numpy as np

    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    batch_np = batch_ids.detach().cpu().numpy()

    _, batch_np_remapped = np.unique(batch_np, return_inverse=True)
    n_batches = int(batch_np_remapped.max()) + 1
    if n_batches < 2:
        return 1.0

    k_actual = min(k, len(emb_np) - 1)
    if k_actual < 1:
        return 1.0

    # k-NN via FAISS (GPU-accelerated)
    distances, indices = _faiss_knn(emb_np, k=k_actual, metric="euclidean")

    # Perplexity: floor(k / 3), matching scib-metrics default
    perplexity = max(1.0, float(np.floor(k_actual / 3)))

    # Perplexity-weighted Simpson index (vectorized, no JAX)
    neighbor_labels = batch_np_remapped[indices]  # (N, k)
    simpson = _perplexity_weighted_simpson(
        distances, neighbor_labels, n_batches, perplexity,
    )

    # LISI = 1 / simpson
    valid = simpson > 0
    lisi_scores = np.full(len(simpson), np.nan)
    lisi_scores[valid] = 1.0 / simpson[valid]

    median_ilisi = float(np.nanmedian(lisi_scores))

    # Normalize: (ilisi - 1) / (n_batches - 1), matching scib convention
    normalized_ilisi = (median_ilisi - 1.0) / (n_batches - 1.0)
    return max(0.0, min(1.0, normalized_ilisi))


def compute_clisi(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    k: int = 30,
) -> float:
    """Compute cell-type LISI (cLISI), normalized to 0-1.

    Higher is better (cell types well separated in neighborhood).
    Uses the same perplexity-weighted Simpson as iLISI, but applied to
    cell-type labels instead of batch labels.  Normalization follows the
    scib-metrics convention: ``(n_labels - median_lisi) / (n_labels - 1)``.

    Args:
        embeddings: (N, D) embedding tensor.
        cell_type_labels: (N,) cell-type labels.
        k: number of nearest neighbors.

    Returns:
        Normalized cLISI in [0, 1].
    """
    import numpy as np

    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    label_np = cell_type_labels.detach().cpu().numpy()

    _, label_remapped = np.unique(label_np, return_inverse=True)
    n_labels = int(label_remapped.max()) + 1
    if n_labels < 2:
        return 1.0

    k_actual = min(k, len(emb_np) - 1)
    if k_actual < 1:
        return 1.0

    distances, indices = _faiss_knn(emb_np, k=k_actual, metric="euclidean")
    perplexity = max(1.0, float(np.floor(k_actual / 3)))

    neighbor_labels = label_remapped[indices]
    simpson = _perplexity_weighted_simpson(
        distances, neighbor_labels, n_labels, perplexity,
    )

    valid = simpson > 0
    lisi_scores = np.full(len(simpson), np.nan)
    lisi_scores[valid] = 1.0 / simpson[valid]

    median_clisi = float(np.nanmedian(lisi_scores))

    # scib convention: (n_labels - clisi) / (n_labels - 1)
    normalized = (n_labels - median_clisi) / (n_labels - 1.0)
    return max(0.0, min(1.0, normalized))


def compute_isolated_labels(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    batch_ids: torch.Tensor,
    iso_threshold: Optional[int] = None,
) -> float:
    """Compute isolated label score via silhouette on rare cell types.

    Identifies cell types that appear in few batches (``<= iso_threshold``),
    then computes the mean per-sample silhouette (rescaled to [0,1]) for
    those types.  Matches the scib-metrics ``isolated_labels`` definition.

    Args:
        embeddings: (N, D) embedding tensor.
        cell_type_labels: (N,) cell-type labels.
        batch_ids: (N,) batch IDs.
        iso_threshold: max batch count for a label to be "isolated".
            ``None`` = minimum batch count across all labels.

    Returns:
        Isolated label score in [0, 1].  Returns 0 if no isolated labels.
    """
    import numpy as np

    emb_np = embeddings.detach().cpu().float().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()
    batch_np = batch_ids.detach().cpu().numpy()

    # Find isolated labels: types present in <= iso_threshold batches
    unique_labels = np.unique(labels_np)
    batch_per_label = {}
    for lbl in unique_labels:
        n_batches_for_lbl = len(np.unique(batch_np[labels_np == lbl]))
        batch_per_label[lbl] = n_batches_for_lbl

    if iso_threshold is None:
        iso_threshold = min(batch_per_label.values()) if batch_per_label else 1

    isolated = [lbl for lbl, nb in batch_per_label.items() if nb <= iso_threshold]
    if len(isolated) == 0:
        return 0.0

    # Compute silhouette samples (GPU-accelerated)
    device = _get_compute_device()
    emb_t = torch.from_numpy(emb_np).to(device)
    labels_t = torch.from_numpy(labels_np).long().to(device)

    n_unique = labels_t.unique().numel()
    if n_unique < 2:
        return 0.0

    dist = torch.cdist(emb_t, emb_t)
    sil_samples = _silhouette_samples_torch(dist, labels_t)  # [-1, 1]
    sil_rescaled = ((sil_samples + 1.0) / 2.0).cpu().numpy()  # [0, 1]

    # Mean silhouette for each isolated label
    scores = []
    for lbl in isolated:
        mask = labels_np == lbl
        if mask.sum() > 0:
            scores.append(float(sil_rescaled[mask].mean()))

    return float(np.mean(scores)) if scores else 0.0


def compute_scgraph(
    gene_expression,
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    batch_ids: torch.Tensor,
    trim_rate: float = 0.05,
    thres_batch: int = 100,
    thres_celltype: int = 10,
    n_hvg: int = 1000,
    n_pca_comps: int = 10,
) -> float:
    """Compute scGraph Corr-Weights score (Islander, Nature Biotech 2025).

    Measures whether an embedding preserves the inter-cell-type distance
    structure observed in gene expression PCA space.  A reference distance
    matrix is built per-batch from PCA on HVGs, then compared to the
    embedding's distance matrix via per-cell-type weighted Pearson
    correlation (weight = 1 / reference_distance).

    Higher is better (embedding preserves biological distance structure).

    Args:
        gene_expression: (N, G) gene expression matrix (log1p-normalized or
            raw counts — HVG selection + PCA are applied internally).
        embeddings: (N, D) embedding vectors to evaluate.
        cell_type_labels: (N,) cell-type labels.
        batch_ids: (N,) batch IDs.
        trim_rate: fraction to trim from each tail for robust centroids.
        thres_batch: minimum cells per batch to include.
        thres_celltype: minimum total cells for a cell type to include.
        n_hvg: number of highly variable genes for PCA reference.
        n_pca_comps: number of PCA components for reference.

    Returns:
        Corr-Weights score in approximately [-1, 1] (typically 0.3–0.95).
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.stats import trim_mean

    if isinstance(gene_expression, torch.Tensor):
        expr_np = gene_expression.detach().cpu().numpy().astype(np.float32)
    else:
        import scipy.sparse as sp

        if sp.issparse(gene_expression):
            expr_np = np.asarray(gene_expression.toarray(), dtype=np.float32)
        else:
            expr_np = np.asarray(gene_expression, dtype=np.float32)
    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    labels_np = cell_type_labels.detach().cpu().numpy()
    batch_np = batch_ids.detach().cpu().numpy()

    unique_labels = np.unique(labels_np)
    # Filter rare cell types
    label_counts = {lbl: int((labels_np == lbl).sum()) for lbl in unique_labels}
    valid_labels = np.array([lbl for lbl, c in label_counts.items() if c >= thres_celltype])
    if len(valid_labels) < 2:
        return 0.0

    # --- Phase 1: Build reference distance matrix from gene expression PCA ---
    unique_batches = np.unique(batch_np)
    batch_dist_frames: list[dict] = []

    for batch_val in unique_batches:
        batch_mask = batch_np == batch_val
        if batch_mask.sum() < thres_batch:
            continue

        expr_batch = expr_np[batch_mask]
        labels_batch = labels_np[batch_mask]

        # HVG selection: top-n_hvg by variance
        gene_var = np.var(expr_batch, axis=0)
        if len(gene_var) > n_hvg:
            hvg_idx = np.argsort(gene_var)[-n_hvg:]
        else:
            hvg_idx = np.arange(len(gene_var))
        expr_hvg = expr_batch[:, hvg_idx]

        # PCA via sklearn (no scanpy dependency)
        from sklearn.decomposition import PCA
        n_comps = min(n_pca_comps, expr_hvg.shape[0] - 1, expr_hvg.shape[1])
        if n_comps < 2:
            continue
        pca_coords = PCA(n_components=n_comps).fit_transform(expr_hvg)

        # Trimmed-mean centroids per cell type
        centroids = {}
        for lbl in valid_labels:
            lbl_mask = labels_batch == lbl
            if lbl_mask.sum() < 2:
                continue
            centroids[lbl] = trim_mean(pca_coords[lbl_mask], proportiontocut=trim_rate, axis=0)

        if len(centroids) < 2:
            continue

        # Pairwise distance matrix
        ct_order = sorted(centroids.keys(), key=lambda x: int(x) if isinstance(x, (int, np.integer)) else x)
        centroid_arr = np.array([centroids[lbl] for lbl in ct_order])
        dist_mat = cdist(centroid_arr, centroid_arr, metric="euclidean")

        # Column-normalize
        col_max = dist_mat.max(axis=0, keepdims=True)
        col_max = np.maximum(col_max, 1e-10)
        dist_mat = dist_mat / col_max

        batch_dist_frames.append({"labels": ct_order, "dist": dist_mat})

    if not batch_dist_frames:
        return 0.0

    # Aggregate across batches: mean of normalized distances
    all_labels_set = set()
    for frame in batch_dist_frames:
        all_labels_set.update(frame["labels"])
    all_labels = sorted(all_labels_set, key=lambda x: int(x) if isinstance(x, (int, np.integer)) else x)
    K = len(all_labels)
    if K < 2:
        return 0.0
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    consensus_sum = np.zeros((K, K), dtype=np.float64)
    consensus_count = np.zeros((K, K), dtype=np.float64)

    for frame in batch_dist_frames:
        for i, li in enumerate(frame["labels"]):
            for j, lj in enumerate(frame["labels"]):
                gi, gj = label_to_idx[li], label_to_idx[lj]
                consensus_sum[gi, gj] += frame["dist"][i, j]
                consensus_count[gi, gj] += 1.0

    valid_entries = consensus_count > 0
    consensus = np.zeros((K, K), dtype=np.float64)
    consensus[valid_entries] = consensus_sum[valid_entries] / consensus_count[valid_entries]

    # Column-normalize consensus
    col_max = consensus.max(axis=0, keepdims=True)
    col_max = np.maximum(col_max, 1e-10)
    consensus = consensus / col_max

    # --- Phase 2: Embedding distance matrix ---
    # Only keep labels with >=2 cells in the embedding (centroid is defined).
    # Labels missing from the embedding are dropped from BOTH the reference
    # consensus and the embedding distance matrix — avoids zero-vector
    # pollution that would create spurious huge distances.
    emb_centroids = {}
    for lbl in all_labels:
        lbl_mask = labels_np == lbl
        if lbl_mask.sum() < 2:
            continue
        emb_centroids[lbl] = trim_mean(emb_np[lbl_mask], proportiontocut=trim_rate, axis=0)

    if len(emb_centroids) < 2:
        return 0.0

    # Rebuild the label list to only include labels with valid centroids.
    kept_labels = [lbl for lbl in all_labels if lbl in emb_centroids]
    kept_idx = [label_to_idx[lbl] for lbl in kept_labels]
    K_kept = len(kept_labels)
    if K_kept < 2:
        return 0.0

    # Subset consensus matrix to kept labels
    consensus = consensus[np.ix_(kept_idx, kept_idx)]
    consensus_count = consensus_count[np.ix_(kept_idx, kept_idx)]

    emb_centroid_arr = np.array([emb_centroids[lbl] for lbl in kept_labels])
    emb_dist = cdist(emb_centroid_arr, emb_centroid_arr, metric="euclidean")
    col_max = emb_dist.max(axis=0, keepdims=True)
    col_max = np.maximum(col_max, 1e-10)
    emb_dist = emb_dist / col_max

    # --- Phase 3: Weighted Pearson correlation per cell type ---
    corr_scores = []
    for c in range(K_kept):
        x = emb_dist[:, c]
        y = consensus[:, c]

        # Skip if insufficient valid entries
        valid = (consensus_count[:, c] > 0) & np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue

        xv, yv = x[valid], y[valid]

        # Weights = 1 / reference_distance (self-distance = 0 → weight = 0)
        with np.errstate(divide="ignore"):
            w = np.where(yv > 1e-10, 1.0 / yv, 0.0)
        w_sum = w.sum()
        if w_sum < 1e-10:
            continue
        w = w / w_sum

        # Weighted Pearson
        mu_x = np.sum(w * xv)
        mu_y = np.sum(w * yv)
        dx = xv - mu_x
        dy = yv - mu_y
        cov = np.sum(w * dx * dy)
        var_x = np.sum(w * dx ** 2)
        var_y = np.sum(w * dy ** 2)
        denom = np.sqrt(var_x * var_y)
        if denom < 1e-10:
            continue
        corr_scores.append(float(cov / denom))

    if not corr_scores:
        return 0.0

    return float(np.mean(corr_scores))


def _compute_graph_connectivity(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    k: int = 30,
) -> float:
    """
    Compute graph connectivity score.
    Measures how well same cell types are connected across batches.

    Args:
        embeddings: (N, D) embedding tensor
        cell_type_labels: (N,) cell type labels
        k: number of nearest neighbors

    Returns:
        Graph connectivity score (float)
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # Move to numpy
    emb_np = embeddings.detach().cpu().numpy()
    labels_np = cell_type_labels.detach().cpu().numpy()

    # Filter valid
    valid_mask = labels_np >= 0
    if valid_mask.sum() < 10:
        return 1.0

    emb_valid = emb_np[valid_mask]
    labels_valid = labels_np[valid_mask]

    unique_types = np.unique(labels_valid)
    if len(unique_types) < 2:
        return 1.0

    # Get k-NN
    _, indices = _faiss_knn(emb_valid, k=k)

    # For each cell type, check connectivity
    connectivity_scores = []

    for ct in unique_types:
        ct_mask = labels_valid == ct
        ct_indices = np.where(ct_mask)[0]

        if len(ct_indices) < 2:
            connectivity_scores.append(1.0)
            continue

        # Build adjacency matrix for this cell type
        n_ct = len(ct_indices)
        ct_index_map = {idx: i for i, idx in enumerate(ct_indices)}

        # Create sparse adjacency
        rows, cols = [], []
        for local_i, global_i in enumerate(ct_indices):
            for neighbor in indices[global_i]:
                if neighbor in ct_index_map:
                    local_j = ct_index_map[neighbor]
                    rows.append(local_i)
                    cols.append(local_j)

        if len(rows) == 0:
            connectivity_scores.append(0.0)
            continue

        adj = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_ct, n_ct)
        )

        # Count connected components
        n_components, _ = connected_components(adj, directed=False)
        # Connectivity = 1 / n_components (1 component = fully connected = 1.0)
        connectivity_scores.append(1.0 / n_components)

    return float(np.mean(connectivity_scores))


def compute_bio_conservation_metrics(
    embeddings: torch.Tensor,
    cell_type_labels: torch.Tensor,
    batch_ids: torch.Tensor,
    subsample: bool = True,
    max_per_type: int = 500,
    total_max: int = 10000,
    _use_faiss: bool = True,
    k: int = 30,
    compute_graph_connectivity: bool = False,
    gene_expression=None,
) -> Dict[str, Optional[float]]:
    """
    Compute comprehensive bio-conservation metrics.

    Args:
        embeddings: (N, D) embedding tensor (e.g., CLS token z_1[:, 0, :])
        cell_type_labels: (N,) cell type labels
        batch_ids: (N,) batch IDs (e.g., dataset_id + donor_id)
        subsample: whether to use stratified subsampling
        max_per_type: max cells per type for subsampling
        total_max: total max cells for subsampling
        use_faiss: whether to use FAISS for k-NN (fallback to sklearn if not available)
        k: number of nearest neighbors for LISI and graph connectivity
        compute_graph_connectivity: whether to compute graph connectivity (expensive, default False)

    Returns:
        Dict with bio-conservation and batch-correction metrics:

        Bio conservation (cell type structure):
        - nmi: Normalized Mutual Information (K-means vs true labels)
        - ari: Adjusted Rand Index
        - asw_label: Silhouette Width for cell types, [0,1]
        - clisi: Cell-type LISI (cell type purity in neighborhoods)
        - isolated_labels: ASW for rare cell types

        Batch correction (batch mixing):
        - bras: Batch-Removal-Adapted Silhouette (robust to nested batches)
        - ilisi: Integration LISI (batch mixing in neighborhoods)
        - graph_connectivity: k-NN graph connectivity per cell type (optional)
    """
    # Ensure float32 for sklearn/FAISS compatibility (BFloat16 not supported)
    embeddings = embeddings.float()

    # Subsample if needed (also subsample gene_expression to keep rows aligned)
    if subsample and len(embeddings) > total_max:
        result = stratified_subsample(
            embeddings,
            cell_type_labels,
            batch_ids,
            max_per_type=max_per_type,
            total_max=total_max,
            return_indices=True,
        )
        embeddings, cell_type_labels, batch_ids, sampled_idx = result
        if gene_expression is not None and sampled_idx is not None:
            if isinstance(gene_expression, torch.Tensor):
                gene_expression = gene_expression[torch.from_numpy(sampled_idx).long()]
            else:
                gene_expression = gene_expression[sampled_idx]

    metrics: Dict[str, Optional[float]] = {}

    # --- Bio conservation ---

    # Clustering metrics (NMI, ARI)
    try:
        clustering_metrics = compute_nmi_ari(embeddings, cell_type_labels)
        metrics.update(clustering_metrics)
    except Exception:
        metrics["nmi"] = None
        metrics["ari"] = None

    # ASW for cell types
    try:
        metrics["asw_label"] = compute_asw_label(embeddings, cell_type_labels)
    except Exception:
        metrics["asw_label"] = None

    # cLISI (cell type purity)
    try:
        metrics["clisi"] = compute_clisi(embeddings, cell_type_labels, k=k)
    except Exception:
        metrics["clisi"] = None

    # Isolated labels (rare cell type separation)
    try:
        metrics["isolated_labels"] = compute_isolated_labels(
            embeddings, cell_type_labels, batch_ids
        )
    except Exception:
        metrics["isolated_labels"] = None

    # --- Batch correction ---

    # BRAS (Batch-Removal-Adapted Silhouette)
    try:
        metrics["bras"] = compute_bras(embeddings, batch_ids, cell_type_labels)
    except Exception:
        metrics["bras"] = None

    # iLISI (batch mixing)
    try:
        metrics["ilisi"] = compute_ilisi(embeddings, batch_ids, k=k)
    except Exception:
        metrics["ilisi"] = None

    # scGraph: inter-cell-type distance structure preservation
    if gene_expression is not None:
        try:
            # scGraph densifies expression for HVG/PCA reference construction,
            # so use an independent cap instead of reusing the main bio-metric
            # subsample size.
            scgraph_embeddings = embeddings
            scgraph_labels = cell_type_labels
            scgraph_batch_ids = batch_ids
            scgraph_gene_expression = gene_expression
            scgraph_total_max = _env_int("SCTRILEMMA_SCGRAPH_TOTAL_MAX", 5000)
            scgraph_max_per_type = _env_int("SCTRILEMMA_SCGRAPH_MAX_PER_TYPE", 250)
            if scgraph_total_max > 0 and len(scgraph_embeddings) > scgraph_total_max:
                (
                    scgraph_embeddings,
                    scgraph_labels,
                    scgraph_batch_ids,
                    scgraph_idx,
                ) = stratified_subsample(
                    scgraph_embeddings,
                    scgraph_labels,
                    scgraph_batch_ids,
                    max_per_type=scgraph_max_per_type,
                    total_max=scgraph_total_max,
                    return_indices=True,
                )
                if scgraph_idx is not None:
                    if isinstance(scgraph_gene_expression, torch.Tensor):
                        idx_tensor = torch.from_numpy(scgraph_idx).long()
                        scgraph_gene_expression = scgraph_gene_expression[idx_tensor]
                    else:
                        scgraph_gene_expression = scgraph_gene_expression[scgraph_idx]
                    print(
                        "  Subsampled "
                        f"{len(embeddings):,} -> {len(scgraph_embeddings):,} "
                        "cells for scGraph"
                    )

            scgraph_max_dense_gb = _env_float("SCTRILEMMA_SCGRAPH_MAX_DENSE_GB", 8.0)
            estimated_dense_gb = _estimated_dense_float32_gb(scgraph_gene_expression)
            if estimated_dense_gb > scgraph_max_dense_gb:
                print(
                    "Warning: scGraph skipped: estimated dense float32 matrix "
                    f"{estimated_dense_gb:.2f} GB exceeds budget "
                    f"{scgraph_max_dense_gb:.2f} GB"
                )
                metrics["scgraph"] = None
            else:
                metrics["scgraph"] = compute_scgraph(
                    scgraph_gene_expression,
                    scgraph_embeddings,
                    scgraph_labels,
                    scgraph_batch_ids,
                )
        except Exception:
            metrics["scgraph"] = None

    # Graph connectivity (optional, expensive)
    if compute_graph_connectivity:
        try:
            metrics["graph_connectivity"] = _compute_graph_connectivity(
                embeddings, cell_type_labels, k=k
            )
        except Exception:
            metrics["graph_connectivity"] = None

    return metrics


# ---------------------------------------------------------------------------
# Generation-metric stubs for sctrilemma/benchmark/metrics.py import compatibility.
# These are referenced by `sctrilemma/benchmark/metrics.py` as
# `_wasserstein2_gpu`, `_frechet_distance_gpu`, etc. They should never be called
# when `--disable-reconstruction` is set; calling them raises NotImplementedError
# so accidental use fails loudly instead of silently returning wrong numbers.
# Generation benchmark proper should land these via the generation plan.
# ---------------------------------------------------------------------------


def wasserstein2_dense(*args, **kwargs):
    raise NotImplementedError(
        "wasserstein2_dense is a stub. Use --disable-reconstruction or add a "
        "generation-distance implementation before enabling this metric."
    )


def frechet_distance_dense(*args, **kwargs):
    raise NotImplementedError("frechet_distance_dense is a stub.")


def compute_cell_distance_ks(*args, **kwargs):
    raise NotImplementedError("compute_cell_distance_ks is a stub.")


def compute_cell_detect_freq_ks(*args, **kwargs):
    raise NotImplementedError("compute_cell_detect_freq_ks is a stub.")


def _compute_wasserstein2(*args, **kwargs):
    raise NotImplementedError("_compute_wasserstein2 is a stub.")


def _compute_frechet_distance(*args, **kwargs):
    raise NotImplementedError("_compute_frechet_distance is a stub.")


def _compute_joint_distribution_metrics(*args, **kwargs):
    raise NotImplementedError("_compute_joint_distribution_metrics is a stub.")
