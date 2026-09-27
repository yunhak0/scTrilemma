from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import AttentionPooling, GeneEmbeddingNetwork


class _SequenceBatchNorm(nn.Module):
    """BatchNorm over the feature axis for (B, T, D) tensors."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class _FeedForward(nn.Module):
    """SwiGLU feed-forward block used by encoder and decoder blocks."""

    def __init__(self, d_model: int, dropout: float, hidden_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(d_model * hidden_mult)
        self.value_proj = nn.Linear(d_model, hidden_dim)
        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value_proj(x)
        gate = F.silu(self.gate_proj(x))
        return self.dropout(self.out_proj(self.dropout(value * gate)))


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention followed by a SwiGLU feed-forward block."""

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.RMSNorm(d_model)
        self.norm_kv = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ff = nn.RMSNorm(d_model)
        self.ff = _FeedForward(d_model, dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = None if context_mask is None else ~context_mask
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        attn_out, _ = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = queries + attn_out
        x = x + self.ff(self.norm_ff(x))
        if query_mask is not None:
            x = x * query_mask.unsqueeze(-1).to(x.dtype)
        return x


class ScTrilemmaVAE(nn.Module):
    """Latent-bottleneck VAE used by the submitted scTrilemma checkpoint.

    The official implementation keeps the active model path only:
    expression-gated gene tokens, a Perceiver-style cross-attention bottleneck,
    C-Route decoder-query modulation, and pseudo-bulk-derived prior conditioning.
    """

    def __init__(
        self,
        d_model: int,
        num_latent_tokens: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        encoder_heads: int,
        decoder_heads: int,
        dropout: float,
        vocab_size: int,
        repr_pooling: str,
        logvar_min: float,
        logvar_max: float,
        zi_return_logits: bool,
        mu_activation_mode: str,
        expr_proj_mode: Literal["linear", "multiplicative"] = "linear",
        tissue_code_dim: int = 0,
        tissue_code_embedding_mode: Literal["soft", "hard"] = "soft",
        tissue_prior_mode: Literal["", "diagonal"] = "",
        tissue_prior_mu_scale: float = 1.0,
        tissue_prior_logvar_mode: Literal["learned", "zero"] = "learned",
        decoder_query_mode: Literal["off", "z_added", "z_modulated"] = "off",
        decoder_query_pooling: Literal[
            "same",
            "mean",
            "attention",
            "mean_attn_residual",
        ] = "same",
        decoder_query_residual_alpha: float = 0.2,
        repr_pooling_residual_alpha: float = 0.2,
        **unused: object,
    ):
        super().__init__()
        if unused:
            unsupported = ", ".join(sorted(unused))
            raise TypeError(f"Unsupported ScTrilemmaVAE arguments: {unsupported}")
        if num_encoder_layers < 1 or num_decoder_layers < 1:
            raise ValueError("num_encoder_layers and num_decoder_layers must be >= 1")
        if mu_activation_mode not in {
            "softmax",
            "normalized_softplus",
            "softplus",
            "log_softmax",
            "exp",
        }:
            raise ValueError(f"Unsupported mu_activation_mode: {mu_activation_mode}")
        if repr_pooling not in {"mean", "attention", "mean_attn_residual", "cls"}:
            raise ValueError(f"Unsupported repr_pooling: {repr_pooling}")
        if expr_proj_mode not in {"linear", "multiplicative"}:
            raise ValueError(f"Unsupported expr_proj_mode: {expr_proj_mode}")
        if tissue_code_embedding_mode not in {"soft", "hard"}:
            raise ValueError(
                f"Unsupported tissue_code_embedding_mode: {tissue_code_embedding_mode}"
            )
        if tissue_prior_mode not in {"", "diagonal"}:
            raise ValueError(f"Unsupported tissue_prior_mode: {tissue_prior_mode}")
        if tissue_prior_logvar_mode not in {"learned", "zero"}:
            raise ValueError(
                f"Unsupported tissue_prior_logvar_mode: {tissue_prior_logvar_mode}"
            )
        if decoder_query_mode not in {"off", "z_added", "z_modulated"}:
            raise ValueError(f"Unsupported decoder_query_mode: {decoder_query_mode}")
        if decoder_query_pooling not in {
            "same",
            "mean",
            "attention",
            "mean_attn_residual",
        }:
            raise ValueError(
                f"Unsupported decoder_query_pooling: {decoder_query_pooling}"
            )

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.repr_pooling = repr_pooling
        self.repr_pooling_residual_alpha = float(repr_pooling_residual_alpha)
        self.decoder_query_mode = decoder_query_mode
        self.decoder_query_pooling = decoder_query_pooling
        self.decoder_query_residual_alpha = float(decoder_query_residual_alpha)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.zi_return_logits = bool(zi_return_logits)
        self.mu_activation_mode = mu_activation_mode
        self.expr_proj_mode = expr_proj_mode
        self.tissue_code_dim = int(tissue_code_dim)
        self.tissue_code_embedding_mode = tissue_code_embedding_mode
        self.tissue_prior_mode = tissue_prior_mode
        self.tissue_prior_mu_scale = float(tissue_prior_mu_scale)
        self.tissue_prior_logvar_mode = tissue_prior_logvar_mode
        self.eps = 1e-8

        self.gene_embedding = GeneEmbeddingNetwork(
            d_model=d_model,
            vocab_size=vocab_size,
            embedding_mode="learnable",
        )
        self.raw_proj = nn.Linear(1, d_model)
        self.expr_mod: nn.Linear | None = None
        if expr_proj_mode == "multiplicative":
            self.expr_mod = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.expr_mod.bias)

        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latent_tokens, d_model) * 0.02
        )
        self.encoder_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(d_model, encoder_heads, dropout)
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_output_bn = _SequenceBatchNorm(d_model)
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

        self.repr_pooler: AttentionPooling | None = None
        if repr_pooling in {"attention", "mean_attn_residual"}:
            self.repr_pooler = AttentionPooling(d_model, num_heads=encoder_heads)

        self.decoder_query_pooler: AttentionPooling | None = None
        if decoder_query_pooling in {"attention", "mean_attn_residual"}:
            self.decoder_query_pooler = AttentionPooling(
                d_model,
                num_heads=encoder_heads,
            )

        self.z_query_gate: nn.Sequential | None = None
        if decoder_query_mode == "z_modulated":
            self.z_query_gate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(d_model, decoder_heads, dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, 2)
        self.px_r = nn.Parameter(torch.zeros(vocab_size))

        self.tissue_prior_mu: nn.Module | None = None
        self.tissue_prior_logvar: nn.Module | None = None
        if self.tissue_code_dim > 0 and tissue_prior_mode == "diagonal":
            rng_state = torch.random.get_rng_state()
            if tissue_code_embedding_mode == "hard":
                self.tissue_prior_mu = nn.Embedding(self.tissue_code_dim, d_model)
                if tissue_prior_logvar_mode == "learned":
                    self.tissue_prior_logvar = nn.Embedding(
                        self.tissue_code_dim,
                        d_model,
                    )
            else:
                self.tissue_prior_mu = nn.Linear(self.tissue_code_dim, d_model)
                if tissue_prior_logvar_mode == "learned":
                    self.tissue_prior_logvar = nn.Linear(
                        self.tissue_code_dim,
                        d_model,
                    )
            if self.tissue_prior_logvar is not None:
                nn.init.zeros_(self.tissue_prior_logvar.weight)
                if isinstance(self.tissue_prior_logvar, nn.Linear):
                    nn.init.zeros_(self.tissue_prior_logvar.bias)
            nn.init.normal_(self.tissue_prior_mu.weight, std=0.01)
            if isinstance(self.tissue_prior_mu, nn.Linear):
                nn.init.zeros_(self.tissue_prior_mu.bias)
            torch.random.set_rng_state(rng_state)

        self._last_prior_mu: torch.Tensor | None = None
        self._last_prior_logvar: torch.Tensor | None = None
        self._last_expr_mask: torch.Tensor | None = None
        self._last_context_stats: dict[str, torch.Tensor] = {}

        # Compatibility placeholders for disabled research losses in the Lightning module.
        self.pb_probe_head: nn.Module | None = None
        self.pb_vib_mu_proj: nn.Module | None = None
        self.pb_vib_logvar_proj: nn.Module | None = None
        self.tissue_aux_head: nn.Module | None = None
        self.pb_orth_residual = False
        self.pb_orth_residual_mode = ""

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def _build_gene_embeddings(
        self,
        meta_features: Dict[str, torch.Tensor],
        gene_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del meta_features
        return self.gene_embedding(gene_indices=gene_indices)

    def _project_tissue_code(
        self,
        projector: nn.Module,
        tissue_code: torch.Tensor,
    ) -> torch.Tensor:
        if self.tissue_code_embedding_mode == "hard":
            if tissue_code.dim() != 2:
                raise ValueError(
                    "hard tissue_code_embedding_mode expects a 2D soft code"
                )
            return projector(tissue_code.argmax(dim=-1).long())
        return projector(tissue_code)

    def encode(
        self,
        meta_features: Optional[Dict[str, torch.Tensor]],
        raw_input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gene_indices: Optional[torch.Tensor] = None,
        pseudo_bulk: Optional[torch.Tensor] = None,
        pseudo_bulk_values: Optional[torch.Tensor] = None,
        tissue_code: Optional[torch.Tensor] = None,
        context_group_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        del pseudo_bulk, pseudo_bulk_values, context_group_labels
        meta_features = meta_features or {}
        gene_embs = self._build_gene_embeddings(meta_features, gene_indices)
        raw_emb = self.raw_proj(raw_input.unsqueeze(-1))

        if self.expr_proj_mode == "multiplicative":
            if self.expr_mod is None:
                raise RuntimeError("expr_mod must be initialized for multiplicative mode")
            gene_tokens = gene_embs * torch.sigmoid(self.expr_mod(raw_emb))
        else:
            gene_tokens = gene_embs + raw_emb

        if mask is not None:
            gene_tokens = gene_tokens * mask.unsqueeze(-1).to(gene_tokens.dtype)

        batch_size = gene_tokens.size(0)
        latent = self.latent_queries.expand(batch_size, -1, -1)
        for block in self.encoder_blocks:
            latent = block(latent, gene_tokens, context_mask=mask)
        latent = self.encoder_output_bn(latent)

        vae_mu = self.mu_proj(latent)
        vae_logvar = self.logvar_proj(latent).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        z = self._reparameterize(vae_mu, vae_logvar)

        self._last_prior_mu = None
        self._last_prior_logvar = None
        if (
            self.tissue_prior_mode == "diagonal"
            and self.tissue_prior_mu is not None
            and tissue_code is not None
        ):
            prior_mu = (
                self.tissue_prior_mu_scale
                * self._project_tissue_code(self.tissue_prior_mu, tissue_code).unsqueeze(1)
            )
            if self.tissue_prior_logvar is None:
                prior_logvar = torch.zeros_like(prior_mu)
            else:
                prior_logvar = self._project_tissue_code(
                    self.tissue_prior_logvar,
                    tissue_code,
                ).unsqueeze(1)
                prior_logvar = prior_logvar.clamp(
                    min=self.logvar_min,
                    max=self.logvar_max,
                )
            self._last_prior_mu = prior_mu.expand_as(vae_mu)
            self._last_prior_logvar = prior_logvar.expand_as(vae_logvar)

        self._last_expr_mask = None
        self._last_context_stats = {}
        return z, vae_mu, vae_logvar, gene_embs, None

    def get_representation(self, latent: torch.Tensor) -> torch.Tensor:
        if self.repr_pooling == "attention":
            if self.repr_pooler is None:
                raise RuntimeError("repr_pooler is not initialized")
            return self.repr_pooler(latent)
        if self.repr_pooling == "mean_attn_residual":
            if self.repr_pooler is None:
                raise RuntimeError("repr_pooler is not initialized")
            mean = latent.mean(dim=1)
            attn = self.repr_pooler(latent)
            return mean + self.repr_pooling_residual_alpha * (attn - mean)
        if self.repr_pooling == "cls":
            return latent[:, 0, :]
        return latent.mean(dim=1)

    def get_decoder_query_representation(self, latent: torch.Tensor) -> torch.Tensor:
        mode = self.decoder_query_pooling
        if mode in {"", "same"}:
            return self.get_representation(latent)
        if mode == "mean":
            return latent.mean(dim=1)
        if mode == "attention":
            if self.decoder_query_pooler is None:
                raise RuntimeError("decoder_query_pooler is not initialized")
            return self.decoder_query_pooler(latent)
        if mode == "mean_attn_residual":
            if self.decoder_query_pooler is None:
                raise RuntimeError("decoder_query_pooler is not initialized")
            mean = latent.mean(dim=1)
            attn = self.decoder_query_pooler(latent)
            return mean + self.decoder_query_residual_alpha * (attn - mean)
        raise RuntimeError(f"Unsupported decoder_query_pooling: {mode}")

    def compute_mvc_pred(
        self,
        cell_emb: torch.Tensor,
        gene_embs_raw: torch.Tensor,
    ) -> torch.Tensor:
        del cell_emb, gene_embs_raw
        raise RuntimeError("MVC is not part of the official scTrilemma VAE.")

    def decode(
        self,
        z: torch.Tensor,
        gene_embs: torch.Tensor,
        context: Optional[torch.Tensor],
        gene_indices: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None,
        raw_input: Optional[torch.Tensor] = None,
        return_refined_kv: bool = False,
        z_gene: Optional[torch.Tensor] = None,
        disable_pb: bool = False,
        tissue_code: Optional[torch.Tensor] = None,
        context_group_labels: Optional[torch.Tensor] = None,
    ):
        del context, raw_input, z_gene, disable_pb, tissue_code, context_group_labels
        queries = gene_embs

        if self.decoder_query_mode != "off":
            z_pool = self.get_decoder_query_representation(z).unsqueeze(1)
            if self.decoder_query_mode == "z_added":
                queries = queries + z_pool
            elif self.decoder_query_mode == "z_modulated":
                if self.z_query_gate is None:
                    raise RuntimeError("z_query_gate must be initialized for z_modulated")
                gate = torch.sigmoid(self.z_query_gate(z_pool.squeeze(1))).unsqueeze(1)
                queries = queries * gate

        if padding_mask is not None:
            queries = queries * padding_mask.unsqueeze(-1).to(queries.dtype)

        memory = z
        for block in self.decoder_blocks:
            queries = block(queries, memory, query_mask=padding_mask)

        out = self.output_proj(queries)
        mu_logits, pi_logits = out.chunk(2, dim=-1)
        mu_logits = mu_logits.squeeze(-1)
        pi_logits = pi_logits.squeeze(-1)

        if padding_mask is not None:
            mu_logits = mu_logits.masked_fill(~padding_mask, float("-inf"))

        if self.mu_activation_mode == "softmax":
            mu = torch.softmax(mu_logits, dim=-1)
        elif self.mu_activation_mode == "normalized_softplus":
            mu = F.softplus(mu_logits)
            mu = mu / (mu.sum(dim=-1, keepdim=True) + self.eps)
        elif self.mu_activation_mode == "softplus":
            mu = F.softplus(mu_logits)
        elif self.mu_activation_mode == "exp":
            if library_size is not None:
                offset = torch.log(library_size.unsqueeze(-1) + self.eps)
                mu = torch.exp((mu_logits + offset).clamp(max=20.0))
            else:
                mu = torch.exp(mu_logits.clamp(max=20.0))
        else:
            stabilized = mu_logits - mu_logits.max(dim=-1, keepdim=True).values
            mu = torch.softmax(stabilized, dim=-1)

        if library_size is not None and self.mu_activation_mode != "exp":
            mu = mu * library_size.unsqueeze(-1)

        if gene_indices is None:
            raise ValueError("gene_indices is required for gene-wise dispersion.")
        theta = F.softplus(self.px_r[gene_indices])
        if padding_mask is not None:
            theta = theta * padding_mask.float() + self.eps

        pi = pi_logits if self.zi_return_logits else torch.sigmoid(pi_logits)
        aux_loss = queries.new_zeros(())
        if return_refined_kv:
            return mu, theta, pi, aux_loss, z
        return mu, theta, pi, aux_loss

    def forward(
        self,
        meta_features: Dict[str, torch.Tensor],
        raw_input: torch.Tensor,
        pseudo_bulk: torch.Tensor,
        pseudo_bulk_values: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        gene_indices: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None,
        tissue_code: Optional[torch.Tensor] = None,
        context_group_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z, vae_mu, vae_logvar, gene_embs, z_gene = self.encode(
            meta_features,
            raw_input,
            mask=mask,
            gene_indices=gene_indices,
            pseudo_bulk=pseudo_bulk,
            pseudo_bulk_values=pseudo_bulk_values,
            tissue_code=tissue_code,
            context_group_labels=context_group_labels,
        )
        zinb_mu, zinb_theta, zinb_pi, _ = self.decode(
            z,
            gene_embs,
            pseudo_bulk,
            gene_indices=gene_indices,
            padding_mask=mask,
            library_size=library_size,
            z_gene=z_gene,
            tissue_code=tissue_code,
            context_group_labels=context_group_labels,
        )
        return {
            "z_1": z,
            "vae_mu": vae_mu,
            "vae_logvar": vae_logvar,
            "zinb_mu": zinb_mu,
            "zinb_theta": zinb_theta,
            "zinb_pi": zinb_pi,
        }
