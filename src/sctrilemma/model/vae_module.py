"""Lightning wrapper for the scTrilemma latent-bottleneck VAE.

This module intentionally keeps only the official VAE training and
evaluation path: ZINB/NB reconstruction, KL regularization, validation
reconstruction metrics, and scIB-style embedding metrics. Research probes,
auxiliary losses, teacher models, and inactive ablation branches live outside
the minimal implementation.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from sctrilemma.model.loss import NBLoss, ZINBLoss
from sctrilemma.model.vae import ScTrilemmaVAE
from sctrilemma.utils.metrics import (
    compute_bio_conservation_metrics,
    compute_combined_zero_metrics,
    compute_count_weight_metrics,
    compute_pearson_correlation,
    compute_spearman_correlation,
)


def _select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    """Read a nested OmegaConf value with a plain default."""
    return OmegaConf.select(cfg, path, default=default)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _all_gather_cat(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Gather variable-length tensors along dim 0 and concatenate."""
    if world_size <= 1:
        return tensor

    local_n = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world_size)]
    dist.all_gather(sizes, local_n)
    sizes_int = [int(s.item()) for s in sizes]
    max_n = max(sizes_int)

    if tensor.shape[0] < max_n:
        pad_shape = (max_n - tensor.shape[0], *tensor.shape[1:])
        pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        padded = torch.cat([tensor, pad], dim=0)
    else:
        padded = tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([chunk[:n] for chunk, n in zip(gathered, sizes_int)], dim=0)


def _batch_raw_input(batch: dict[str, Any]) -> torch.Tensor:
    """Return the expression input used by the VAE encoder."""
    if "raw_input" in batch:
        return batch["raw_input"]
    return batch["residual"]


def _as_str_list(values: Any, n: int) -> list[str]:
    if values is None:
        return ["unknown"] * n
    if isinstance(values, str):
        return [values] * n
    if isinstance(values, torch.Tensor):
        flat = values.detach().cpu().reshape(-1).tolist()
        return [str(v) for v in flat]
    if isinstance(values, np.ndarray):
        return [str(v) for v in values.reshape(-1).tolist()]
    if isinstance(values, Iterable):
        out = [str(v) for v in values]
        if len(out) == n:
            return out
        if len(out) == 1:
            return out * n
    return ["unknown"] * n


class ScTrilemmaModule(pl.LightningModule):
    """Official LightningModule for scTrilemma."""

    def __init__(self, cfg: DictConfig | None = None, **legacy_hparams: Any) -> None:
        super().__init__()
        if cfg is None:
            nested_cfg = legacy_hparams.get("cfg")
            if nested_cfg is not None:
                cfg = nested_cfg
            elif legacy_hparams:
                cfg = OmegaConf.create(legacy_hparams)
            else:
                raise ValueError("ScTrilemmaModule requires a Hydra config")
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = ScTrilemmaVAE(
            d_model=int(_select(cfg, "model.d_model")),
            num_latent_tokens=int(_select(cfg, "model.num_latent_tokens")),
            num_encoder_layers=int(_select(cfg, "model.num_encoder_layers")),
            num_decoder_layers=int(_select(cfg, "model.num_decoder_layers")),
            encoder_heads=int(_select(cfg, "model.encoder_heads")),
            decoder_heads=int(_select(cfg, "model.decoder_heads")),
            dropout=float(_select(cfg, "model.dropout", 0.1)),
            vocab_size=int(_select(cfg, "model.vocab_size")),
            repr_pooling=str(_select(cfg, "model.repr_pooling", "mean")),
            logvar_min=float(_select(cfg, "model.logvar_min", -10.0)),
            logvar_max=float(_select(cfg, "model.logvar_max", 10.0)),
            zi_return_logits=bool(_select(cfg, "model.zi_return_logits", False)),
            mu_activation_mode=str(_select(cfg, "model.mu_activation_mode", "softmax")),
            expr_proj_mode=str(_select(cfg, "model.expr_proj_mode", "multiplicative")),
            tissue_code_dim=int(_select(cfg, "model.tissue_code_dim", 0)),
            tissue_code_embedding_mode=str(
                _select(cfg, "model.tissue_code_embedding_mode", "soft")
            ),
            tissue_prior_mode=str(_select(cfg, "model.tissue_prior_mode", "")),
            tissue_prior_mu_scale=float(_select(cfg, "model.tissue_prior_mu_scale", 1.0)),
            tissue_prior_logvar_mode=str(
                _select(cfg, "model.tissue_prior_logvar_mode", "learned")
            ),
            decoder_query_mode=str(_select(cfg, "model.decoder_query_mode", "off")),
            decoder_query_pooling=str(_select(cfg, "model.decoder_query_pooling", "same")),
            decoder_query_residual_alpha=float(
                _select(cfg, "model.decoder_query_residual_alpha", 0.2)
            ),
            repr_pooling_residual_alpha=float(
                _select(cfg, "model.repr_pooling_residual_alpha", 0.2)
            ),
        )

        gene_likelihood = str(_select(cfg, "training.gene_likelihood", "zinb")).lower()
        if gene_likelihood not in {"zinb", "nb"}:
            raise ValueError(f"Unsupported gene_likelihood: {gene_likelihood}")
        zinb_reduction = str(_select(cfg, "training.zinb_reduction", "per_cell")).lower()
        if zinb_reduction not in {"mean", "per_cell", "sum"}:
            raise ValueError(f"Unsupported zinb_reduction: {zinb_reduction}")
        self.gene_likelihood = gene_likelihood
        self.recon_loss_fn = (
            NBLoss(reduction=zinb_reduction)
            if gene_likelihood == "nb"
            else ZINBLoss(
                reduction=zinb_reduction,
                logits=bool(_select(cfg, "model.zi_return_logits", False)),
            )
        )

        self.lambda_kl_target = float(_select(cfg, "training.lambda_kl", 0.001))
        self.kl_warmup_steps = int(_select(cfg, "training.kl_warmup_steps", 0))
        self.lambda_zinb_start = float(_select(cfg, "training.lambda_zinb", 1.0))
        self.lambda_zinb_end = _select(cfg, "training.lambda_zinb_end", None)
        self.lambda_zinb_end = (
            self.lambda_zinb_start
            if self.lambda_zinb_end is None
            else float(self.lambda_zinb_end)
        )
        self.lambda_zinb_decay_steps = int(
            _select(cfg, "training.lambda_zinb_decay_steps", 0)
        )
        self.lambda_zinb_trigger_step = int(
            _select(cfg, "training.lambda_zinb_trigger_step", 0)
        )

        self._current_batch_size: int | None = None
        self._clear_validation_buffers()

    def log(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if "batch_size" not in kwargs and self._current_batch_size is not None:
            kwargs["batch_size"] = self._current_batch_size
        super().log(name, value, *args, **kwargs)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self._run_model(batch)

    def on_train_batch_start(
        self, batch: dict[str, Any], batch_idx: int
    ) -> None:
        self._current_batch_size = int(_batch_raw_input(batch).shape[0])

    def on_validation_batch_start(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._current_batch_size = int(_batch_raw_input(batch).shape[0])

    def on_validation_epoch_start(self) -> None:
        self._clear_validation_buffers()

    def _clear_validation_buffers(self) -> None:
        self._val_embeddings: list[torch.Tensor] = []
        self._val_cell_types: list[torch.Tensor] = []
        self._val_batch_ids: list[torch.Tensor] = []
        self._val_dataset_ids: list[str] = []

    def _run_model(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        expression = _batch_raw_input(batch)
        gene_indices = batch["gene_indices"]
        padding_mask = batch.get("padding_mask")
        tissue_code = batch.get("tissue_code")
        context_group_labels = batch.get("batch_labels")

        z, vae_mu, vae_logvar, gene_embs, z_gene = self.model.encode(
            {},
            expression,
            mask=padding_mask,
            gene_indices=gene_indices,
            pseudo_bulk=batch.get("pseudo_bulk"),
            tissue_code=tissue_code,
            context_group_labels=context_group_labels,
        )
        library_size = batch.get("library_size")
        if library_size is None:
            library_size = batch.get("library_size_raw")
        zinb_mu, zinb_theta, zinb_pi, _ = self.model.decode(
            z,
            gene_embs,
            batch.get("pseudo_bulk"),
            gene_indices=gene_indices,
            padding_mask=padding_mask,
            library_size=library_size,
            z_gene=z_gene,
        )
        return {
            "z_1": z,
            "vae_mu": vae_mu,
            "vae_logvar": vae_logvar,
            "zinb_mu": zinb_mu,
            "zinb_theta": zinb_theta,
            "zinb_pi": zinb_pi,
        }

    def _get_kl_lambda(self, step: int) -> float:
        if self.kl_warmup_steps <= 0:
            return self.lambda_kl_target
        progress = min(max(float(step + 1) / float(self.kl_warmup_steps), 0.0), 1.0)
        return self.lambda_kl_target * progress

    def _get_lambda_zinb(self, step: int) -> float:
        if self.lambda_zinb_start == self.lambda_zinb_end:
            return self.lambda_zinb_start
        if self.lambda_zinb_decay_steps <= 0:
            return self.lambda_zinb_end
        if step < self.lambda_zinb_trigger_step:
            return self.lambda_zinb_start

        progress = min(
            (step - self.lambda_zinb_trigger_step)
            / float(self.lambda_zinb_decay_steps),
            1.0,
        )
        return (
            self.lambda_zinb_start
            + progress * (self.lambda_zinb_end - self.lambda_zinb_start)
        )

    def _compute_kl_loss(
        self,
        vae_mu: torch.Tensor,
        vae_logvar: torch.Tensor,
        prior_mu: torch.Tensor | None,
        prior_logvar: torch.Tensor | None,
    ) -> torch.Tensor:
        if prior_mu is None or prior_logvar is None:
            kl_per_dim = -0.5 * (1.0 + vae_logvar - vae_mu.square() - vae_logvar.exp())
        else:
            prior_mu = prior_mu.to(device=vae_mu.device, dtype=vae_mu.dtype)
            prior_logvar = prior_logvar.to(device=vae_mu.device, dtype=vae_mu.dtype)
            if prior_mu.shape != vae_mu.shape:
                prior_mu = prior_mu.expand_as(vae_mu)
            if prior_logvar.shape != vae_logvar.shape:
                prior_logvar = prior_logvar.expand_as(vae_logvar)
            kl_per_dim = 0.5 * (
                prior_logvar
                - vae_logvar
                + (vae_logvar.exp() + (vae_mu - prior_mu).square())
                / prior_logvar.exp().clamp_min(1e-8)
                - 1.0
            )
        return kl_per_dim.sum(dim=(1, 2)).mean()

    def _compute_recon_loss(
        self,
        outputs: dict[str, torch.Tensor],
        raw_counts: torch.Tensor,
        padding_mask: torch.Tensor | None,
        count_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gene_likelihood == "nb":
            return self.recon_loss_fn(
                outputs["zinb_mu"],
                outputs["zinb_theta"],
                raw_counts,
                mask=padding_mask,
                weight=count_weights,
            )
        return self.recon_loss_fn(
            outputs["zinb_mu"],
            outputs["zinb_theta"],
            outputs["zinb_pi"],
            raw_counts,
            mask=padding_mask,
            weight=count_weights,
        )

    def _loss_terms(
        self, batch: dict[str, Any], outputs: dict[str, torch.Tensor], step: int
    ) -> dict[str, torch.Tensor | float]:
        raw_counts = batch["raw_counts"]
        padding_mask = batch.get("padding_mask")
        count_weights = batch.get("count_weights")
        recon_loss = self._compute_recon_loss(
            outputs,
            raw_counts,
            padding_mask,
            count_weights,
        )
        prior_mu = getattr(self.model, "_last_prior_mu", None)
        prior_logvar = getattr(self.model, "_last_prior_logvar", None)
        kl_loss = self._compute_kl_loss(
            outputs["vae_mu"],
            outputs["vae_logvar"],
            prior_mu,
            prior_logvar,
        )
        lambda_kl = self._get_kl_lambda(step)
        lambda_zinb = self._get_lambda_zinb(step)
        loss = lambda_zinb * recon_loss + lambda_kl * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "lambda_kl": lambda_kl,
            "lambda_zinb": lambda_zinb,
        }

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self._run_model(batch)
        terms = self._loss_terms(batch, outputs, self.global_step)
        loss = terms["loss"]
        assert isinstance(loss, torch.Tensor)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", terms["recon_loss"], on_step=True, on_epoch=True)
        self.log("train/kl_loss", terms["kl_loss"], on_step=True, on_epoch=True)
        self.log(
            "train/lambda_kl_current",
            terms["lambda_kl"],
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/lambda_zinb_current",
            terms["lambda_zinb"],
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(
        self,
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        prefix = "val_full" if dataloader_idx == 1 else "val"
        outputs = self._run_model(batch)
        terms = self._loss_terms(batch, outputs, self.global_step)
        loss = terms["loss"]
        assert isinstance(loss, torch.Tensor)

        raw_counts = batch["raw_counts"]
        padding_mask = batch.get("padding_mask")
        count_weights = batch.get("count_weights")

        self.log(f"{prefix}/loss", loss, on_epoch=True, sync_dist=True)
        self.log(
            f"{prefix}/recon_loss",
            terms["recon_loss"],
            on_epoch=True,
            sync_dist=True,
        )
        if self.gene_likelihood == "zinb":
            self.log(
                f"{prefix}/zinb_nll",
                terms["recon_loss"],
                on_epoch=True,
                sync_dist=True,
            )
        self.log(f"{prefix}/kl_loss", terms["kl_loss"], on_epoch=True, sync_dist=True)
        self.log(
            f"{prefix}/recon_pearson",
            compute_pearson_correlation(outputs["zinb_mu"], raw_counts, padding_mask),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/recon_spearman",
            compute_spearman_correlation(outputs["zinb_mu"], raw_counts, padding_mask),
            on_epoch=True,
            sync_dist=True,
        )

        if self.gene_likelihood == "zinb":
            zero_metrics = compute_combined_zero_metrics(
                outputs["zinb_mu"],
                outputs["zinb_theta"],
                outputs["zinb_pi"],
                raw_counts,
                padding_mask,
                zi_logits=bool(_select(self.cfg, "model.zi_return_logits", False)),
            )
            for key, value in zero_metrics.items():
                self.log(
                    f"{prefix}/{key}",
                    value,
                    on_epoch=True,
                    sync_dist=True,
                )

        if count_weights is not None:
            for key, value in compute_count_weight_metrics(
                count_weights,
                padding_mask,
            ).items():
                self.log(
                    f"{prefix}/{key}",
                    value,
                    on_epoch=True,
                    sync_dist=True,
                )

        self._collect_bio_batch(batch, outputs)
        return loss

    def _collect_bio_batch(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> None:
        if not bool(_select(self.cfg, "bio_metrics.enabled", True)):
            return
        if bool(batch.get("_is_padding", False)):
            return
        if "cell_type_labels" not in batch or "batch_labels" not in batch:
            return

        embedding = self.model.get_representation(outputs["vae_mu"]).detach().float().cpu()
        cell_types = batch["cell_type_labels"].detach().long().cpu()
        batch_ids = batch["batch_labels"].detach().long().cpu()
        n = int(embedding.shape[0])

        self._val_embeddings.append(embedding)
        self._val_cell_types.append(cell_types)
        self._val_batch_ids.append(batch_ids)
        dataset_values = batch.get("dataset_id", batch.get("dataset_ids"))
        self._val_dataset_ids.extend(_as_str_list(dataset_values, n))

    def on_validation_epoch_end(self) -> None:
        if not bool(_select(self.cfg, "bio_metrics.enabled", True)):
            self._clear_validation_buffers()
            return
        if not self._val_embeddings:
            self._clear_validation_buffers()
            return

        embeddings = torch.cat(self._val_embeddings, dim=0).to(self.device)
        labels = torch.cat(self._val_cell_types, dim=0).to(self.device)
        batch_ids = torch.cat(self._val_batch_ids, dim=0).to(self.device)
        dataset_ids = list(self._val_dataset_ids)

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if world_size > 1:
            embeddings = _all_gather_cat(embeddings, world_size)
            labels = _all_gather_cat(labels, world_size)
            batch_ids = _all_gather_cat(batch_ids, world_size)
            gathered_dataset_ids: list[list[str] | None] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_dataset_ids, dataset_ids)
            dataset_ids = [
                item
                for sublist in gathered_dataset_ids
                if sublist is not None
                for item in sublist
            ]

        embeddings_cpu = embeddings.detach().float().cpu()
        labels_cpu = labels.detach().long().cpu()
        batch_cpu = batch_ids.detach().long().cpu()

        bio_cfg = _select(self.cfg, "bio_metrics", {}) or {}
        k = _to_int(getattr(bio_cfg, "k", None), 30)
        max_per_type = _to_int(getattr(bio_cfg, "max_per_type", None), 500)
        total_max = _to_int(getattr(bio_cfg, "total_max", None), 10000)
        subsample = bool(getattr(bio_cfg, "subsample", True))
        compute_graph = bool(getattr(bio_cfg, "compute_graph_connectivity", False))

        metrics = compute_bio_conservation_metrics(
            embeddings_cpu,
            labels_cpu,
            batch_cpu,
            subsample=subsample,
            max_per_type=max_per_type,
            total_max=total_max,
            k=k,
            compute_graph_connectivity=compute_graph,
        )
        self._log_bio_metrics("val", metrics)

        if bool(getattr(bio_cfg, "per_dataset_every_val", False)) and dataset_ids:
            self._log_per_dataset_bio_metrics(
                embeddings_cpu,
                labels_cpu,
                batch_cpu,
                dataset_ids,
                max_per_type=max_per_type,
                total_max=total_max,
                k=_to_int(getattr(bio_cfg, "per_dataset_k", None), k),
                compute_graph=compute_graph,
            )

        self._clear_validation_buffers()
        gc.collect()

    def _log_bio_metrics(self, prefix: str, metrics: dict[str, float | None]) -> None:
        for key, value in metrics.items():
            if value is None or not np.isfinite(value):
                continue
            self.log(
                f"{prefix}/bio_{key}",
                torch.tensor(float(value), device=self.device),
                on_epoch=True,
                sync_dist=True,
            )

    def _log_per_dataset_bio_metrics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        batch_ids: torch.Tensor,
        dataset_ids: list[str],
        *,
        max_per_type: int,
        total_max: int,
        k: int,
        compute_graph: bool,
    ) -> None:
        if len(dataset_ids) != int(embeddings.shape[0]):
            return

        by_key: dict[str, list[float]] = {}
        n_used = 0
        dataset_array = np.asarray(dataset_ids, dtype=object)
        for dataset_id in np.unique(dataset_array):
            mask_np = dataset_array == dataset_id
            if int(mask_np.sum()) < 50:
                continue
            mask = torch.from_numpy(mask_np)
            local_labels = labels[mask]
            if torch.unique(local_labels).numel() < 2:
                continue
            local_batch = batch_ids[mask]
            local_metrics = compute_bio_conservation_metrics(
                embeddings[mask],
                local_labels,
                local_batch,
                subsample=True,
                max_per_type=max_per_type,
                total_max=total_max,
                k=min(k, max(2, int(mask_np.sum()) - 1)),
                compute_graph_connectivity=compute_graph,
            )
            for key, value in local_metrics.items():
                if value is not None and np.isfinite(value):
                    by_key.setdefault(key, []).append(float(value))
            n_used += 1

        for key, values in by_key.items():
            if not values:
                continue
            self.log(
                f"val/bio_{key}_per_dataset",
                torch.tensor(float(np.mean(values)), device=self.device),
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            "val/bio_per_dataset_n_used",
            torch.tensor(float(n_used), device=self.device),
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, Any]:
        lr = float(_select(self.cfg, "optim.lr", _select(self.cfg, "training.lr", 2e-4)))
        weight_decay = float(_select(self.cfg, "optim.weight_decay", 0.0))
        betas = _select(self.cfg, "optim.betas", [0.9, 0.999])
        eps = float(_select(self.cfg, "optim.eps", 1e-8))
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(float(betas[0]), float(betas[1])),
            eps=eps,
        )

        sched_name = str(_select(self.cfg, "sched.name", "") or "").lower()
        if sched_name not in {"cosine", "cosine_warmup"}:
            return optimizer

        max_steps = int(_select(self.cfg, "trainer.max_steps", 0) or 0)
        if max_steps <= 0:
            try:
                max_steps = int(self.trainer.estimated_stepping_batches)
            except Exception:
                max_steps = 0
        if max_steps <= 0:
            return optimizer

        warmup_steps_raw = _select(self.cfg, "sched.warmup_steps", None)
        if warmup_steps_raw is None or str(warmup_steps_raw).lower() == "auto":
            warmup_steps = int(0.05 * max_steps)
        else:
            warmup_steps = int(warmup_steps_raw or 0)
        min_lr = float(_select(self.cfg, "sched.min_lr", 0.0) or 0.0)
        min_factor = min(max(min_lr / lr, 0.0), 1.0) if lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step) / float(warmup_steps), min_factor)
            denom = max(max_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / float(denom), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
