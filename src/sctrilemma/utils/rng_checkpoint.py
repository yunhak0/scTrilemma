"""Callback: save/restore RNG state + resume metadata in Lightning checkpoints.

Lightning's default checkpoint captures model weights, optimizer states, LR
schedulers, global_step, epoch, and loop state — but NOT the torch / cuda /
numpy / python RNG states. On resume, the main process RNG is fresh (seeded
via `pl.seed_everything`), which causes short-term training dynamics to
diverge from an uninterrupted run even when all other state is restored.

This callback patches the gap:

  - on_save_checkpoint: snapshot rank-0 RNG into ckpt["rng_state"]
  - on_load_checkpoint: restore rank-0 RNG from ckpt["rng_state"]
  - on_save_checkpoint: also store wandb_run_id so resume can continue the
    same wandb run line.

It does NOT attempt to save per-DataLoader-worker RNG state (workers are
stateless sub-processes; saving their RNG would require PyTorch-internal
hooks).  Resumed runs therefore reproduce the checkpointed RNG state but not
the exact data order of the interrupted run.

Usage:
    trainer = pl.Trainer(
        ...,
        callbacks=[
            ModelCheckpoint(...),
            RngStateCheckpoint(),
            ...
        ],
    )
"""
from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

RNG_STATE_KEY = "rng_state"
RESUME_META_KEY = "resume_meta"


class RngStateCheckpoint(pl.Callback):
    """Serialize torch / cuda / numpy / python RNG into every checkpoint.

    Only rank 0 writes; on load every rank restores its own copy from the
    broadcasted ckpt dict. That is a minor simplification — torch's per-rank
    RNG state would in principle differ, but for reproducibility-of-trajectory
    purposes rank-0 RNG is by far the dominant driver (model init, augmentation
    decisions), so this is sufficient.

    The ``wandb_run_id`` is recorded via ``set_wandb_run_id`` before the first
    checkpoint save so resume can wire ``WandbLogger(id=..., resume="allow")``
    in the next launch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._wandb_run_id: str | None = None

    # -------- external setters (called by train entrypoint) --------
    def set_wandb_run_id(self, run_id: str | None) -> None:
        self._wandb_run_id = run_id

    # -------- Lightning hooks --------
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        try:
            cuda_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            )
            checkpoint[RNG_STATE_KEY] = {
                "torch": torch.get_rng_state(),
                "cuda": cuda_states,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            }
        except Exception as e:  # pyright: ignore
            logger.warning(f"RngStateCheckpoint: save failed (non-fatal): {e}")

        meta: dict[str, Any] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_step": int(getattr(trainer, "global_step", 0) or 0),
        }
        if self._wandb_run_id:
            meta["wandb_run_id"] = self._wandb_run_id
        checkpoint[RESUME_META_KEY] = meta

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        rng = checkpoint.get(RNG_STATE_KEY)
        if not rng:
            logger.warning(
                "RngStateCheckpoint: ckpt has no rng_state — likely a pre-2026-04-19 "
                "checkpoint. Skipping RNG restore (main RNG stays at its seeded value)."
            )
            return
        try:
            torch.set_rng_state(rng["torch"])
            cuda_states = rng.get("cuda") or []
            if torch.cuda.is_available() and cuda_states:
                # Restore per-device; torch.cuda.set_rng_state_all handles list.
                torch.cuda.set_rng_state_all(cuda_states)
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
            logger.info(
                "RngStateCheckpoint: restored torch/cuda/numpy/python RNG from ckpt"
            )
        except Exception as e:  # pyright: ignore
            logger.warning(f"RngStateCheckpoint: restore failed (non-fatal): {e}")


def read_resume_meta(ckpt_dict: dict[str, Any]) -> dict[str, Any]:
    """Convenience reader for train entrypoints — returns {} if absent."""
    meta = ckpt_dict.get(RESUME_META_KEY)
    if isinstance(meta, dict):
        return meta
    return {}
