"""Train scTrilemma.

Usage (from the repository root):

    pixi run python -m sctrilemma.train_vae
    pixi run python -m sctrilemma.train_vae experiment=<name>   # overlay from configs/experiment/

The default configuration in ``configs/sctrilemma.yaml`` is the recipe used for
the released checkpoint.
"""

import json
import logging
import os
import warnings
from numbers import Integral
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info

from sctrilemma.data.context_datamodule import ContextDataModule
from sctrilemma.model.vae_module import ScTrilemmaModule
from sctrilemma.utils.launch_utils import (
    _apply_perf_flags,
    _build_run_name,
    _get_git_code_state,
)
from sctrilemma.utils.logger import log_config, setup_logger
from sctrilemma.utils.rng_checkpoint import (
    RngStateCheckpoint,
    read_resume_meta,
)
from sctrilemma.utils.training import (
    estimate_val_batches_from_data,
    find_latest_checkpoint,
    get_total_cells_from_metadata,
)

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
torch.serialization.add_safe_globals([DictConfig, ListConfig])


def _normalize_open_shards_schedule(raw_schedule: Any) -> list[tuple[int, int]]:
    """Validate ``data.num_open_shards_schedule``.

    The schedule is a list of ``[step, n]`` pairs: from global step ``step``
    onward, each data-loader worker reads from ``n`` cached shards at a time
    when filling its shuffle buffer. Steps must start at 0 and increase.
    """
    if raw_schedule is None:
        return []
    if not isinstance(raw_schedule, (list, tuple, ListConfig)):
        raise ValueError("data.num_open_shards_schedule must be a list of [step, n] pairs")

    schedule: list[tuple[int, int]] = []
    for idx, pair in enumerate(raw_schedule):
        if not isinstance(pair, (list, tuple, ListConfig)) or len(pair) != 2:
            raise ValueError(f"data.num_open_shards_schedule[{idx}] must be [step, n]")
        if isinstance(pair[0], bool) or not isinstance(pair[0], Integral):
            raise ValueError(f"data.num_open_shards_schedule[{idx}] step must be an int")
        if isinstance(pair[1], bool) or not isinstance(pair[1], Integral):
            raise ValueError(f"data.num_open_shards_schedule[{idx}] n must be an int")
        step = int(pair[0])
        n = int(pair[1])
        if step < 0:
            raise ValueError(f"data.num_open_shards_schedule[{idx}] step must be non-negative")
        if n <= 0:
            raise ValueError(f"data.num_open_shards_schedule[{idx}] n must be positive")
        if schedule and step <= schedule[-1][0]:
            raise ValueError("data.num_open_shards_schedule steps must be strictly increasing")
        schedule.append((step, n))

    if schedule and schedule[0][0] != 0:
        raise ValueError("data.num_open_shards_schedule must start at step 0")
    return schedule


def _open_shards_for_step(schedule: list[tuple[int, int]], step: int) -> int:
    if not schedule:
        raise ValueError("schedule must not be empty")
    current = schedule[0][1]
    for boundary_step, n in schedule:
        if step < boundary_step:
            break
        current = n
    return current


class OpenShardsScheduleCallback(pl.Callback):
    """Apply ``data.num_open_shards_schedule`` to the data module during training."""

    def __init__(self, schedule: list[tuple[int, int]]):
        super().__init__()
        if not schedule:
            raise ValueError("OpenShardsScheduleCallback requires a non-empty schedule")
        self.schedule = schedule
        self._last_applied: int | None = None

    def _apply(self, trainer: pl.Trainer, step: int) -> None:
        dm = trainer.datamodule
        if dm is None or not hasattr(dm, "set_num_open_shards"):
            raise RuntimeError("OpenShardsScheduleCallback requires ContextDataModule")

        n = _open_shards_for_step(self.schedule, step)
        current = dm.current_num_open_shards() if hasattr(dm, "current_num_open_shards") else None
        if current == n and self._last_applied == n:
            return

        dm.set_num_open_shards(n)
        self._last_applied = n
        rank_zero_info(f"Data loading: step={step} num_open_shards={n}")
        if trainer.logger is not None:
            trainer.logger.log_metrics({"data/num_open_shards": float(n)}, step=step)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dm = trainer.datamodule
        train_dataset = getattr(dm, "train_dataset", None)
        if train_dataset is not None and not getattr(train_dataset, "_cache_available", False):
            raise RuntimeError(
                "data.num_open_shards_schedule requires the preprocessed .npz cache "
                "(see scripts/preprocessing/3_preprocess_cache.sh)."
            )
        self._apply(trainer, int(trainer.global_step))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._apply(trainer, int(trainer.global_step))


def _resolve_from_cwd(path: str | None, original_cwd: str) -> str | None:
    """Resolve a relative config path against the directory the job was launched from."""
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(original_cwd, path)


@hydra.main(version_base=None, config_path="../../configs", config_name="sctrilemma")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _apply_perf_flags(cfg, local_rank)
    _ = pl.seed_everything(cfg.training.seed)

    run_dir = os.getcwd()
    log_path = os.path.join(run_dir, "training.log")
    original_cwd = get_original_cwd()

    setup_logger(log_file=log_path, to_console=True)
    logger = logging.getLogger(__name__)
    if local_rank == 0:
        logger.info(f"Hydra run dir: {run_dir}")
        logger.info(f"Original cwd: {original_cwd}")
        log_config(cfg, logger)

    open_shards_schedule = _normalize_open_shards_schedule(
        cfg.data.get("num_open_shards_schedule", None)
    )
    initial_open_shards = (
        _open_shards_for_step(open_shards_schedule, 0)
        if open_shards_schedule
        else int(cfg.data.get("num_open_shards", 1))
    )
    val_subset_manifest = _resolve_from_cwd(cfg.data.get("val_subset_manifest", None), original_cwd)

    dm = ContextDataModule(
        data_root=cfg.data.root,
        census_version=cfg.data.census_version,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        pseudo_bulk_path=cfg.data.pseudo_bulk_path,
        gene_vocab_path=cfg.data.gene_vocab_path,
        cell_type_vocab_path=cfg.data.cell_type_vocab_path,
        min_genes=cfg.data.min_genes,
        val_split_ratio=cfg.data.val_split_ratio,
        split_seed=cfg.data.split_seed,
        val_full=cfg.data.get("val_full", False),
        do_log1p=cfg.data.do_log1p,
        do_normalize=cfg.data.do_normalize,
        target_sum=cfg.data.target_sum,
        max_len=cfg.data.get("max_len", 2048),
        max_tokens_per_batch=cfg.training.max_tokens_per_batch,
        batch_buffer_size=cfg.training.batch_buffer_size,
        num_open_shards=initial_open_shards,
        enable_open_shards_schedule=bool(open_shards_schedule),
        celltype_stratified=bool(cfg.data.get("celltype_stratified", False)),
        crop_size=cfg.training.get("crop_size", None),
        val_crop_size=cfg.data.get("val_crop_size", None),
        zero_sampling=cfg.data.get("zero_sampling", False),
        zero_sampling_fraction=cfg.data.get("zero_sampling_fraction", 0.0),
        zero_sampling_strategy=cfg.data.get("zero_sampling_strategy", "random"),
        count_weights_cap=cfg.data.get("count_weights_cap", None),
        use_count_weights=cfg.data.get("use_count_weights", True),
        cache_path=cfg.data.get("cache_path", None),
        prefetch_factor=cfg.training.prefetch_factor,
        persistent_workers=cfg.training.persistent_workers,
        pin_memory=cfg.training.pin_memory,
        val_batch_buffer_size=cfg.training.get("val_batch_buffer_size", None),
        val_num_workers=cfg.training.get("val_num_workers", None),
        val_prefetch_factor=cfg.training.get("val_prefetch_factor", None),
        val_persistent_workers=cfg.training.get("val_persistent_workers", None),
        val_pin_memory=cfg.training.get("val_pin_memory", None),
        use_raw_expr=cfg.data.get("use_raw_expr", False),
        tissue_code_path=cfg.data.get("tissue_code_path", None),
        val_subset_manifest=val_subset_manifest,
        val_data_root=cfg.data.get("val_data_root", None),
        val_census_version=cfg.data.get("val_census_version", None),
    )

    if cfg.model.get("vocab_size") is None or cfg.model.get("vocab_size") != 61890:
        with open(cfg.data.gene_vocab_path) as f:
            vocab = json.load(f)
        real_vocab_size = len(vocab)
        if local_rank == 0:
            logger.info(f"Detected vocab size: {real_vocab_size}")
        OmegaConf.set_struct(cfg, False)
        cfg.model.vocab_size = real_vocab_size
        OmegaConf.set_struct(cfg, True)

    explicit_max_steps = (
        int(cfg.trainer.max_steps)
        if cfg.trainer.get("max_steps") and cfg.trainer.max_steps > 0
        else None
    )
    total_steps = explicit_max_steps if explicit_max_steps else -1
    local_train_len = 0
    local_train_len_ready = False
    try:
        total_cells = get_total_cells_from_metadata(cfg.data.root, cfg.data.census_version)
        val_ratio = cfg.data.val_split_ratio
        train_cells = int(total_cells * (1 - val_ratio))

        if cfg.trainer.devices == -1:
            devices = torch.cuda.device_count()
        elif isinstance(cfg.trainer.devices, list):
            devices = len(cfg.trainer.devices)
        else:
            devices = int(cfg.trainer.devices)

        world_size = devices * cfg.trainer.num_nodes
        batch_size = cfg.training.batch_size
        accum_grad = cfg.trainer.accumulate_grad_batches

        if batch_size and world_size and accum_grad:
            steps_per_epoch = train_cells // (batch_size * world_size * accum_grad)
            if train_cells % (batch_size * world_size * accum_grad) != 0:
                steps_per_epoch += 1
            total_steps = int(steps_per_epoch * cfg.trainer.max_epochs)
            if explicit_max_steps:
                total_steps = explicit_max_steps

            local_train_len = (train_cells // world_size) // batch_size
            local_train_len_ready = True

            local_val_len = estimate_val_batches_from_data(
                data_root=cfg.data.root,
                census_version=cfg.data.census_version,
                val_split_ratio=cfg.data.val_split_ratio,
                split_seed=cfg.data.split_seed,
                batch_size=batch_size,
                world_size=world_size,
                crop_size=cfg.training.get("crop_size", None),
                max_tokens_per_batch=cfg.training.get("max_tokens_per_batch", None),
                val_subset_manifest=val_subset_manifest,
                val_data_root=cfg.data.get("val_data_root", None),
                val_census_version=cfg.data.get("val_census_version", None),
            )

            dm.estimated_train_len = int(local_train_len)
            dm.estimated_val_len = int(local_val_len)
            if local_rank == 0:
                logger.info(
                    f"Estimated train/val batches per rank: {dm.estimated_train_len}/{dm.estimated_val_len}"
                )
    except Exception as e:
        if local_rank == 0:
            logger.warning(f"Failed to estimate train/val batches: {e}")

    ckpt_dir = os.path.join(original_cwd, "checkpoints", cfg.experiment_name)
    resume_ckpt_path = None
    if cfg.get("ckpt_path") is not None:
        resume_ckpt_path = _resolve_from_cwd(cfg.ckpt_path, original_cwd)
        if local_rank == 0:
            logger.info(f"Using checkpoint from CLI: {resume_ckpt_path}")
    elif cfg.get("auto_resume", False):
        found_ckpt = find_latest_checkpoint(
            outputs_dir=os.path.join(original_cwd, "checkpoints"),
            experiment_name=cfg.experiment_name,
        )
        if found_ckpt:
            resume_ckpt_path = found_ckpt
            if local_rank == 0:
                logger.info(f"Auto-resuming from latest checkpoint: {resume_ckpt_path}")
    resume_global_step = 0
    resume_epoch = 0
    resume_wandb_run_id: str | None = None
    if resume_ckpt_path is not None:
        resume_meta_ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        resume_global_step = int(resume_meta_ckpt.get("global_step", 0))
        resume_epoch = int(resume_meta_ckpt.get("epoch", 0))
        resume_meta_block = read_resume_meta(resume_meta_ckpt)
        resume_wandb_run_id = resume_meta_block.get("wandb_run_id")
        del resume_meta_ckpt
        if local_rank == 0:
            logger.info(
                "Resume metadata: "
                + f"epoch={resume_epoch}, global_step={resume_global_step}"
                + (f", wandb_run_id={resume_wandb_run_id}" if resume_wandb_run_id
                   else ", wandb_run_id=<absent, starting a fresh wandb run>")
            )

        # Offset the dataset shuffle seed so a resumed run does not replay the
        # data order already consumed before the checkpoint.
        if local_train_len_ready and local_train_len > 0:
            epoch_base_guess = resume_global_step // int(local_train_len) + 1
        else:
            epoch_base_guess = resume_global_step // 10000 + 1
        dm.set_epoch_base(epoch_base_guess)
        if local_rank == 0:
            logger.info(f"Resume: dm.epoch_base = {epoch_base_guess}")

    if open_shards_schedule:
        resume_n = _open_shards_for_step(open_shards_schedule, resume_global_step)
        dm.set_num_open_shards(resume_n)
        if local_rank == 0:
            logger.info(
                f"num_open_shards schedule={open_shards_schedule}, "
                f"value at step {resume_global_step}: {resume_n}"
            )

    model = ScTrilemmaModule(cfg)

    wandb_logger = None
    if cfg.logging.wandb.enabled and local_rank == 0:
        wandb_name_override = cfg.logging.wandb.get("name", None)
        run_name = wandb_name_override if wandb_name_override else _build_run_name(
            cfg.experiment_name, resume_global_step, cfg.trainer.get("max_steps"),
        )
        run_tags = list(cfg.logging.wandb.tags)
        if resume_ckpt_path is not None:
            run_tags.append("resume" if resume_wandb_run_id else "resume-no-continuity")
        # If the checkpoint recorded its wandb run id, continue that run so the
        # curves stay on a single line; otherwise start a fresh run.
        wandb_logger_kwargs: dict = dict(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            name=run_name,
            mode=cfg.logging.wandb.mode,
            tags=run_tags,
            log_model="end",
        )
        if resume_wandb_run_id:
            wandb_logger_kwargs["id"] = resume_wandb_run_id
            wandb_logger_kwargs["resume"] = "allow"
            logger.info(f"Continuing wandb run id={resume_wandb_run_id}")
        wandb_logger = WandbLogger(**wandb_logger_kwargs)  # pyright: ignore[reportArgumentType]
        wandb_logger.experiment.define_metric("trainer/global_step")
        wandb_logger.experiment.define_metric("*", step_metric="trainer/global_step")
        try:
            code_state = _get_git_code_state(original_cwd)
            wandb_logger.experiment.config.update(code_state, allow_val_change=True)
            logger.info(f"Code state logged: {code_state['code.status_summary']}")
        except Exception as e:
            wandb_logger.experiment.config.update(
                {"code.state_available": False, "code.state_error": str(e)},
                allow_val_change=True,
            )
            logger.warning(f"Code state logging failed: {e}")
        wandb_logger.experiment.config.update(
            {
                "run.experiment_name": cfg.experiment_name,
                "run.is_resume": resume_ckpt_path is not None,
                "run.resume_ckpt_path": resume_ckpt_path,
                "run.resume_global_step": resume_global_step,
                "run.resume_epoch": resume_epoch,
            },
            allow_val_change=True,
        )

    # save_top_k=-1 keeps every validation checkpoint; a positive value keeps
    # the best ones by the monitored metric.
    ckpt_save_top_k = int(cfg.trainer.get("ckpt_save_top_k", 3))
    ckpt_every_n_train_steps = cfg.trainer.get("ckpt_every_n_train_steps", None)
    ckpt_kwargs = dict(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{step}",
        monitor="val/zinb_nll",
        mode="min",
        save_last=True,
        save_top_k=ckpt_save_top_k,
    )
    if ckpt_every_n_train_steps is not None:
        ckpt_kwargs["every_n_train_steps"] = int(ckpt_every_n_train_steps)
        ckpt_kwargs["save_top_k"] = ckpt_save_top_k
    checkpoint_callback = ModelCheckpoint(**ckpt_kwargs)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    # Saves/restores RNG states and records the wandb run id in every checkpoint.
    rng_callback = RngStateCheckpoint()
    if wandb_logger is not None:
        rng_callback.set_wandb_run_id(str(wandb_logger.experiment.id))
    callbacks: list[pl.Callback] = [checkpoint_callback, lr_monitor, rng_callback]
    if open_shards_schedule:
        callbacks.append(OpenShardsScheduleCallback(open_shards_schedule))

    limit_train_batches = cfg.trainer.get("limit_train_batches", None)
    if limit_train_batches is None and explicit_max_steps:
        limit_train_batches = explicit_max_steps
    elif limit_train_batches is None and total_steps > 0 and local_train_len_ready:
        limit_train_batches = int(local_train_len)

    trainer = pl.Trainer(
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        accelerator="auto",
        max_epochs=cfg.trainer.max_epochs,
        max_steps=total_steps if total_steps > 0 else -1,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        limit_val_batches=cfg.trainer.get("limit_val_batches", 1.0),
        limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.0,
        gradient_clip_val=cfg.trainer.grad_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt_path, weights_only=False)

    final_ckpt_path = os.path.join(ckpt_dir, "final.ckpt")
    trainer.save_checkpoint(final_ckpt_path)
    logger.info(f"Saved final checkpoint: {final_ckpt_path}")

    if wandb_logger is not None and local_rank == 0:
        try:
            import wandb

            ckpt_paths = [("final", final_ckpt_path)]
            if checkpoint_callback.last_model_path:
                ckpt_paths.append(("last", checkpoint_callback.last_model_path))
            if checkpoint_callback.best_model_path:
                ckpt_paths.append(("best", checkpoint_callback.best_model_path))

            for alias, path in ckpt_paths:
                if not path or not os.path.exists(path):
                    continue
                artifact_name = f"{cfg.logging.wandb.name}-{alias}".replace("/", "-")
                artifact = wandb.Artifact(name=artifact_name, type="checkpoint")
                _ = artifact.add_file(path)
                wandb_logger.experiment.log_artifact(artifact, aliases=[alias, "latest"])
                logger.info(f"Logged checkpoint artifact ({alias}): {path}")
        except Exception as e:
            logger.warning(f"Failed to log checkpoint artifact: {e}")


if __name__ == "__main__":
    main()
