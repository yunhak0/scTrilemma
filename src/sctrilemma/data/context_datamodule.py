from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .context_collate import context_collate_fn
from .context_dataset import ContextDataset
from .context_transform import ContextTransform


class ContextDataModule(pl.LightningDataModule):
    """DataModule for the cache-backed scTrilemma training pipeline."""

    def __init__(
        self,
        data_root: str,
        census_version: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pseudo_bulk_path: str | None = None,
        gene_vocab_path: str | None = None,
        cell_type_vocab_path: str | None = None,
        min_genes: int = 5,
        val_split_ratio: float = 0.1,
        split_seed: int = 42,
        val_use_train: bool = False,
        val_full: bool = False,
        do_log1p: bool = True,
        do_normalize: bool = True,
        target_sum: float = 10000.0,
        max_len: int | None = 2048,
        max_tokens_per_batch: int = 500000,
        batch_buffer_size: int = 4096,
        num_open_shards: int = 1,
        enable_open_shards_schedule: bool = False,
        celltype_stratified: bool = False,
        crop_size: int | None = None,
        val_crop_size: int | None = None,
        val_subset_manifest: str | None = None,
        zero_sampling: bool = False,
        zero_sampling_fraction: float = 0.0,
        zero_sampling_strategy: str = "random",
        count_weights_cap: float | None = None,
        use_count_weights: bool = True,
        cache_path: str | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        val_batch_buffer_size: int | None = None,
        val_num_workers: int | None = None,
        val_prefetch_factor: int | None = None,
        val_persistent_workers: bool | None = None,
        val_pin_memory: bool | None = None,
        use_raw_expr: bool = False,
        tissue_code_path: str | None = None,
        val_data_root: str | None = None,
        val_census_version: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_root = data_root
        self.census_version = census_version
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pseudo_bulk_path = pseudo_bulk_path
        self.gene_vocab_path = gene_vocab_path
        self.cell_type_vocab_path = cell_type_vocab_path
        self.min_genes = int(min_genes)
        self.val_split_ratio = float(val_split_ratio)
        self.split_seed = int(split_seed)
        self.val_use_train = bool(val_use_train)
        self.val_full = bool(val_full)
        self.do_log1p = bool(do_log1p)
        self.do_normalize = bool(do_normalize)
        self.target_sum = float(target_sum)
        self.max_len = max_len

        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.batch_buffer_size = int(batch_buffer_size)
        self.num_open_shards = max(1, int(num_open_shards))
        self.enable_open_shards_schedule = bool(enable_open_shards_schedule)
        self._num_open_shards_state: torch.Tensor | None = None
        if self.enable_open_shards_schedule:
            self._num_open_shards_state = torch.tensor(
                [self.num_open_shards], dtype=torch.int32
            )
            self._num_open_shards_state.share_memory_()

        self.celltype_stratified = bool(celltype_stratified)
        self.crop_size = crop_size
        self.val_crop_size = val_crop_size
        self.val_subset_manifest = val_subset_manifest
        self.zero_sampling = bool(zero_sampling)
        self.zero_sampling_fraction = float(zero_sampling_fraction)
        self.zero_sampling_strategy = zero_sampling_strategy
        self.count_weights_cap = count_weights_cap
        self.use_count_weights = bool(use_count_weights)
        self.cache_path = cache_path
        self.use_raw_expr = bool(use_raw_expr)
        self.tissue_code_path = tissue_code_path
        self.val_data_root = val_data_root
        self.val_census_version = val_census_version

        self.loader_kwargs = self._loader_kwargs(
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.val_loader_kwargs = self._loader_kwargs(
            num_workers=self.num_workers if val_num_workers is None else int(val_num_workers),
            pin_memory=pin_memory if val_pin_memory is None else bool(val_pin_memory),
            prefetch_factor=(
                prefetch_factor
                if val_prefetch_factor is None
                else int(val_prefetch_factor)
            ),
            persistent_workers=(
                persistent_workers
                if val_persistent_workers is None
                else bool(val_persistent_workers)
            ),
        )
        self.val_batch_buffer_size = (
            min(self.batch_buffer_size, self.batch_size * 2)
            if val_batch_buffer_size is None
            else int(val_batch_buffer_size)
        )

        self.train_dataset: ContextDataset | None = None
        self.val_dataset: ContextDataset | None = None
        self.val_full_dataset: ContextDataset | None = None
        self.transform: ContextTransform | None = None

        self.estimated_train_len: int | None = None
        self.estimated_val_len: int | None = None
        self.estimated_val_full_len: int | None = None
        self._epoch_base = 0
        self._cached_val_dataloaders: DataLoader | list[DataLoader] | None = None

    @staticmethod
    def _loader_kwargs(
        *,
        num_workers: int,
        pin_memory: bool,
        prefetch_factor: int,
        persistent_workers: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(prefetch_factor)
            kwargs["persistent_workers"] = bool(persistent_workers)
        return kwargs

    def _make_dataset(
        self,
        *,
        split: str,
        crop_size: int | None,
        batch_buffer_size: int,
        max_tokens_per_batch: int,
        rank: int,
        world_size: int,
        estimated_len: int | None,
        use_val_source: bool = False,
        epoch_base: int = 0,
    ) -> ContextDataset:
        return ContextDataset(
            data_root=self.data_root,
            census_version=self.census_version,
            split=split,
            pseudo_bulk_path=self.pseudo_bulk_path,
            gene_vocab_path=self.gene_vocab_path,
            cell_type_vocab_path=self.cell_type_vocab_path,
            min_genes=self.min_genes,
            val_split_ratio=self.val_split_ratio,
            split_seed=self.split_seed,
            val_subset_manifest=self.val_subset_manifest,
            do_log1p=self.do_log1p,
            do_normalize=self.do_normalize,
            target_sum=self.target_sum,
            max_tokens_per_batch=max_tokens_per_batch,
            batch_size=self.batch_size,
            batch_buffer_size=batch_buffer_size,
            num_open_shards=self.num_open_shards,
            num_open_shards_state=self._num_open_shards_state,
            celltype_stratified=self.celltype_stratified,
            crop_size=crop_size,
            zero_sampling=self.zero_sampling,
            zero_sampling_fraction=self.zero_sampling_fraction,
            zero_sampling_strategy=self.zero_sampling_strategy,
            count_weights_cap=self.count_weights_cap,
            use_count_weights=self.use_count_weights,
            cache_path=self.cache_path,
            tissue_code_path=self.tissue_code_path,
            val_data_root=self.val_data_root if use_val_source else None,
            val_census_version=self.val_census_version if use_val_source else None,
            rank=rank,
            world_size=world_size,
            estimated_len=estimated_len,
            epoch_base=epoch_base,
        )

    def setup(self, stage: str | None = None) -> None:
        rank = int(getattr(self.trainer, "global_rank", 0)) if self.trainer else 0
        world_size = int(getattr(self.trainer, "world_size", 1)) if self.trainer else 1

        self.transform = ContextTransform(
            max_len=self.max_len,
            use_raw_expr=self.use_raw_expr,
        )

        if stage in {"fit", None}:
            self.train_dataset = self._make_dataset(
                split="train",
                crop_size=self.crop_size,
                batch_buffer_size=self.batch_buffer_size,
                max_tokens_per_batch=self.max_tokens_per_batch,
                rank=rank,
                world_size=world_size,
                estimated_len=self.estimated_train_len,
                epoch_base=self._epoch_base,
            )

        if stage in {"fit", "validate", None}:
            val_split = "train" if self.val_use_train else "val"
            self.val_dataset = self._make_dataset(
                split=val_split,
                crop_size=self.val_crop_size,
                batch_buffer_size=self.val_batch_buffer_size,
                max_tokens_per_batch=min(self.max_tokens_per_batch, 1_000_000),
                rank=rank,
                world_size=world_size,
                estimated_len=self.estimated_val_len,
                use_val_source=True,
            )
            if self.val_full:
                self.val_full_dataset = self._make_dataset(
                    split=val_split,
                    crop_size=None,
                    batch_buffer_size=self.val_batch_buffer_size,
                    max_tokens_per_batch=min(self.max_tokens_per_batch, 1_000_000),
                    rank=rank,
                    world_size=world_size,
                    estimated_len=self.estimated_val_full_len,
                    use_val_source=True,
                )

        for dataset in (self.train_dataset, self.val_dataset, self.val_full_dataset):
            if dataset is not None:
                dataset._transform = self.transform
        self._cached_val_dataloaders = None

    def _collate_wrapper(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.transform is not None and "raw_input" not in batch[0]:
            batch = [self.transform(item) for item in batch]
        return context_collate_fn(batch)

    def set_epoch_base(self, epoch_base: int) -> None:
        self._epoch_base = int(epoch_base)
        if self.train_dataset is not None:
            self.train_dataset.set_epoch_base(self._epoch_base)

    def current_num_open_shards(self) -> int:
        state = self._num_open_shards_state
        if state is not None:
            return max(1, int(state.item()))
        return max(1, int(self.num_open_shards))

    def set_num_open_shards(self, k: int) -> None:
        if not self.enable_open_shards_schedule:
            raise RuntimeError(
                "set_num_open_shards requires enable_open_shards_schedule=True"
            )
        self.num_open_shards = max(1, int(k))
        assert self._num_open_shards_state is not None
        self._num_open_shards_state[0] = self.num_open_shards
        if self.train_dataset is not None:
            self.train_dataset.set_num_open_shards(self.num_open_shards)

    def _make_loader(
        self,
        dataset: ContextDataset,
        loader_kwargs: dict[str, Any],
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._collate_wrapper,
            **loader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        if self.trainer is not None:
            self.train_dataset.set_epoch(self.trainer.current_epoch)
        return self._make_loader(self.train_dataset, self.loader_kwargs)

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        assert self.val_dataset is not None
        if self.trainer is not None:
            self.val_dataset.set_epoch(self.trainer.current_epoch)
            if self.val_full_dataset is not None:
                self.val_full_dataset.set_epoch(self.trainer.current_epoch)

        loader_kwargs = self._effective_val_loader_kwargs()
        if self._cached_val_dataloaders is None:
            val_loader = self._make_loader(self.val_dataset, loader_kwargs)
            if self.val_full_dataset is None:
                self._cached_val_dataloaders = val_loader
            else:
                self._cached_val_dataloaders = [
                    val_loader,
                    self._make_loader(self.val_full_dataset, loader_kwargs),
                ]
        return self._cached_val_dataloaders

    def _effective_val_loader_kwargs(self) -> dict[str, Any]:
        loader_kwargs = dict(self.val_loader_kwargs)
        if (
            self.trainer is not None
            and int(getattr(self.trainer, "world_size", 1)) > 1
            and int(loader_kwargs.get("num_workers", 0)) > 0
        ):
            loader_kwargs["num_workers"] = 0
            loader_kwargs.pop("prefetch_factor", None)
            loader_kwargs.pop("persistent_workers", None)
        return loader_kwargs
