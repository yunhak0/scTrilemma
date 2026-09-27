from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import struct
import time
import zipfile
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)

_INTERLEAVE_DEBUG = os.environ.get("INTERLEAVE_DEBUG", "0") == "1"
CACHE_VERSION = 1
CACHE_ARRAY_KEYS = (
    "csr_data",
    "csr_indices",
    "csr_indptr",
    "donor_codes",
    "donor_id_map",
    "cell_type_labels",
    "library_sizes",
    "nnz_counts",
    "dataset_id",
    "n_cells",
    "vocab_size",
)
CACHE_MEMMAP_KEYS = frozenset(
    {
        "csr_data",
        "csr_indices",
        "csr_indptr",
        "donor_codes",
        "cell_type_labels",
        "library_sizes",
        "nnz_counts",
    }
)


def _parse_val_subset_manifest_ids(
    manifest_path_str: str,
    by_dataset_dir: Path,
) -> set[str]:
    manifest_path = Path(manifest_path_str).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"val_subset_manifest not found: {manifest_path}")

    ids: set[str] = set()
    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line)
        ids.add(entry.parent.name if entry.suffix == ".h5ad" else entry.name)
    return ids


class ContextDataset(IterableDataset[list[dict[str, Any]]]):
    """Cache-backed iterable dataset for scTrilemma.

    The official pipeline expects h5ad files to be preprocessed into per-shard
    ``.npz`` caches under ``<root>/<version>/by_dataset_cache``. Keeping the
    runtime path cache-only avoids a second, divergent h5ad parsing pipeline.
    """

    def __init__(
        self,
        data_root: str | Path,
        census_version: str,
        split: str = "train",
        pseudo_bulk_path: str | None = None,
        gene_vocab_path: str | None = None,
        cell_type_vocab_path: str | None = None,
        min_genes: int = 5,
        val_split_ratio: float = 0.1,
        split_seed: int = 42,
        val_subset_manifest: str | None = None,
        do_log1p: bool = True,
        do_normalize: bool = True,
        target_sum: float = 10000.0,
        max_tokens_per_batch: int = 500000,
        batch_size: int = 32,
        batch_buffer_size: int = 4096,
        num_open_shards: int = 1,
        num_open_shards_state: torch.Tensor | None = None,
        celltype_stratified: bool = False,
        crop_size: int | None = None,
        zero_sampling: bool = False,
        zero_sampling_fraction: float = 0.0,
        zero_sampling_strategy: str = "random",
        count_weights_cap: float | None = None,
        use_count_weights: bool = True,
        cache_path: str | None = None,
        tissue_code_path: str | None = None,
        val_data_root: str | Path | None = None,
        val_census_version: str | None = None,
        rank: int = 0,
        world_size: int = 1,
        estimated_len: int | None = None,
        shuffle_files: bool | None = None,
        shuffle_batches: bool | None = None,
        epoch_base: int = 0,
    ) -> None:
        super().__init__()
        if split in {"val", "test"} and val_data_root is not None:
            self.data_root = Path(val_data_root)
            self.census_version = val_census_version or census_version
        else:
            self.data_root = Path(data_root)
            self.census_version = census_version
        self.split = split
        self.min_genes = int(min_genes)
        self.do_log1p = bool(do_log1p)
        self.do_normalize = bool(do_normalize)
        self.target_sum = float(target_sum)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self.epoch_base = int(epoch_base)
        self.estimated_len = estimated_len
        self.shuffle_files = split == "train" if shuffle_files is None else bool(shuffle_files)
        self.shuffle_batches = split == "train" if shuffle_batches is None else bool(shuffle_batches)

        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.batch_size = int(batch_size)
        self.batch_buffer_size = int(batch_buffer_size)
        self.num_open_shards = max(1, int(num_open_shards))
        self.num_open_shards_state = num_open_shards_state
        self.celltype_stratified = bool(celltype_stratified)
        self.crop_size = crop_size
        self.zero_sampling = bool(zero_sampling)
        self.zero_sampling_fraction = float(zero_sampling_fraction)
        self.zero_sampling_strategy = zero_sampling_strategy
        self.count_weights_cap = count_weights_cap
        self.use_count_weights = bool(use_count_weights)
        self.val_split_ratio = float(val_split_ratio)
        self.split_seed = int(split_seed)
        self.val_subset_manifest = val_subset_manifest
        self._transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None

        if gene_vocab_path is None:
            raise ValueError("gene_vocab_path must be provided")
        with open(gene_vocab_path) as f:
            self.gene_vocab = json.load(f)
        self.vocab_size = len(self.gene_vocab)
        self.all_gene_indices = np.arange(self.vocab_size)
        self._gene_vocab_hash = self._hash_file(Path(gene_vocab_path))

        self.cell_type_vocab = None
        if cell_type_vocab_path and Path(cell_type_vocab_path).exists():
            with open(cell_type_vocab_path) as f:
                self.cell_type_vocab = json.load(f)

        self._cache_dir = self._find_cache_dir(cache_path)
        self._cache_available = True

        self.pseudo_bulk_dict: dict[str, torch.Tensor] = {}
        if pseudo_bulk_path and Path(pseudo_bulk_path).exists():
            if self.rank == 0:
                logger.info("Loading pseudo-bulk from %s", pseudo_bulk_path)
            self.pseudo_bulk_dict = torch.load(
                pseudo_bulk_path,
                map_location="cpu",
                weights_only=True,
            )
        else:
            logger.warning("Pseudo-bulk path %s not found. Context will be zeros.", pseudo_bulk_path)

        self.tissue_code_dict: dict[str, torch.Tensor] = {}
        if tissue_code_path and Path(tissue_code_path).exists():
            if self.rank == 0:
                logger.info("Loading tissue codes from %s", tissue_code_path)
            self.tissue_code_dict = torch.load(
                tissue_code_path,
                map_location="cpu",
                weights_only=True,
            )

        self.target_dirs, self.all_files = self._discover_files()
        if self.world_size > 1 and self.split in {"train", "val", "test"}:
            self._assign_files_for_epoch()
        else:
            self.my_files = list(self.all_files)

        if self.zero_sampling_strategy not in {"random", "deterministic"}:
            raise ValueError(
                "zero_sampling_strategy must be 'random' or 'deterministic', "
                f"got {self.zero_sampling_strategy}"
            )
        self.zero_sampling_order = self._build_zero_sampling_order()
        self.batch_id_to_idx = self._build_batch_id_index()

        if self.rank == 0:
            logger.info(
                "[%s] Found %d files in %d datasets; rank %d assigned %d files.",
                split,
                len(self.all_files),
                len(self.target_dirs),
                self.rank,
                len(self.my_files),
            )

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _find_cache_dir(self, cache_path: str | None) -> Path:
        candidates: list[Path] = []
        if cache_path is not None:
            candidates.append(Path(cache_path) / self.census_version / "by_dataset_cache")
        candidates.append(self.data_root / self.census_version / "by_dataset_cache")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Preprocessed cache not found. Expected one of: "
            + ", ".join(str(p) for p in candidates)
        )

    def _discover_files(self) -> tuple[list[Path], list[Path]]:
        by_dataset_dir = self.data_root / self.census_version / "by_dataset"
        if not by_dataset_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {by_dataset_dir}")

        dataset_dirs = sorted(d for d in by_dataset_dir.iterdir() if d.is_dir())
        if self.val_subset_manifest:
            manifest_ids = _parse_val_subset_manifest_ids(
                self.val_subset_manifest,
                by_dataset_dir,
            )
            if self.split == "train":
                target_dirs = [d for d in dataset_dirs if d.name not in manifest_ids]
            elif self.split in {"val", "test"}:
                target_dirs = [d for d in dataset_dirs if d.name in manifest_ids]
            else:
                target_dirs = dataset_dirs
        else:
            rng = random.Random(self.split_seed)
            rng.shuffle(dataset_dirs)
            n_val = int(len(dataset_dirs) * self.val_split_ratio)
            if self.split == "train":
                target_dirs = dataset_dirs[n_val:]
            elif self.split == "val":
                target_dirs = dataset_dirs[:n_val]
            else:
                target_dirs = dataset_dirs

        all_files: list[Path] = []
        for directory in target_dirs:
            all_files.extend(sorted(directory.glob("*.h5ad")))
        if self.val_subset_manifest and self.split in {"val", "test"}:
            target_dirs, all_files = self._filter_manifest_files(by_dataset_dir, target_dirs, all_files)
        return target_dirs, all_files

    def _filter_manifest_files(
        self,
        by_dataset_dir: Path,
        target_dirs: list[Path],
        all_files: list[Path],
    ) -> tuple[list[Path], list[Path]]:
        assert self.val_subset_manifest is not None
        manifest_path = Path(self.val_subset_manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        manifest_path = manifest_path.resolve()
        selected_files: set[Path] = set()
        selected_dirs: set[Path] = set()
        for raw_line in manifest_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entry = Path(line)
            if entry.suffix == ".h5ad":
                selected_files.add((entry if entry.is_absolute() else by_dataset_dir / entry).resolve())
            else:
                selected_dirs.add((entry if entry.is_absolute() else by_dataset_dir / entry).resolve())

        filtered = [f for f in all_files if f.resolve() in selected_files]
        filtered.extend(
            f for f in all_files
            if f.parent.resolve() in selected_dirs and f.resolve() not in selected_files
        )
        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in filtered:
            resolved = path.resolve()
            if resolved not in seen:
                deduped.append(path)
                seen.add(resolved)
        if not deduped:
            raise ValueError(
                f"Validation subset manifest {manifest_path} did not match any files"
            )
        matched_dirs = {f.parent.resolve() for f in deduped}
        return [d for d in target_dirs if d.resolve() in matched_dirs], deduped

    def _build_zero_sampling_order(self) -> np.ndarray | None:
        if self.zero_sampling_strategy == "random":
            return None
        if not self.pseudo_bulk_dict:
            return np.arange(self.vocab_size)

        sum_expr = None
        count = 0
        for value in self.pseudo_bulk_dict.values():
            if not isinstance(value, torch.Tensor) or value.numel() != self.vocab_size:
                continue
            sum_expr = value.float().clone() if sum_expr is None else sum_expr + value.float()
            count += 1
        if sum_expr is None or count == 0:
            return np.arange(self.vocab_size)
        return torch.argsort(sum_expr / count, descending=True).cpu().numpy()

    def _build_batch_id_index(self) -> dict[str, int]:
        keys = set(self.pseudo_bulk_dict.keys())
        keys.update(f"{d.name}_unknown" for d in self.target_dirs)
        return {key: i for i, key in enumerate(sorted(keys))}

    def _smart_sharding(self, all_files: list[Path]) -> list[Path]:
        file_weights = sorted(
            ((f, f.stat().st_size) for f in all_files),
            key=lambda x: x[1],
            reverse=True,
        )
        bins: list[list[Path]] = [[] for _ in range(self.world_size)]
        bin_weights = [0] * self.world_size
        for file_path, weight in file_weights:
            idx = bin_weights.index(min(bin_weights))
            bins[idx].append(file_path)
            bin_weights[idx] += weight
        return bins[self.rank]

    def _assign_files_for_epoch(self) -> None:
        if self.split == "train":
            self.my_files = self._smart_sharding(self.all_files)
            return
        real_rank = self.rank
        self.rank = (self.rank + self.epoch) % self.world_size
        self.my_files = self._smart_sharding(self.all_files)
        self.rank = real_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.split in {"val", "test"} and self.world_size > 1:
            self._assign_files_for_epoch()

    def set_epoch_base(self, epoch_base: int) -> None:
        self.epoch_base = int(epoch_base)

    def current_num_open_shards(self) -> int:
        if self.num_open_shards_state is not None:
            return max(1, int(self.num_open_shards_state.item()))
        return max(1, int(self.num_open_shards))

    def set_num_open_shards(self, k: int) -> None:
        self.num_open_shards = max(1, int(k))
        if self.num_open_shards_state is not None:
            self.num_open_shards_state[0] = self.num_open_shards

    def __len__(self) -> int:
        return int(self.estimated_len or 0)

    def _resolve_cache_path(self, h5ad_path: Path) -> Path:
        return self._cache_dir / h5ad_path.parent.name / f"{h5ad_path.stem}.npz"

    @staticmethod
    def _cache_memmap_dir(cache_file: Path) -> Path:
        return cache_file.with_suffix(".mmap")

    @staticmethod
    def _npz_member_memmap(cache_file: Path, member_name: str) -> np.memmap:
        with zipfile.ZipFile(cache_file) as archive:
            info = archive.getinfo(member_name)
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{cache_file}:{member_name} is compressed")

        with open(cache_file, "rb") as f:
            f.seek(info.header_offset)
            local_header = f.read(30)
            if len(local_header) != 30:
                raise ValueError(f"{cache_file}:{member_name} has a truncated zip header")
            (
                signature,
                _version,
                _flags,
                _compression,
                _mtime,
                _mdate,
                _crc,
                _compressed_size,
                _file_size,
                filename_len,
                extra_len,
            ) = struct.unpack("<IHHHHHIIIHH", local_header)
            if signature != 0x04034B50:
                raise ValueError(f"{cache_file}:{member_name} has an invalid zip header")

            f.seek(info.header_offset + 30 + filename_len + extra_len)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(f)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise ValueError(f"{cache_file}:{member_name} uses unsupported npy {version}")
            if dtype.hasobject:
                raise ValueError(f"{cache_file}:{member_name} has object dtype")
            data_offset = f.tell()

        order = "F" if fortran_order else "C"
        return np.memmap(
            cache_file,
            dtype=dtype,
            mode="r",
            shape=shape,
            offset=data_offset,
            order=order,
        )

    def _load_cached_arrays_from_npz(self, cache_file: Path) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(cache_file) as archive:
            names = set(archive.namelist())
            missing = [f"{key}.npy" for key in CACHE_ARRAY_KEYS if f"{key}.npy" not in names]
            if missing:
                raise KeyError(f"{cache_file} missing cache members: {missing}")
            for key in CACHE_ARRAY_KEYS:
                member_name = f"{key}.npy"
                if key in CACHE_MEMMAP_KEYS:
                    arrays[key] = self._npz_member_memmap(cache_file, member_name)
                else:
                    with archive.open(member_name) as f:
                        arrays[key] = np.load(f, allow_pickle=True)
        return arrays

    def _load_cached_arrays(self, cache_file: Path) -> dict[str, np.ndarray]:
        mmap_dir = self._cache_memmap_dir(cache_file)
        if (mmap_dir / "_SUCCESS.json").exists():
            try:
                return {
                    key: np.load(
                        mmap_dir / f"{key}.npy",
                        allow_pickle=True,
                        mmap_mode="r" if key in CACHE_MEMMAP_KEYS else None,
                    )
                    for key in CACHE_ARRAY_KEYS
                }
            except Exception as exc:
                logger.warning(
                    "Failed to load memmap cache %s; falling back to %s: %s",
                    mmap_dir,
                    cache_file,
                    exc,
                )

        try:
            return self._load_cached_arrays_from_npz(cache_file)
        except Exception as exc:
            logger.warning(
                "Failed to memmap %s directly; falling back to np.load: %s",
                cache_file,
                exc,
            )
            npz = np.load(cache_file, allow_pickle=True)
            return {key: npz[key] for key in CACHE_ARRAY_KEYS}

    def _validate_cache_file(self, cache_file: Path, h5ad_path: Path) -> bool:
        meta_path = cache_file.with_suffix(".meta.json")
        if not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text())
            return (
                meta.get("cache_version") == CACHE_VERSION
                and meta.get("gene_vocab_hash") == self._gene_vocab_hash
                and meta.get("source_h5ad_size") == h5ad_path.stat().st_size
            )
        except Exception:
            return False

    def _valid_cached_pairs(self, files: list[Path]) -> list[tuple[Path, Path]]:
        pairs: list[tuple[Path, Path]] = []
        for file_path in files:
            cache_file = self._resolve_cache_path(file_path)
            if cache_file.exists() and self._validate_cache_file(cache_file, file_path):
                pairs.append((cache_file, file_path))
            else:
                logger.warning("Skipping %s because cache is missing or stale", file_path)
        return pairs

    def _select_zero_indices_batch(
        self,
        nonzero_indices_list: list[np.ndarray],
        num_pad_list: list[int],
    ) -> list[np.ndarray]:
        results: list[np.ndarray] = [
            np.empty((0,), dtype=np.int64) for _ in nonzero_indices_list
        ]
        if not hasattr(self, "_zero_mask_buf"):
            self._zero_mask_buf = np.zeros(self.vocab_size, dtype=bool)
        mask_buf = self._zero_mask_buf

        if self.zero_sampling_strategy == "random":
            for idx, nonzero in enumerate(nonzero_indices_list):
                num_pad = num_pad_list[idx]
                if num_pad <= 0:
                    continue
                mask_buf[:] = True
                mask_buf[nonzero] = False
                available = self.all_gene_indices[mask_buf]
                replace = len(available) < num_pad
                results[idx] = np.random.choice(available, num_pad, replace=replace)
            return results

        ordered = self.zero_sampling_order
        if ordered is None:
            ordered = np.arange(self.vocab_size)
        for idx, nonzero in enumerate(nonzero_indices_list):
            num_pad = num_pad_list[idx]
            if num_pad <= 0:
                continue
            mask_buf[:] = False
            mask_buf[nonzero] = True
            available = ordered[~mask_buf[ordered]]
            if len(available) >= num_pad:
                results[idx] = available[:num_pad].copy()
            elif len(available) == 0:
                results[idx] = np.random.choice(self.all_gene_indices, num_pad, replace=True)
            else:
                reps = int(np.ceil(num_pad / len(available)))
                results[idx] = np.tile(available, reps)[:num_pad]
        return results

    def _select_zero_indices(
        self,
        nonzero_indices_full: np.ndarray,
        num_pad: int,
    ) -> np.ndarray:
        if num_pad <= 0:
            return np.empty((0,), dtype=np.int64)
        return self._select_zero_indices_batch([nonzero_indices_full], [num_pad])[0]

    def _apply_crop_pad(
        self,
        nonzero_indices_full: np.ndarray,
        nonzero_values_full: np.ndarray,
        nonzero_raw_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.crop_size is None:
            return nonzero_indices_full, nonzero_values_full, nonzero_raw_full

        final_indices = nonzero_indices_full
        final_values = nonzero_values_full
        final_raw = nonzero_raw_full
        curr_len = len(nonzero_indices_full)

        if self.zero_sampling:
            target_zero = int(round(self.crop_size * self.zero_sampling_fraction))
            target_nonzero = max(self.crop_size - target_zero, 0)
            if curr_len > target_nonzero:
                choice = np.random.choice(curr_len, target_nonzero, replace=False)
                final_indices = final_indices[choice]
                final_values = final_values[choice]
                final_raw = final_raw[choice]
        elif curr_len > self.crop_size:
            choice = np.random.choice(curr_len, self.crop_size, replace=False)
            final_indices = final_indices[choice]
            final_values = final_values[choice]
            final_raw = final_raw[choice]

        num_pad = self.crop_size - len(final_indices)
        if num_pad > 0:
            pad_indices = self._select_zero_indices(nonzero_indices_full, num_pad)
            final_indices = np.concatenate([final_indices, pad_indices])
            final_values = np.concatenate(
                [final_values, np.zeros(num_pad, dtype=final_values.dtype)]
            )
            final_raw = np.concatenate(
                [final_raw, np.zeros(num_pad, dtype=final_raw.dtype)]
            )
        return final_indices, final_values, final_raw

    def _compute_count_weights(
        self,
        final_raw: np.ndarray,
        full_nonzero_count: int,
    ) -> np.ndarray | None:
        if not (self.zero_sampling and self.use_count_weights):
            return None
        total_nonzero = float(full_nonzero_count)
        total_zero = float(self.vocab_size - full_nonzero_count)
        sampled_nonzero = float((final_raw > 0).sum())
        sampled_zero = float((final_raw == 0).sum())
        w_nonzero = total_nonzero / sampled_nonzero if sampled_nonzero > 0 else 1.0
        w_zero = total_zero / sampled_zero if sampled_zero > 0 else 1.0
        weights = np.where(final_raw > 0, w_nonzero, w_zero).astype(np.float32)
        if self.count_weights_cap is not None:
            weights = np.clip(weights, None, self.count_weights_cap)
        return weights

    def _context_tables(
        self,
        dataset_id: str,
        donor_id_map: np.ndarray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[int]]:
        unknown_key = f"{dataset_id}_unknown"
        unknown_pb = self.pseudo_bulk_dict.get(unknown_key)
        if unknown_pb is None:
            unknown_pb = torch.zeros(self.vocab_size)

        tc_dim = 0
        if self.tissue_code_dict:
            first_tc = next(iter(self.tissue_code_dict.values()))
            tc_dim = int(first_tc.numel())
        zero_tc = torch.zeros(tc_dim) if tc_dim > 0 else None

        pb_table: list[torch.Tensor] = []
        tc_table: list[torch.Tensor] = []
        batch_ids: list[str] = []
        batch_labels: list[int] = []
        for donor in donor_id_map:
            key = f"{dataset_id}_{donor}"
            if key in self.pseudo_bulk_dict:
                pb = self.pseudo_bulk_dict[key]
                batch_id = key
            elif unknown_key in self.pseudo_bulk_dict:
                pb = self.pseudo_bulk_dict[unknown_key]
                batch_id = unknown_key
            else:
                pb = unknown_pb
                batch_id = unknown_key
            pb_table.append(pb)
            batch_ids.append(batch_id)
            batch_labels.append(self.batch_id_to_idx.get(batch_id, -1))

            if zero_tc is not None:
                if key in self.tissue_code_dict:
                    tc_table.append(self.tissue_code_dict[key])
                elif unknown_key in self.tissue_code_dict:
                    tc_table.append(self.tissue_code_dict[unknown_key])
                else:
                    tc_table.append(zero_tc)
        return pb_table, tc_table, batch_ids, batch_labels

    def _cell_order(
        self,
        n_obs: int,
        ct_labels: np.ndarray,
        source_path: Path,
    ) -> np.ndarray:
        selected = np.arange(n_obs)
        if not self.celltype_stratified or ct_labels is None or len(selected) == 0:
            return selected

        groups: dict[int, list[int]] = defaultdict(list)
        for cell_i in selected:
            groups[int(ct_labels[cell_i])].append(int(cell_i))

        rng = random.Random(
            hash((str(source_path.resolve()), self.epoch, self.epoch_base, self.rank))
            & 0xFFFFFFFF
        )
        keys = list(groups.keys())
        rng.shuffle(keys)
        for key in keys:
            rng.shuffle(groups[key])

        iters = {key: iter(groups[key]) for key in keys}
        order: list[int] = []
        while iters:
            done: list[int] = []
            for key in list(iters.keys()):
                try:
                    order.append(next(iters[key]))
                except StopIteration:
                    done.append(key)
            for key in done:
                del iters[key]
        return np.asarray(order, dtype=selected.dtype)

    def _iter_cached_cells(
        self,
        cache_file: Path,
        original_h5ad: Path,
    ) -> Iterator[dict[str, Any]]:
        data = self._load_cached_arrays(cache_file)
        csr_data = data["csr_data"]
        csr_indices = data["csr_indices"]
        csr_indptr = data["csr_indptr"]
        donor_codes = data["donor_codes"]
        donor_id_map = data["donor_id_map"]
        ct_labels = data["cell_type_labels"]
        library_sizes = data["library_sizes"]
        nnz_counts = data["nnz_counts"]
        dataset_id = str(np.asarray(data["dataset_id"]).item())
        n_obs = int(np.asarray(data["n_cells"]).item())

        pb_table, tc_table, batch_ids, batch_labels = self._context_tables(
            dataset_id,
            donor_id_map,
        )
        for cell_i in self._cell_order(n_obs, ct_labels, original_h5ad):
            if nnz_counts[cell_i] < self.min_genes:
                continue
            lib_size = float(library_sizes[cell_i])
            if lib_size < 1e-6:
                continue

            row_start = int(csr_indptr[cell_i])
            row_end = int(csr_indptr[cell_i + 1])
            gene_indices = csr_indices[row_start:row_end].copy()
            raw_values = csr_data[row_start:row_end].copy()

            expr = (raw_values / lib_size) * self.target_sum if self.do_normalize else raw_values.copy()
            if self.do_log1p:
                expr = np.log1p(expr)
            nonzero_mask = expr > 0
            if nonzero_mask.sum() < self.min_genes:
                continue

            nonzero_indices = gene_indices[nonzero_mask]
            nonzero_values = expr[nonzero_mask]
            nonzero_raw = raw_values[nonzero_mask]
            full_nonzero_count = len(nonzero_indices)

            final_indices, final_values, final_raw = self._apply_crop_pad(
                nonzero_indices,
                nonzero_values,
                nonzero_raw,
            )
            donor_code = int(donor_codes[cell_i])
            item = {
                "gene_indices": torch.from_numpy(final_indices.astype(np.int64)).long(),
                "expr_values": torch.from_numpy(final_values).float(),
                "raw_counts": torch.from_numpy(final_raw).float(),
                "library_size": float(final_raw.sum()),
                "library_size_full": lib_size,
                "count_weights": self._compute_count_weights(
                    final_raw,
                    full_nonzero_count,
                ),
                "pseudo_bulk": pb_table[donor_code],
                "tissue_code": tc_table[donor_code] if tc_table else None,
                "dataset_id": dataset_id,
                "donor_id": str(donor_id_map[donor_code]),
                "batch_id": batch_ids[donor_code],
                "batch_label": int(batch_labels[donor_code]),
                "cell_type_label": int(ct_labels[cell_i]) if ct_labels is not None else -1,
                "length": len(final_indices),
            }
            if self._transform is not None:
                item = self._transform(item)
            yield item

    def _iter_cached_file(
        self,
        cache_file: Path,
        original_h5ad: Path,
        buffer: list[dict[str, Any]],
    ) -> Iterator[list[dict[str, Any]]]:
        for item in self._iter_cached_cells(cache_file, original_h5ad):
            buffer.append(item)
            if len(buffer) >= self.batch_buffer_size:
                yield from self._flush_buffer(buffer)
                buffer.clear()

    def _iter_interleaved_cached_pass(
        self,
        my_worker_files: list[Path],
        buffer: list[dict[str, Any]],
        tag_if_padding: Callable[[Iterator[list[dict[str, Any]]]], Iterator[list[dict[str, Any]]]],
    ) -> Iterator[list[dict[str, Any]]]:
        start = time.perf_counter()
        cached_pairs = self._valid_cached_pairs(my_worker_files)
        if not cached_pairs:
            logger.warning("Interleave mode active but no cached files are available.")
            return

        file_queue: deque[tuple[Path, Path]] = deque(cached_pairs)
        active: list[Iterator[dict[str, Any]]] = []
        paused: list[Iterator[dict[str, Any]]] = []

        def target_k() -> int:
            return min(self.current_num_open_shards(), len(cached_pairs))

        def sync_active() -> None:
            while len(active) > target_k():
                paused.append(active.pop())
            while len(active) < target_k() and paused:
                active.append(paused.pop())
            while len(active) < target_k() and file_queue:
                cache_file, h5ad_path = file_queue.popleft()
                active.append(self._iter_cached_cells(cache_file, h5ad_path))

        sync_active()
        rounds = 0
        cells = 0
        while active or file_queue or paused:
            sync_active()
            if not active:
                break
            next_active: list[Iterator[dict[str, Any]]] = []
            for gen in active:
                try:
                    buffer.append(next(gen))
                    cells += 1
                    if len(buffer) >= self.batch_buffer_size:
                        yield from tag_if_padding(self._flush_buffer(buffer))
                        buffer.clear()
                    next_active.append(gen)
                except StopIteration:
                    continue
                except Exception as exc:
                    logger.warning("Interleave generator error: %s", exc)
            active = next_active
            rounds += 1

        if _INTERLEAVE_DEBUG:
            logger.info(
                "[interleave] pass done rounds=%d cells=%d elapsed=%.1fs",
                rounds,
                cells,
                time.perf_counter() - start,
            )

    def __iter__(self) -> Iterator[list[dict[str, Any]]]:
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        rng = random.Random(self.epoch + self.epoch_base + self.rank * 1000)
        files_to_process = list(self.my_files)
        if self.shuffle_files:
            rng.shuffle(files_to_process)
        my_worker_files = files_to_process[worker_id::num_workers]

        buffer: list[dict[str, Any]] = []
        should_cycle = self.split == "train" or (
            self.split in {"val", "test"}
            and (
                self.world_size > 1
                or (self.estimated_len is not None and self.estimated_len > 0)
            )
        )
        padding_state = [False]

        def tag_if_padding(
            batch_gen: Iterator[list[dict[str, Any]]],
        ) -> Iterator[list[dict[str, Any]]]:
            for batch in batch_gen:
                if padding_state[0]:
                    for item in batch:
                        item["_is_padding"] = True
                yield batch

        while True:
            if self.shuffle_files:
                rng.shuffle(my_worker_files)

            use_interleave = (
                self.current_num_open_shards() > 1
                or self.num_open_shards_state is not None
            )
            if use_interleave:
                yield from self._iter_interleaved_cached_pass(
                    my_worker_files,
                    buffer,
                    tag_if_padding,
                )
            else:
                for cache_file, file_path in self._valid_cached_pairs(my_worker_files):
                    yield from tag_if_padding(
                        self._iter_cached_file(cache_file, file_path, buffer)
                    )

            if buffer:
                yield from tag_if_padding(self._flush_buffer(buffer))
                buffer.clear()
            if not should_cycle:
                break
            padding_state[0] = True

    def _flush_buffer(
        self,
        buffer: list[dict[str, Any]],
    ) -> Iterator[list[dict[str, Any]]]:
        if self.crop_size is not None:
            if self.shuffle_batches and not self.celltype_stratified:
                random.shuffle(buffer)
            batches = [
                buffer[i : i + self.batch_size]
                for i in range(0, len(buffer), self.batch_size)
            ]
            if self.shuffle_batches:
                random.shuffle(batches)
            yield from batches
            return

        buffer.sort(key=lambda item: item["length"])
        batches: list[list[dict[str, Any]]] = []
        current_batch: list[dict[str, Any]] = []
        current_tokens = 0
        for item in buffer:
            item_len = int(item["length"])
            if (
                len(current_batch) + 1 > self.batch_size
                or current_tokens + item_len > self.max_tokens_per_batch
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = item_len
            else:
                current_batch.append(item)
                current_tokens += item_len
        if current_batch:
            batches.append(current_batch)
        if self.shuffle_batches:
            random.shuffle(batches)
        yield from batches
