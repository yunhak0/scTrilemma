import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq


def get_total_cells_from_metadata(data_root, census_version, organism="homo_sapiens"):
    """Calculate total cells from metadata parquet file."""
    version_clean = census_version.replace("-", "")
    metadata_path = (
        Path(data_root)
        / version_clean
        / f"cell_metadata_{organism}_{version_clean}.parquet"
    )

    if not metadata_path.exists():
        logging.warning(
            f"Metadata file not found at {metadata_path}. Using default total cells."
        )
        return 1_000_000  # Default fallback

    try:
        # Read metadata only (O(1) memory)
        metadata = pq.read_metadata(metadata_path)
        return metadata.num_rows
    except Exception as e:
        logging.warning(f"Failed to read metadata: {e}. Using default total cells.")
        return 1_000_000


def _resolve_by_dataset_dir(data_root: str | Path, census_version: str) -> Path:
    """Resolve a by-dataset directory from either a root or direct by_dataset path."""
    data_root = Path(data_root)
    if data_root.name == "by_dataset":
        return data_root
    version_clean = census_version.replace("-", "")
    return data_root / version_clean / "by_dataset"


def _select_val_files_from_manifest(
    by_dataset_dir: Path,
    manifest_path: str | Path,
) -> list[Path]:
    """Replicate ContextDataset subset-manifest selection for val-length estimation."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Validation subset manifest not found: {manifest_path}")

    all_files = sorted(by_dataset_dir.glob("*/*.h5ad"))
    selected_files: set[Path] = set()
    selected_dirs: set[Path] = set()

    for raw_line in manifest_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        entry = Path(line)
        if entry.suffix == ".h5ad":
            file_path = entry if entry.is_absolute() else by_dataset_dir / entry
            selected_files.add(file_path.resolve())
        else:
            dir_path = entry if entry.is_absolute() else by_dataset_dir / entry
            selected_dirs.add(dir_path.resolve())

    filtered_files = [f for f in all_files if f.resolve() in selected_files]
    if selected_dirs:
        filtered_files.extend(
            f
            for f in all_files
            if f.parent.resolve() in selected_dirs and f.resolve() not in selected_files
        )

    deduped_files: list[Path] = []
    seen: set[Path] = set()
    for file_path in filtered_files:
        resolved = file_path.resolve()
        if resolved not in seen:
            deduped_files.append(file_path)
            seen.add(resolved)
    return deduped_files


def estimate_val_batches_from_data(
    data_root: str | Path,
    census_version: str,
    val_split_ratio: float,
    split_seed: int,
    batch_size: int,
    world_size: int,
    crop_size: int | None = None,
    max_tokens_per_batch: int | None = None,
    val_subset_manifest: str | None = None,
    val_data_root: str | Path | None = None,
    val_census_version: str | None = None,
) -> int:
    """Estimate per-rank val batch count from actual val dataset files.

    Handles two validation regimes:
    - legacy dataset-level split via ``val_split_ratio``
    - explicit subset-manifest validation via ``val_subset_manifest``

    Batch count is estimated from file sizes (file size ∝ cell count).

    Returns at least 1.
    """
    effective_root = val_data_root or data_root
    effective_version = val_census_version or census_version
    by_dataset_dir = _resolve_by_dataset_dir(effective_root, effective_version)

    if not by_dataset_dir.exists():
        logging.warning(
            f"by_dataset dir not found at {by_dataset_dir}; returning 1"
        )
        return 1

    dataset_dirs = sorted([d for d in by_dataset_dir.iterdir() if d.is_dir()])
    if not dataset_dirs:
        return 1

    if val_subset_manifest:
        val_files = _select_val_files_from_manifest(by_dataset_dir, val_subset_manifest)
    else:
        rng = random.Random(split_seed)
        rng.shuffle(dataset_dirs)
        n_val = int(len(dataset_dirs) * val_split_ratio)
        val_dirs = dataset_dirs[:n_val]
        if not val_dirs:
            return 1
        val_files = []
        for d in val_dirs:
            val_files.extend(sorted(d.glob("*.h5ad")))

    if not val_files:
        return 1

    val_bytes = sum(f.stat().st_size for f in val_files)

    # Compute bytes_per_cell dynamically from all h5ad files + metadata
    total_cells = get_total_cells_from_metadata(effective_root, effective_version)
    all_files: list[Path] = []
    for d in dataset_dirs:
        all_files.extend(sorted(d.glob("*.h5ad")))
    total_bytes = sum(f.stat().st_size for f in all_files) if all_files else 1

    bytes_per_cell = total_bytes / max(total_cells, 1)
    estimated_val_cells = int(val_bytes / max(bytes_per_cell, 1))

    # Determine effective batch size
    if crop_size is not None:
        # Crop mode: _flush_buffer uses batch_size directly
        effective_batch_size = batch_size
    elif max_tokens_per_batch is not None and crop_size is None:
        # Token-budget mode: effective batch ≈ max_tokens_per_batch / avg_gene_count
        # Conservative: use batch_size as upper bound
        effective_batch_size = batch_size
    else:
        effective_batch_size = batch_size

    # Replicate _smart_sharding greedy bin-packing to find the largest bin.
    # Using the max-bin cell count (instead of even world_size split) ensures
    # limit_val_batches covers 100% of the largest rank's data.
    if world_size > 1:
        file_weights = [(f, f.stat().st_size) for f in val_files]
        file_weights.sort(key=lambda x: x[1], reverse=True)
        bin_weights = [0] * world_size
        for _f, w in file_weights:
            min_idx = bin_weights.index(min(bin_weights))
            bin_weights[min_idx] += w
        max_bin_bytes = max(bin_weights) if bin_weights else val_bytes
        per_rank_cells = int(max_bin_bytes / max(bytes_per_cell, 1))
    else:
        per_rank_cells = estimated_val_cells

    raw_batches = per_rank_cells // max(effective_batch_size, 1)

    # 10% safety margin (over-estimate is safer than under-estimate for cycling)
    result = int(math.ceil(raw_batches * 1.10))
    return max(1, result)


def find_latest_checkpoint(
    outputs_dir: str,
    experiment_name: str,
    stage: str | None = None,
    ckpt_name: str = "last.ckpt",
) -> Optional[str]:
    """
    Find the most recent checkpoint for a given experiment.

    Searches: {outputs_dir}/{experiment_name}/{ckpt_name}, or
    {outputs_dir}/{experiment_name}/{stage}/{ckpt_name} when stage is set.

    Returns the checkpoint path if found, or None if not found.
    """
    if stage:
        ckpt_path = os.path.join(outputs_dir, experiment_name, stage, ckpt_name)
    else:
        ckpt_path = os.path.join(outputs_dir, experiment_name, ckpt_name)

    if os.path.exists(ckpt_path):
        return ckpt_path

    return None
