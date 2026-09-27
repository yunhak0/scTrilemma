"""Shared helpers for the Table 1 embedding-scoring modules.

Every scorer reads one embedding cache directory (``<dataset>.npz`` with aligned arrays
``embeddings`` (N, D) float32, ``labels`` (N,) str and ``batches`` (N,) str, as written by
``sctrilemma.benchmark.export_embeddings``) and the precomputed metric samples of
``experiments.scoring.metric_samples``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.preprocessing import LabelEncoder

from experiments.common import read_ids, stable_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IDS_FILE = ROOT / "configs/zsb/full_89_ids.txt"
DEFAULT_CACHE_DIR = ROOT / "outputs/experiments/embeddings/sctrilemma/embeddings"
DEFAULT_SCORING_DIR = ROOT / "outputs/experiments/scoring"
DEFAULT_METRIC_SAMPLES_DIR = DEFAULT_SCORING_DIR / "metric_samples"
DEFAULT_MODEL_NAME = "sctrilemma"

# Input protocol: the cache holds at most 100,000 cells per dataset, drawn with seed 0.
INPUT_MAX_CELLS = 100_000
INPUT_SAMPLE_SEED = 0

# Metric-sample protocol: stratified <= 10,000-cell samples, 20 sample seeds, 20 K-means seeds.
MAX_PER_TYPE = 500
MIN_CELLS_FOR_RARE = 50
TOTAL_MAX = 10_000
SAMPLE_SEEDS = tuple(range(20))
KMEANS_SEEDS = tuple(range(20))


def resolve_dataset_ids(ids_file: Path, explicit: Sequence[str] | None) -> list[str]:
    """Return the requested dataset IDs, validated against the manifest order."""
    manifest = read_ids(ids_file)
    if not explicit:
        return manifest
    requested = list(explicit)
    unknown = sorted(set(requested) - set(manifest))
    if unknown or len(set(requested)) != len(requested):
        raise ValueError(f"Dataset IDs must be a unique subset of {ids_file}; unknown={unknown}")
    return requested


def dataset_seed(dataset_id: str, sample_seed: int) -> int:
    """Dataset-specific NumPy seed: ``(sample_seed + int(sha1(dataset_id)[:8], 16)) mod 2**32``."""
    return stable_seed(sample_seed, dataset_id)


def encode(values: np.ndarray) -> np.ndarray:
    """Integer-code a string array (sorted unique values -> 0..K-1)."""
    return LabelEncoder().fit_transform(values).astype(np.int64, copy=False)


def cache_path(cache_dir: Path, dataset_id: str) -> Path:
    return cache_dir / f"{dataset_id}.npz"


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an aligned embedding cache as (embeddings float32, labels str, batches str)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding cache: {path}")
    with np.load(path, allow_pickle=False) as cached:
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        labels = np.asarray(cached["labels"], dtype=str)
        batches = np.asarray(cached["batches"], dtype=str)
    if embeddings.ndim != 2 or not (len(embeddings) == len(labels) == len(batches)):
        raise ValueError(f"{path}: cache arrays are not aligned ({embeddings.shape})")
    return embeddings, labels, batches


def legacy_stratified_sample_indices(
    label_codes: np.ndarray,
    *,
    max_per_type: int = MAX_PER_TYPE,
    min_cells_for_rare: int = MIN_CELLS_FOR_RARE,
    total_max: int = TOTAL_MAX,
    seed: int,
) -> np.ndarray | None:
    """Stratified cell sample of the paper's benchmark, drawn with the legacy NumPy RNG.

    Reproduces the benchmark sampler that was executed under ``np.random.seed(seed)``:
    every label with fewer than ``min_cells_for_rare`` cells keeps all of them, the other
    labels are capped at ``max_per_type``, and when the total exceeds ``total_max`` the
    per-label budget is redistributed (rare labels reserved, one cell reserved per other
    label, the remainder split proportionally with largest-remainder rounding). The cells
    of each label are then drawn with ``RandomState(seed).choice(..., replace=False)`` in
    sorted-label order, which yields the same stream as the global legacy RNG. Returns
    ``None`` when no label is valid (code >= 0); labels are visited in sorted order and
    the returned indices are ordered by label, not by cache row.
    """
    labels_np = np.asarray(label_codes)
    valid_mask = labels_np >= 0
    if not valid_mask.any():
        return None
    valid_indices = np.where(valid_mask)[0]
    valid_labels = labels_np[valid_mask]

    unique_types, counts = np.unique(valid_labels, return_counts=True)
    type_to_indices = {ct: valid_indices[valid_labels == ct] for ct in unique_types}

    target_counts: dict[int, int] = {}
    rare_types: set[int] = set()
    for ct, count in zip(unique_types, counts, strict=True):
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

    rng = np.random.RandomState(seed)
    sampled: list[int] = []
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
        sampled.extend(chosen.tolist())
    return np.asarray(sampled, dtype=np.int64)


def metric_sample_path(sample_dir: Path, dataset_id: str, sample_seed: int) -> Path:
    return sample_dir / f"{dataset_id}_sample_seed_{sample_seed:02d}.npz"


def load_metric_sample(sample_dir: Path, dataset_id: str, sample_seed: int, n_rows: int) -> np.ndarray:
    """Load one precomputed metric sample and validate it against the cache size."""
    path = metric_sample_path(sample_dir, dataset_id, sample_seed)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing metric sample {path}; run `python -m experiments.scoring.metric_samples` first"
        )
    with np.load(path, allow_pickle=False) as cached:
        indices = np.asarray(cached["cache_row_indices"], dtype=np.int64)
        recorded_seed = int(cached["sample_seed"])
    if recorded_seed != sample_seed or not 0 < len(indices) <= TOTAL_MAX:
        raise ValueError(f"{dataset_id}/sample{sample_seed}: invalid metric-sample file {path}")
    if int(indices.min()) < 0 or int(indices.max()) >= n_rows:
        raise ValueError(f"{dataset_id}/sample{sample_seed}: sample indices out of range for {n_rows} rows")
    return indices


def atomic_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    """Write a CSV atomically (temporary file in the same directory, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
