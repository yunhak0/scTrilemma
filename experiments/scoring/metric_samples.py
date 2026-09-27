# ruff: noqa: E402
"""Generate the 20 stratified metric samples per dataset used by every Table 1 scorer.

For each dataset and sample seed ``s`` in 0..19 the cache labels are integer-coded, a
dataset-specific NumPy seed ``(s + int(sha1(dataset_id)[:8], 16)) mod 2**32`` is derived,
and the benchmark's stratified sampler (<= 500 cells per cell type, rare types with fewer
than 50 cells kept whole, <= 10,000 cells in total) is drawn with the legacy NumPy RNG
seeded that way. The sampled cache rows are written to
``<output-dir>/<dataset>_sample_seed_<s:02d>.npz`` with arrays ``cache_row_indices``
(int64), ``dataset_numpy_seed`` (uint32 scalar) and ``sample_seed`` (int64 scalar).

    python -m experiments.scoring.metric_samples --cache-dir <dir with <dataset>.npz>
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ.get("CPU_THREADS", "4"))

import numpy as np

from experiments.scoring._common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_METRIC_SAMPLES_DIR,
    MAX_PER_TYPE,
    MIN_CELLS_FOR_RARE,
    SAMPLE_SEEDS,
    TOTAL_MAX,
    atomic_csv,
    atomic_json,
    cache_path,
    dataset_seed,
    encode,
    legacy_stratified_sample_indices,
    load_cache,
    metric_sample_path,
    resolve_dataset_ids,
)

MANIFEST_FIELDS = [
    "dataset",
    "sample_seed",
    "dataset_numpy_seed",
    "n_cache_cells",
    "n_sample_cells",
    "n_labels",
    "n_batches",
    "status",
]


def save_metric_sample(path: Path, indices: np.ndarray, numpy_seed: int, sample_seed: int) -> None:
    """Write one metric-sample file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        cache_row_indices=np.asarray(indices, dtype=np.int64),
        dataset_numpy_seed=np.asarray(numpy_seed, dtype=np.uint32),
        sample_seed=np.asarray(sample_seed, dtype=np.int64),
    )
    temporary.replace(path)


def sample_dataset(
    dataset_id: str,
    labels: np.ndarray,
    sample_seed: int,
    *,
    max_per_type: int,
    min_cells_for_rare: int,
    total_max: int,
) -> tuple[np.ndarray, int]:
    """Return (cache_row_indices, dataset_numpy_seed) for one dataset and sample seed."""
    seed = dataset_seed(dataset_id, sample_seed)
    indices = legacy_stratified_sample_indices(
        encode(labels),
        max_per_type=max_per_type,
        min_cells_for_rare=min_cells_for_rare,
        total_max=total_max,
        seed=seed,
    )
    if indices is None:
        indices = np.arange(len(labels), dtype=np.int64)
    return indices, seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Directory with <dataset>.npz embedding caches")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_METRIC_SAMPLES_DIR)
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=list(SAMPLE_SEEDS))
    parser.add_argument("--max-per-type", type=int, default=MAX_PER_TYPE)
    parser.add_argument("--min-cells-for-rare", type=int, default=MIN_CELLS_FOR_RARE)
    parser.add_argument("--total-max", type=int, default=TOTAL_MAX)
    parser.add_argument("--force", action="store_true", help="Rewrite existing sample files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(seed < 0 for seed in args.sample_seeds) or len(set(args.sample_seeds)) != len(args.sample_seeds):
        raise ValueError("--sample-seeds must be unique non-negative integers")
    dataset_ids = resolve_dataset_ids(args.dataset_ids_file, args.dataset_ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "metadata.json", {
        "analysis": "stratified metric samples for the Table 1 repeat protocol",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cache_dir": str(args.cache_dir.resolve()),
        "dataset_ids_file": str(args.dataset_ids_file),
        "n_datasets": len(dataset_ids),
        "sample_seeds": list(args.sample_seeds),
        "dataset_seed_derivation": "(sample_seed + int(sha1(dataset_id)[:8], 16)) mod 2**32",
        "sampler": "legacy NumPy RNG (np.random.seed) stratified subsample",
        "max_per_type": args.max_per_type,
        "min_cells_for_rare": args.min_cells_for_rare,
        "total_max": args.total_max,
    })

    rows: list[dict[str, object]] = []
    for position, dataset_id in enumerate(dataset_ids, start=1):
        _, labels, batches = load_cache(cache_path(args.cache_dir, dataset_id))
        n_labels = int(np.unique(labels).size)
        n_batches = int(np.unique(batches).size)
        written = 0
        for sample_seed in args.sample_seeds:
            path = metric_sample_path(output_dir, dataset_id, sample_seed)
            if path.exists() and not args.force:
                with np.load(path, allow_pickle=False) as cached:
                    indices = np.asarray(cached["cache_row_indices"], dtype=np.int64)
                    seed = int(cached["dataset_numpy_seed"])
                status = "reused"
            else:
                indices, seed = sample_dataset(
                    dataset_id,
                    labels,
                    sample_seed,
                    max_per_type=args.max_per_type,
                    min_cells_for_rare=args.min_cells_for_rare,
                    total_max=args.total_max,
                )
                save_metric_sample(path, indices, seed, sample_seed)
                status = "written"
                written += 1
            rows.append({
                "dataset": dataset_id,
                "sample_seed": sample_seed,
                "dataset_numpy_seed": seed,
                "n_cache_cells": int(len(labels)),
                "n_sample_cells": int(len(indices)),
                "n_labels": n_labels,
                "n_batches": n_batches,
                "status": status,
            })
        atomic_csv(output_dir / "metric_sample_manifest.csv", MANIFEST_FIELDS, rows)
        print(
            f"[{position}/{len(dataset_ids)}] {dataset_id}: {written} written, "
            f"{len(args.sample_seeds) - written} reused, cache={len(labels):,} cells",
            flush=True,
        )
    print(f"Done: {len(rows)} metric samples at {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
