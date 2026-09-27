# ruff: noqa: E402
"""scIB principal-component-regression (PCR) comparison over the 20 metric samples.

For every dataset whose cache has more than one batch and for each of the 20 metric samples,
the source counts of the sampled cells (``load_dataset`` + the 100,000-cell input cap at
seed 0, i.e. the rows the cache was exported from) are normalised (``normalize_total`` 1e4,
``log1p``) and reduced with PCA (arpack, all genes); ``pcr_pre`` is the PCR of the PCA
coordinates on the categorical batch covariate, ``pcr_post`` the PCR of the embedding on
the same cells, and ``value = max(0, (pcr_pre - pcr_post) / pcr_pre)``.

One CSV shard per dataset is written under ``shards/<model>/`` (resumable; datasets can be
split across ``--worker-count`` processes) and the aggregate step assembles
``pcr_repeat20_long.csv``, ``pcr_dataset_repeat_summary.csv`` and ``pcr_repeat20_summary.csv``.

    SCTRILEMMA_DATA_ROOT=/path/to/cellxgene \\
    python -m experiments.scoring.pcr_repeat --cache-dir <dir> --metric-samples-dir <dir>
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "JAX_NUM_THREADS",
):
    os.environ.setdefault(_name, os.environ.get("CPU_THREADS", "4"))
# The PCR is tiny; the reference values were produced on the CPU jax backend.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4")

import numpy as np
import pandas as pd
import scanpy as sc
from scib_metrics.utils import principal_component_regression

from experiments.scoring._common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_METRIC_SAMPLES_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SCORING_DIR,
    INPUT_MAX_CELLS,
    INPUT_SAMPLE_SEED,
    SAMPLE_SEEDS,
    atomic_csv,
    atomic_json,
    cache_path,
    load_cache,
    load_metric_sample,
    read_csv_rows,
    resolve_dataset_ids,
)
from sctrilemma.benchmark.data_utils import DEFAULT_TARGET_PATH, load_dataset
from sctrilemma.benchmark.run import _batch_key_for_adata, _sample_cells_for_analysis

LABEL_KEY = "cell_type_ontology_term_id"
LONG_FIELDS = [
    "dataset",
    "model",
    "sample_seed",
    "value",
    "pcr_pre",
    "pcr_post",
    "n_sample_requested",
    "n_sample_common",
    "n_batches",
    "seconds",
    "status",
]


def eligible_ids(cache_dir: Path, dataset_ids: Sequence[str]) -> list[str]:
    """Datasets whose cache has more than one batch (PCR is undefined otherwise)."""
    eligible: list[str] = []
    for dataset_id in dataset_ids:
        with np.load(cache_path(cache_dir, dataset_id), allow_pickle=False) as cached:
            if np.unique(np.asarray(cached["batches"], dtype=str)).size > 1:
                eligible.append(dataset_id)
    return eligible


def load_source(
    dataset_id: str,
    *,
    target_path: Path,
    cache_labels: np.ndarray,
    cache_batches: np.ndarray,
    max_cells: int,
    input_sample_seed: int,
):
    """Load the input-capped AnnData and check that it is row-aligned with the cache."""
    source = load_dataset(dataset_id, str(target_path))
    source = _sample_cells_for_analysis(
        source, max_cells=max_cells, seed=input_sample_seed, dataset_id=dataset_id
    )
    batch_key = _batch_key_for_adata(source)
    labels = np.asarray(
        source.obs[LABEL_KEY].astype("string").fillna("unknown").astype(str).to_numpy(), dtype=str
    )
    batches = np.asarray(
        source.obs[batch_key].astype("string").fillna("unknown").astype(str).to_numpy(), dtype=str
    )
    if not np.array_equal(labels, cache_labels) or not np.array_equal(batches, cache_batches):
        raise RuntimeError(f"{dataset_id}: source/cache row alignment failed")
    return source, batches


def pcr_score(pcr_pre: float, embedding: np.ndarray, batches: np.ndarray) -> tuple[float, float]:
    """Post-integration PCR of the embedding and the scaled comparison score."""
    pcr_post = principal_component_regression(embedding, batches, categorical=True)
    if pcr_pre <= 0:
        return float("nan"), float(pcr_post)
    score = max(0.0, (pcr_pre - pcr_post) / pcr_pre)
    return float(score), float(pcr_post)


def shard_path(output_dir: Path, model: str, dataset_id: str) -> Path:
    return output_dir / "shards" / model / f"{dataset_id}.csv"


def shard_complete(path: Path, dataset_id: str, model: str, sample_seeds: Sequence[int]) -> bool:
    """A shard is complete when it holds one row per sample seed for this model."""
    if not path.exists():
        return False
    try:
        rows = read_csv_rows(path)
    except (OSError, ValueError):
        return False
    found = {(row["model"], int(row["sample_seed"])) for row in rows}
    expected = {(model, seed) for seed in sample_seeds}
    return found == expected and all(row["dataset"] == dataset_id for row in rows)


def score_dataset(
    dataset_id: str,
    model: str,
    *,
    cache_dir: Path,
    sample_dir: Path,
    target_path: Path,
    sample_seeds: Sequence[int],
    max_cells: int,
    input_sample_seed: int,
) -> list[dict[str, object]]:
    """PCR comparison of one dataset on every metric sample."""
    embeddings, cache_labels, cache_batches = load_cache(cache_path(cache_dir, dataset_id))
    source, batches = load_source(
        dataset_id,
        target_path=target_path,
        cache_labels=cache_labels,
        cache_batches=cache_batches,
        max_cells=max_cells,
        input_sample_seed=input_sample_seed,
    )
    rows: list[dict[str, object]] = []
    for seed in sample_seeds:
        started = time.monotonic()
        indices = load_metric_sample(sample_dir, dataset_id, seed, len(embeddings))
        sample_batches = batches[indices]
        n_batches = int(np.unique(sample_batches).size)
        if len(indices) < 3 or n_batches < 2:
            raise ValueError(
                f"{dataset_id}/seed{seed}: insufficient cells or batches "
                f"({len(indices)} cells, {n_batches} batches)"
            )
        sampled = source[indices].copy()
        sc.pp.normalize_total(sampled, target_sum=10_000)
        sc.pp.log1p(sampled)
        sc.tl.pca(sampled, svd_solver="arpack", use_highly_variable=False)
        pcr_pre = principal_component_regression(
            np.asarray(sampled.obsm["X_pca"], dtype=np.float32), sample_batches, categorical=True
        )
        score, pcr_post = pcr_score(pcr_pre, embeddings[indices], sample_batches)
        rows.append({
            "dataset": dataset_id,
            "model": model,
            "sample_seed": seed,
            "value": score,
            "pcr_pre": float(pcr_pre),
            "pcr_post": pcr_post,
            "n_sample_requested": len(indices),
            "n_sample_common": len(indices),
            "n_batches": n_batches,
            "seconds": time.monotonic() - started,
            "status": "complete",
        })
        del sampled
        gc.collect()
        print(
            f"  {dataset_id} seed={seed:02d}: n={len(indices)}, pre={pcr_pre:.6f}, "
            f"post={pcr_post:.6f}, value={score:.6f}, elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    del source, embeddings
    gc.collect()
    return rows


def aggregate(output_dir: Path, *, model: str, eligible: Sequence[str], sample_seeds: Sequence[int]) -> None:
    """Assemble every complete shard (all models found) into the long CSV and summaries."""
    shard_root = output_dir / "shards"
    all_rows: list[dict[str, str]] = []
    complete_by_model: dict[str, int] = {}
    if shard_root.exists():
        for model_dir in sorted(path for path in shard_root.iterdir() if path.is_dir()):
            for path in sorted(model_dir.glob("*.csv")):
                if shard_complete(path, path.stem, model_dir.name, sample_seeds):
                    all_rows.extend(read_csv_rows(path))
                    complete_by_model[model_dir.name] = complete_by_model.get(model_dir.name, 0) + 1
    atomic_csv(output_dir / "pcr_repeat20_long.csv", LONG_FIELDS, all_rows)
    complete = complete_by_model.get(model, 0)
    atomic_json(output_dir / "progress.json", {
        "status": "complete" if complete == len(eligible) else "partial",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "eligible_datasets": len(eligible),
        "complete_datasets": complete,
        "remaining_datasets": len(eligible) - complete,
        "complete_datasets_per_model": complete_by_model,
    })
    if not all_rows:
        return
    frame = pd.DataFrame(all_rows)
    for column in ("value", "pcr_pre", "pcr_post", "n_sample_requested", "n_sample_common"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    dataset_repeat = frame.groupby(["dataset", "model"], as_index=False).agg(
        value_mean=("value", "mean"),
        value_repeat_sd=("value", "std"),
        value_repeat_min=("value", "min"),
        value_repeat_max=("value", "max"),
        n_repeats=("value", "count"),
        mean_common_cells=("n_sample_common", "mean"),
    )
    atomic_csv(
        output_dir / "pcr_dataset_repeat_summary.csv",
        list(dataset_repeat.columns),
        dataset_repeat.to_dict("records"),
    )
    summary_rows: list[dict[str, object]] = []
    for model_name, group in dataset_repeat.groupby("model", sort=False):
        summary_rows.append({
            "model": model_name,
            "value_mean": float(group["value_mean"].mean()),
            "value_sd_across_datasets": float(group["value_mean"].std(ddof=1)),
            "n_datasets": int(group["dataset"].nunique()),
            "mean_dataset_repeat_sd": float(group["value_repeat_sd"].mean()),
            "mean_dataset_repeat_range": float((group["value_repeat_max"] - group["value_repeat_min"]).mean()),
            "mean_common_cells": float(group["mean_common_cells"].mean()),
        })
    atomic_csv(output_dir / "pcr_repeat20_summary.csv", list(summary_rows[0]), summary_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Directory with <dataset>.npz embedding caches")
    parser.add_argument("--metric-samples-dir", type=Path, default=DEFAULT_METRIC_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORING_DIR / "pcr_repeat")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Value of the `model` column")
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--target-path", type=Path, default=Path(DEFAULT_TARGET_PATH),
                        help="Held-out Census release, <root>/20251108/by_dataset")
    parser.add_argument("--max-cells", type=int, default=INPUT_MAX_CELLS,
                        help="Input cap the cache was exported with")
    parser.add_argument("--input-sample-seed", type=int, default=INPUT_SAMPLE_SEED)
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=list(SAMPLE_SEEDS))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = args.model_name
    output_dir = args.output_dir.resolve()
    all_ids = eligible_ids(args.cache_dir, resolve_dataset_ids(args.dataset_ids_file, None))
    if args.aggregate_only:
        aggregate(output_dir, model=model, eligible=all_ids, sample_seeds=args.sample_seeds)
        return 0
    requested = resolve_dataset_ids(args.dataset_ids_file, args.dataset_ids)
    ineligible = [dataset_id for dataset_id in requested if dataset_id not in all_ids]
    if args.dataset_ids and ineligible:
        raise ValueError(f"Single-batch datasets have no PCR comparison: {ineligible}")
    requested = [dataset_id for dataset_id in requested if dataset_id in all_ids]
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("--worker-index must be in [0, --worker-count)")
    datasets = requested[args.worker_index :: args.worker_count]

    atomic_json(output_dir / f"metadata__{model}.json", {
        "analysis": "20-repeat scIB PCR comparison on the precomputed metric samples",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "cache_dir": str(args.cache_dir.resolve()),
        "metric_samples_dir": str(args.metric_samples_dir.resolve()),
        "target_path": str(args.target_path),
        "eligible_dataset_count": len(all_ids),
        "metric_sample_seeds": list(args.sample_seeds),
        "input_max_cells": args.max_cells,
        "input_sample_seed": args.input_sample_seed,
        "normalization": "scanpy normalize_total(target_sum=1e4), log1p, PCA(svd_solver=arpack, all genes)",
        "pcr_formula": "max(0, (PCR_pre - PCR_post) / PCR_pre), categorical batch covariate",
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
    })
    for position, dataset_id in enumerate(datasets, start=1):
        path = shard_path(output_dir, model, dataset_id)
        if shard_complete(path, dataset_id, model, args.sample_seeds):
            print(f"[{position}/{len(datasets)}] {dataset_id}: reused", flush=True)
            continue
        started = time.monotonic()
        try:
            rows = score_dataset(
                dataset_id,
                model,
                cache_dir=args.cache_dir,
                sample_dir=args.metric_samples_dir,
                target_path=args.target_path,
                sample_seeds=args.sample_seeds,
                max_cells=args.max_cells,
                input_sample_seed=args.input_sample_seed,
            )
            atomic_csv(path, LONG_FIELDS, rows)
            print(
                f"[{position}/{len(datasets)}] {dataset_id}: wrote {len(rows)} values "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )
        except Exception as error:  # noqa: BLE001
            atomic_json(output_dir / "failures" / model / f"{dataset_id}.json", {
                "dataset": dataset_id,
                "model": model,
                "worker_index": args.worker_index,
                "error": f"{type(error).__name__}: {error}",
                "updated_at_utc": datetime.now(UTC).isoformat(),
            })
            print(f"[{position}/{len(datasets)}] {dataset_id}: FAILED {type(error).__name__}: {error}", flush=True)
    if not args.no_aggregate:
        aggregate(output_dir, model=model, eligible=all_ids, sample_seeds=args.sample_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
