# ruff: noqa: E402
"""K-means initialisation sensitivity of NMI/ARI on the seed-0 metric sample (Table 1 protocol).

For every dataset the seed-0 metric sample (``experiments.scoring.metric_samples``) is
scored with FAISS CPU K-means (``k`` = number of cell types, ``niter`` 20, ``nredo`` 1) for
K-means seeds 0..19, giving NMI and ARI per seed; cLISI (perplexity-weighted Simpson
index on the k = 50 Euclidean k-NN graph) and BRAS (cosine, datasets with more than one
batch only) are computed once on the same sample. Output is one long CSV per model with
the columns ``dataset, model, metric, kmeans_seed, sample_seed, nredo, value`` plus a summary.

    python -m experiments.scoring.kmeans_repeat --cache-dir <dir> --metric-samples-dir <dir>
"""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ.get("CPU_THREADS", "4"))

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.scoring._common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_METRIC_SAMPLES_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SCORING_DIR,
    KMEANS_SEEDS,
    atomic_csv,
    atomic_json,
    cache_path,
    encode,
    load_cache,
    load_metric_sample,
    resolve_dataset_ids,
)
from sctrilemma.utils.metrics import compute_bras, compute_clisi

K = 50
NITER = 20
NREDO = 1
SAMPLE_SEED = 0
LONG_FIELDS = ["dataset", "model", "metric", "kmeans_seed", "sample_seed", "nredo", "value"]
SUMMARY_FIELDS = [
    "model",
    "summary_kind",
    "metric",
    "kmeans_seed",
    "value_mean",
    "value_std",
    "n_datasets",
    "n_values",
    "mean_dataset_seed_std",
    "mean_dataset_seed_range",
]


def nmi_ari_faiss(
    embeddings: np.ndarray,
    labels: np.ndarray,
    kmeans_seed: int,
    *,
    niter: int = NITER,
    nredo: int = NREDO,
    threads: int = 4,
) -> tuple[float, float]:
    """NMI/ARI of FAISS K-means with one initialisation at ``kmeans_seed``."""
    import faiss

    faiss.omp_set_num_threads(threads)
    valid_mask = labels >= 0
    embeddings = np.ascontiguousarray(embeddings[valid_mask], dtype=np.float32)
    labels = labels[valid_mask]
    if len(labels) < 10:
        return 0.0, 0.0
    n_clusters = min(int(np.unique(labels).size), len(labels) - 1)
    if n_clusters < 2:
        return 0.0, 0.0
    kmeans = faiss.Kmeans(
        d=embeddings.shape[1],
        k=n_clusters,
        niter=niter,
        nredo=nredo,
        seed=kmeans_seed,
        verbose=False,
    )
    kmeans.train(embeddings)
    _, assignments = kmeans.index.search(embeddings, 1)
    predicted = assignments.ravel()
    return (
        float(normalized_mutual_info_score(labels, predicted)),
        float(adjusted_rand_score(labels, predicted)),
    )


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def summary_rows(
    model: str, long_rows: Sequence[dict[str, object]], kmeans_seeds: Sequence[int]
) -> list[dict[str, object]]:
    """Per-dataset seed means (no pseudo-replication), per-seed global means, fixed-sample metrics."""
    summary: list[dict[str, object]] = []
    for metric in ("nmi", "ari"):
        metric_rows = [row for row in long_rows if row["metric"] == metric]
        by_dataset: dict[str, list[float]] = {}
        for row in metric_rows:
            by_dataset.setdefault(str(row["dataset"]), []).append(float(row["value"]))
        per_dataset_means = [float(np.mean(values)) for values in by_dataset.values()]
        per_dataset_stds = [float(np.std(values, ddof=1)) for values in by_dataset.values()]
        per_dataset_ranges = [float(np.ptp(values)) for values in by_dataset.values()]
        mean_value, std_value = _mean_std(per_dataset_means)
        summary.append({
            "model": model,
            "summary_kind": "dataset_seed_mean",
            "metric": metric,
            "kmeans_seed": "",
            "value_mean": mean_value,
            "value_std": std_value,
            "n_datasets": len(per_dataset_means),
            "n_values": len(metric_rows),
            "mean_dataset_seed_std": float(np.mean(per_dataset_stds)) if per_dataset_stds else float("nan"),
            "mean_dataset_seed_range": float(np.mean(per_dataset_ranges)) if per_dataset_ranges else float("nan"),
        })
        for seed in kmeans_seeds:
            seed_values = [float(row["value"]) for row in metric_rows if row["kmeans_seed"] == seed]
            mean_value, std_value = _mean_std(seed_values)
            summary.append({
                "model": model,
                "summary_kind": "global_kmeans_seed",
                "metric": metric,
                "kmeans_seed": seed,
                "value_mean": mean_value,
                "value_std": std_value,
                "n_datasets": len(seed_values),
                "n_values": len(seed_values),
                "mean_dataset_seed_std": "",
                "mean_dataset_seed_range": "",
            })
    for metric in ("clisi", "bras"):
        values = [float(row["value"]) for row in long_rows if row["metric"] == metric]
        mean_value, std_value = _mean_std(values)
        summary.append({
            "model": model,
            "summary_kind": "deterministic_fixed_sample",
            "metric": metric,
            "kmeans_seed": "",
            "value_mean": mean_value,
            "value_std": std_value,
            "n_datasets": len(values),
            "n_values": len(values),
            "mean_dataset_seed_std": "",
            "mean_dataset_seed_range": "",
        })
    return summary


def score_dataset(
    dataset_id: str,
    model: str,
    *,
    cache_dir: Path,
    sample_dir: Path,
    sample_seed: int,
    kmeans_seeds: Sequence[int],
    k: int,
    niter: int,
    nredo: int,
    threads: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Score one dataset; returns the long rows and a sample-manifest row."""
    embeddings, labels, batches = load_cache(cache_path(cache_dir, dataset_id))
    label_codes = encode(labels)
    batch_codes = encode(batches)
    indices = load_metric_sample(sample_dir, dataset_id, sample_seed, len(embeddings))
    sample_emb = np.ascontiguousarray(embeddings[indices], dtype=np.float32)
    sample_labels = label_codes[indices]
    sample_batches = batch_codes[indices]

    rows: list[dict[str, object]] = []
    for kmeans_seed in kmeans_seeds:
        nmi, ari = nmi_ari_faiss(
            sample_emb, sample_labels, kmeans_seed, niter=niter, nredo=nredo, threads=threads
        )
        for metric, value in (("nmi", nmi), ("ari", ari)):
            rows.append({
                "dataset": dataset_id, "model": model, "metric": metric,
                "kmeans_seed": kmeans_seed, "sample_seed": sample_seed, "nredo": nredo, "value": value,
            })
    clisi = compute_clisi(torch.from_numpy(sample_emb), torch.from_numpy(sample_labels), k=k)
    rows.append({
        "dataset": dataset_id, "model": model, "metric": "clisi",
        "kmeans_seed": "", "sample_seed": sample_seed, "nredo": nredo, "value": float(clisi),
    })
    n_batches = int(np.unique(batches).size)
    if n_batches > 1:
        bras = compute_bras(
            torch.from_numpy(sample_emb),
            torch.from_numpy(sample_batches),
            torch.from_numpy(sample_labels),
        )
        rows.append({
            "dataset": dataset_id, "model": model, "metric": "bras",
            "kmeans_seed": "", "sample_seed": sample_seed, "nredo": nredo, "value": float(bras),
        })
    manifest = {
        "dataset": dataset_id,
        "n_cache_cells": int(len(embeddings)),
        "n_metric_sample_cells": int(len(indices)),
        "n_labels": int(np.unique(sample_labels).size),
        "n_batches": n_batches,
        "bras_included": n_batches > 1,
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Directory with <dataset>.npz embedding caches")
    parser.add_argument("--metric-samples-dir", type=Path, default=DEFAULT_METRIC_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORING_DIR / "kmeans_repeat")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Value of the `model` column")
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED,
                        help="Metric sample scored (the paper uses the seed-0 sample)")
    parser.add_argument("--kmeans-seeds", type=int, nargs="+", default=list(KMEANS_SEEDS))
    parser.add_argument("--k", type=int, default=K, help="k-NN size of cLISI")
    parser.add_argument("--niter", type=int, default=NITER)
    parser.add_argument("--nredo", type=int, default=NREDO)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("CPU_THREADS", "4")),
                        help="FAISS OpenMP threads")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_ids = resolve_dataset_ids(args.dataset_ids_file, args.dataset_ids)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model_name

    long_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for position, dataset_id in enumerate(dataset_ids, start=1):
        rows, manifest = score_dataset(
            dataset_id,
            model,
            cache_dir=args.cache_dir,
            sample_dir=args.metric_samples_dir,
            sample_seed=args.sample_seed,
            kmeans_seeds=args.kmeans_seeds,
            k=args.k,
            niter=args.niter,
            nredo=args.nredo,
            threads=args.threads,
        )
        long_rows.extend(rows)
        manifest_rows.append(manifest)
        print(
            f"[{position}/{len(dataset_ids)}] {dataset_id}: sample={manifest['n_metric_sample_cells']:,}, "
            f"batches={manifest['n_batches']}",
            flush=True,
        )

    atomic_csv(output_dir / f"kmeans_repeat_long__{model}.csv", LONG_FIELDS, long_rows)
    atomic_csv(
        output_dir / f"kmeans_repeat_summary__{model}.csv",
        SUMMARY_FIELDS,
        summary_rows(model, long_rows, args.kmeans_seeds),
    )
    atomic_csv(
        output_dir / f"kmeans_repeat_sample_manifest__{model}.csv",
        ["dataset", "n_cache_cells", "n_metric_sample_cells", "n_labels", "n_batches", "bras_included"],
        manifest_rows,
    )
    atomic_json(output_dir / f"kmeans_repeat_metadata__{model}.json", {
        "analysis": "FAISS K-means initialisation sensitivity on the fixed metric sample",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "cache_dir": str(args.cache_dir.resolve()),
        "metric_samples_dir": str(args.metric_samples_dir.resolve()),
        "sample_seed": args.sample_seed,
        "kmeans_seeds": list(args.kmeans_seeds),
        "kmeans_backend": "faiss.Kmeans (CPU)",
        "niter": args.niter,
        "nredo": args.nredo,
        "k": args.k,
        "n_datasets": len(dataset_ids),
        "n_bras_datasets": sum(bool(row["bras_included"]) for row in manifest_rows),
        "single_batch_bras_excluded": True,
    })
    print(
        f"Done: {len(long_rows)} long-form rows; BRAS included for "
        f"{sum(bool(row['bras_included']) for row in manifest_rows)} datasets -> {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
