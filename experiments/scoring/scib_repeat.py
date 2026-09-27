# ruff: noqa: E402
"""Table 1 embedding metrics with scib-metrics under the 20-repeat protocol.

NMI/ARI: ``scib_metrics.utils.KMeans`` (k = number of cell types) at seeds 0..19 on the
seed-0 metric sample, scored with the same scikit-learn calls as the library wrapper
(``--verify`` asserts that seed 0 reproduces ``nmi_ari_cluster_labels_kmeans`` exactly).
silhouette_label, isolated_labels, clisi_knn and, for datasets with more than one batch,
ilisi_knn and bras are computed on each of the 20 metric samples; the LISI scores use the
pynndescent k = 90 neighbour graph. Output: ``scib_repeat20_long__<model>.csv`` with the
columns ``dataset, model, metric, value, sample_seed, kmeans_seed, n_cells, n_labels,
n_batches`` (``kmeans_seed`` is -1 for the sample-level metrics).

    python -m experiments.scoring.scib_repeat --cache-dir <dir> --metric-samples-dir <dir>

The module is resumable (datasets already present in the output file are skipped) and can
be split over ``--num-shards`` processes, each writing a ``__shard<k>`` file.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ.get("CPU_THREADS", "4"))
# jax (scib-metrics) grabs most of the GPU memory otherwise; this does not change results.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.scoring._common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_METRIC_SAMPLES_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_SCORING_DIR,
    cache_path,
    load_cache,
    load_metric_sample,
    resolve_dataset_ids,
)

N_SAMPLE_SEEDS = 20
N_KMEANS_SEEDS = 20
N_NEIGHBORS = 90
FIELDS = ["dataset", "model", "metric", "value", "sample_seed", "kmeans_seed", "n_cells", "n_labels", "n_batches"]


def nmi_ari_seeded(X: np.ndarray, labels: np.ndarray, seed: int) -> tuple[float, float]:
    """scib-metrics KMeans at an explicit seed, scored with the wrapper's scikit-learn calls."""
    from scib_metrics.utils import KMeans

    predicted = KMeans(n_clusters=len(np.unique(labels)), seed=seed).fit(X).labels_
    return (
        float(normalized_mutual_info_score(labels, predicted, average_method="arithmetic")),
        float(adjusted_rand_score(labels, predicted)),
    )


def verify_wrapper_equivalence(X: np.ndarray, labels: np.ndarray) -> None:
    """Seed 0 through the seeded path must equal the library wrapper."""
    from scib_metrics.metrics import nmi_ari_cluster_labels_kmeans

    reference = nmi_ari_cluster_labels_kmeans(X, labels)
    nmi, ari = nmi_ari_seeded(X, labels, 0)
    if abs(reference["nmi"] - nmi) >= 1e-9 or abs(reference["ari"] - ari) >= 1e-9:
        raise AssertionError(f"wrapper {reference} != seeded path nmi={nmi}, ari={ari}")
    print(f"[verify] wrapper NMI={reference['nmi']:.6f} ARI={reference['ari']:.6f} == seeded path", flush=True)


def sample_metrics(
    X: np.ndarray, labels: np.ndarray, batches: np.ndarray, *, n_neighbors: int = N_NEIGHBORS
) -> dict[str, float]:
    """Library metrics that need no clustering seed."""
    from scib_metrics import bras, clisi_knn, ilisi_knn, isolated_labels, silhouette_label
    from scib_metrics.nearest_neighbors import pynndescent

    out: dict[str, float] = {}
    out["silhouette_label"] = float(silhouette_label(X, labels))
    out["isolated_labels"] = float(isolated_labels(X, labels, batches))
    neighbors = pynndescent(X, n_neighbors=n_neighbors)
    out["clisi_knn"] = float(clisi_knn(neighbors, labels))
    if len(np.unique(batches)) > 1:
        out["ilisi_knn"] = float(ilisi_knn(neighbors, batches))
        out["bras"] = float(bras(X, labels, batches))
    return out


def score_dataset(
    dataset_id: str,
    model: str,
    *,
    cache_dir: Path,
    sample_dir: Path,
    n_sample_seeds: int,
    n_kmeans_seeds: int,
    n_neighbors: int,
) -> list[dict[str, object]]:
    """All rows of one dataset."""
    X_all, labels_all, batches_all = load_cache(cache_path(cache_dir, dataset_id))
    rows: list[dict[str, object]] = []

    idx0 = load_metric_sample(sample_dir, dataset_id, 0, len(X_all))
    X0, y0 = X_all[idx0], labels_all[idx0]
    n_batches0 = len(np.unique(batches_all[idx0]))
    for kseed in range(n_kmeans_seeds):
        nmi, ari = nmi_ari_seeded(X0, y0, kseed)
        for name, value in (("nmi", nmi), ("ari", ari)):
            rows.append({
                "dataset": dataset_id, "model": model, "metric": name, "value": value,
                "sample_seed": 0, "kmeans_seed": kseed, "n_cells": len(idx0),
                "n_labels": len(np.unique(y0)), "n_batches": n_batches0,
            })

    for sseed in range(n_sample_seeds):
        idx = load_metric_sample(sample_dir, dataset_id, sseed, len(X_all))
        scores = sample_metrics(X_all[idx], labels_all[idx], batches_all[idx], n_neighbors=n_neighbors)
        for name, value in scores.items():
            rows.append({
                "dataset": dataset_id, "model": model, "metric": name, "value": value,
                "sample_seed": sseed, "kmeans_seed": -1, "n_cells": len(idx),
                "n_labels": len(np.unique(labels_all[idx])),
                "n_batches": len(np.unique(batches_all[idx])),
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Directory with <dataset>.npz embedding caches")
    parser.add_argument("--metric-samples-dir", type=Path, default=DEFAULT_METRIC_SAMPLES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCORING_DIR / "scib_repeat")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Value of the `model` column")
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--n-sample-seeds", type=int, default=N_SAMPLE_SEEDS)
    parser.add_argument("--n-kmeans-seeds", type=int, default=N_KMEANS_SEEDS)
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS, help="pynndescent k for the LISI scores")
    parser.add_argument("--verify", action="store_true", help="run the seed-0 wrapper equivalence check and exit")
    parser.add_argument("--shard-index", type=int, default=0, help="dataset shard to score (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split datasets into this many shards; >1 writes __shard<k> files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ids = resolve_dataset_ids(args.dataset_ids_file, args.dataset_ids)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1:
        ids = ids[args.shard_index :: args.num_shards]
    model = args.model_name

    if args.verify:
        dataset_id = ids[0]
        X, labels, _ = load_cache(cache_path(args.cache_dir, dataset_id))
        idx = load_metric_sample(args.metric_samples_dir, dataset_id, 0, len(X))
        verify_wrapper_equivalence(X[idx], labels[idx])
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"__shard{args.shard_index}" if args.num_shards > 1 else ""
    out_path = output_dir / f"scib_repeat20_long__{model}{shard_suffix}.csv"
    done: set[str] = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as existing:
            done = {row["dataset"] for row in csv.DictReader(existing)}
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not done and handle.tell() == 0:
            writer.writeheader()
        for position, dataset_id in enumerate(ids, start=1):
            if dataset_id in done:
                print(f"[{model}] [{position}/{len(ids)}] {dataset_id}: reused", flush=True)
                continue
            rows = score_dataset(
                dataset_id,
                model,
                cache_dir=args.cache_dir,
                sample_dir=args.metric_samples_dir,
                n_sample_seeds=args.n_sample_seeds,
                n_kmeans_seeds=args.n_kmeans_seeds,
                n_neighbors=args.n_neighbors,
            )
            writer.writerows(rows)
            handle.flush()
            print(f"[{model}] [{position}/{len(ids)}] {dataset_id}: {len(rows)} rows", flush=True)
    print(f"Done -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
