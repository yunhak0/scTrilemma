"""Score reconstruction caches against the raw sampled counts (Table 2 / Table 19 protocol).

Per dataset, cells are matched by the exact ordered ``soma_joinid`` of the staged sample and
genes are the shared per-dataset gene list (``experiments/data/genelists``) minus the few
genes listed in ``data/table2_excluded_genes.tsv`` (shared genes that one of the methods
compared in the paper could not decode for that dataset, so they were outside the paper's
common gene universe), restricted to the genes present in both the raw matrix and the cache,
then filtered to genes detected (raw count > 0) in at least ``--min-detection-rate`` of the
sampled cells (0.02 = "det02" in the paper). Raw counts and the reconstruction are independently normalised to log1p(CP10K) on
that gene set and scored with ``experiments.common.score_matrix`` (per-cell Pearson and
Spearman averaged over cells, MAE, MSE). Datasets are aggregated with equal weight.

Any cache with arrays ``recon`` (N, G) float32, ``gene_names`` (G) and ``soma_joinid`` (N)
can be scored (``--cache-dir`` / ``--model-name``); ``generate.py`` writes such caches.

    pixi run python -m experiments.reconstruction.score_agreement
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.common import (
    cache_payload,
    dense_columns,
    feature_lookup,
    load_sampled_adata,
    log1p_cp10k,
    normalize_gene,
    read_ids,
    score_matrix,
)
from experiments.reconstruction._util import (
    DEFAULT_GENELIST_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SAMPLES_DIR,
    MODEL_NAME,
    describe_path,
    display_name,
    read_genelist,
)

METRIC_COLUMNS = ["recon_pearson", "recon_spearman", "recon_mae", "recon_mse"]
DEFAULT_EXCLUDED_GENES = Path(__file__).resolve().parent / "data" / "table2_excluded_genes.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+", help="Subset of dataset IDs to score")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Sampled datasets written by experiments/prepare_samples.py",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT / "cache" / MODEL_NAME,
        help="Directory with one <dataset_id>.npz reconstruction cache per dataset",
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="Value of the 'model' column")
    parser.add_argument("--model-display", default=None, help="Table label (default: derived)")
    parser.add_argument(
        "--genelist-dir",
        type=Path,
        default=DEFAULT_GENELIST_DIR,
        help="Shared per-dataset gene lists that define the scored gene universe",
    )
    parser.add_argument(
        "--no-genelist",
        action="store_true",
        help="Score every gene shared by the raw matrix and the cache instead",
    )
    parser.add_argument(
        "--excluded-genes",
        type=Path,
        default=DEFAULT_EXCLUDED_GENES,
        help="TSV (dataset_id, gene) of shared genes outside the paper's common gene universe",
    )
    parser.add_argument(
        "--ignore-excluded-genes",
        action="store_true",
        help="Score the full shared gene list (a few more genes per dataset than the paper)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_ROOT / "results_det02")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-common-genes", type=int, default=100)
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.02,
        help="Keep genes detected (raw count > 0) in at least this fraction of the sampled "
        "cells (0 disables the filter)",
    )
    return parser.parse_args()


def load_excluded_genes(path: Path | None) -> dict[str, set[str]]:
    """Normalised excluded genes per dataset from a (dataset_id, gene) TSV."""
    if path is None:
        return {}
    excluded: dict[str, set[str]] = {}
    table = pd.read_csv(path, sep="\t", dtype=str)
    for dataset_id, gene in zip(table["dataset_id"], table["gene"]):
        excluded.setdefault(str(dataset_id), set()).add(normalize_gene(gene))
    return excluded


def gene_index_map(genes: list[str]) -> dict[str, int]:
    """First column of each normalised gene identifier."""
    mapping: dict[str, int] = {}
    for index, gene in enumerate(genes):
        mapping.setdefault(normalize_gene(gene), index)
    return mapping


def align_dataset(
    dataset_id: str,
    *,
    samples_dir: Path,
    cache_dir: Path,
    genelist_dir: Path | None,
    excluded_genes: set[str],
    min_detection_rate: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return log1p(CP10K) raw and reconstructed matrices on the matched cells and genes."""
    adata = load_sampled_adata(samples_dir / f"{dataset_id}.h5ad", dataset_id)
    target_soma = adata.obs["soma_joinid"].to_numpy().astype(np.int64)
    raw_lookup = feature_lookup(adata)
    recon, genes, soma = cache_payload(cache_dir / f"{dataset_id}.npz")
    if not np.array_equal(soma, target_soma):
        raise RuntimeError(f"{cache_dir.name}/{dataset_id} does not match the staged cell order")
    gene_map = gene_index_map(genes)
    common = set(raw_lookup) & set(gene_map)
    if genelist_dir is not None:
        common &= {normalize_gene(gene) for gene in read_genelist(genelist_dir / f"{dataset_id}.txt")}
    common -= excluded_genes
    ordered_genes = [
        gene for gene, _ in sorted(raw_lookup.items(), key=lambda item: item[1]) if gene in common
    ]
    raw_counts = np.asarray(dense_columns(adata.X, [raw_lookup[gene] for gene in ordered_genes]))
    if min_detection_rate > 0:
        detected = (raw_counts > 0).mean(axis=0) >= min_detection_rate
        ordered_genes = [gene for gene, keep in zip(ordered_genes, detected) if keep]
        raw_counts = raw_counts[:, detected]
    raw = log1p_cp10k(raw_counts)
    reconstructed = log1p_cp10k(recon[:, [gene_map[gene] for gene in ordered_genes]])
    return raw, reconstructed, int(target_soma.size), len(ordered_genes)


def write_results_markdown(aggregate: pd.DataFrame, output_path: Path, detection: float) -> None:
    """Human-readable summary table next to the CSVs."""
    columns = [
        ("recon_pearson", "Pearson (higher is better)"),
        ("recon_spearman", "Spearman (higher is better)"),
        ("recon_mae", "MAE (lower is better)"),
        ("recon_mse", "MSE (lower is better)"),
    ]
    lines = [
        "# Direct reconstruction agreement",
        "",
        "Deterministic sample of at most 2,500 cells per held-out dataset; genes are the shared "
        "per-dataset gene list detected in at least "
        f"{detection:.0%} of the sampled cells. Raw expression and the reconstruction are "
        "independently normalised to log1p(CP10K) on that gene set. Values are equal-dataset "
        f"mean +- sample SD across {int(aggregate['n_datasets'].max())} datasets.",
        "",
        "| Method | " + " | ".join(label for _, label in columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for _, row in aggregate.iterrows():
        values = [f"{row[f'{metric}_mean']:.4f} +- {row[f'{metric}_sd']:.4f}" for metric, _ in columns]
        lines.append(f"| {row['model_display']} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Direct agreement with observed counts is a secondary diagnostic, not a denoising "
            "ground truth: observed profiles contain sampling noise and technical variation "
            "that a reconstruction need not reproduce exactly.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    model = args.model_name
    label = display_name(model, args.model_display)
    genelist_dir = None if args.no_genelist else args.genelist_dir
    excluded_path = None if args.ignore_excluded_genes else args.excluded_genes
    excluded_genes = load_excluded_genes(excluded_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_ids": dataset_ids,
                "samples_dir": describe_path(args.samples_dir),
                "cache_dir": describe_path(args.cache_dir),
                "genelist_dir": describe_path(genelist_dir),
                "excluded_genes": describe_path(excluded_path),
                "model": model,
                "model_display": label,
                "min_detection_rate": args.min_detection_rate,
                "min_common_genes": args.min_common_genes,
                "normalization": "independent log1p(CP10K) on the per-dataset shared genes",
                "cell_alignment": "exact ordered soma_joinid",
                "dataset_aggregation": "equal dataset weight; sample SD across datasets",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for index, dataset_id in enumerate(dataset_ids, start=1):
        started = time.perf_counter()
        try:
            raw, reconstructed, n_cells, n_genes = align_dataset(
                dataset_id,
                samples_dir=args.samples_dir,
                cache_dir=args.cache_dir,
                genelist_dir=genelist_dir,
                excluded_genes=excluded_genes.get(dataset_id, set()),
                min_detection_rate=args.min_detection_rate,
            )
            if n_genes < args.min_common_genes:
                raise RuntimeError(
                    f"Only {n_genes} common genes, below threshold {args.min_common_genes}"
                )
            support_rows.append(
                {"dataset_id": dataset_id, "n_cells": n_cells, "n_common_genes": n_genes}
            )
            metrics = score_matrix(raw, reconstructed, device=device)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "model": model,
                    "model_display": label,
                    "n_cells": n_cells,
                    "n_common_genes": n_genes,
                    **metrics,
                }
            )
            print(
                f"[{index}/{len(dataset_ids)}] {dataset_id}: cells={n_cells}, "
                f"common_genes={n_genes}, time={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"dataset_id": dataset_id, "reason": str(exc)})
            print(f"[{index}/{len(dataset_ids)}] {dataset_id}: FAILED {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    pd.DataFrame(support_rows).to_csv(args.output_dir / "evaluation_support.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / "failures.csv", index=False)
    results.to_csv(args.output_dir / "per_dataset.csv", index=False)
    if failures:
        print(f"Incomplete: {len(failures)} dataset failures", flush=True)
        return 2
    if len(results) != len(dataset_ids) or not np.isfinite(results[METRIC_COLUMNS].to_numpy()).all():
        raise RuntimeError(
            f"Result integrity failure: rows={len(results)}, expected={len(dataset_ids)}"
        )

    aggregate_row: dict[str, object] = {
        "model": model,
        "model_display": label,
        "n_datasets": results["dataset_id"].nunique(),
    }
    for metric in METRIC_COLUMNS:
        aggregate_row[f"{metric}_mean"] = float(results[metric].mean())
        aggregate_row[f"{metric}_sd"] = float(results[metric].std(ddof=1))
    aggregate = pd.DataFrame([aggregate_row])
    aggregate.to_csv(args.output_dir / "aggregate_equal_dataset.csv", index=False)
    write_results_markdown(aggregate, args.output_dir / "RESULTS.md", args.min_detection_rate)
    print("\n" + aggregate.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Done: datasets={len(dataset_ids)}, model={model}, rows={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
