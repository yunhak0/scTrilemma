"""Contrast-level DEG concordance of reconstruction caches with the raw counts.

For a within-cell-type contrast (e.g. disease versus normal) the raw and the reconstructed
log1p(CP10K) profiles give one logFC per gene; genes detected in the two groups (summed
detection rate above ``min_detection_rate``) are compared by Spearman correlation of the
logFC vectors, Jaccard overlap of the top-``k`` genes by |logFC| and sign concordance on the
raw top-``k`` genes. ``rq4_panel.py`` drives this module for the paper's seven-cohort panel;
the command line below scores one or more staged datasets with a fixed contrast.

Reconstruction caches: ``<cache_dir>/<dataset_id>.npz`` with ``recon`` (n_cells, n_genes)
float32, ``gene_names`` (n_genes,) and ``soma_joinid`` (n_cells,) for row alignment.

    pixi run python -m experiments.table2_joint.deg_concordance --positive-label "colorectal cancer"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from experiments.common import (
    dense_columns,
    feature_lookup,
    load_sampled_adata,
    normalize_gene,
    read_ids,
)
from experiments.table2_joint._common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RECON_DIR,
    DEFAULT_SAMPLES_DIR,
    MODEL_NAME,
    GeneUniverse,
    add_gene_universe_arguments,
    cache_path,
    describe_path,
    display_name,
    parse_model_dirs,
)

DEFAULT_DATASET_IDS = ["829a3cd1-a466-49f1-b2e9-d3f6b7f392e2"]
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "deg_concordance"


@dataclass(frozen=True)
class Contrast:
    contrast_id: str
    dataset_id: str
    cell_type: str
    positive_label: str
    negative_label: str
    positive_n: int
    negative_n: int


@dataclass
class ReconResult:
    model_name: str
    recon: np.ndarray
    gene_names: list[str]


def load_recon_cache(
    model: str,
    location: Path,
    dataset_id: str,
    adata: ad.AnnData,
) -> ReconResult:
    """Load one reconstruction cache and align its rows to ``adata`` by ``soma_joinid``."""
    path = cache_path(location, dataset_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        recon = np.asarray(payload["recon"], dtype=np.float32)
        genes = np.asarray(payload["gene_names"]).astype(str).tolist()
        cache_ids = np.asarray(payload["soma_joinid"]).astype(str)
    target_ids = adata.obs["soma_joinid"].astype(str).to_numpy()
    if len(np.unique(cache_ids)) != len(cache_ids):
        raise RuntimeError(f"{model}/{dataset_id}: duplicate cache IDs")
    positions = pd.Series(np.arange(len(cache_ids)), index=cache_ids).reindex(target_ids)
    if positions.isna().any():
        raise RuntimeError(
            f"{model}/{dataset_id}: missing {int(positions.isna().sum())} target cells"
        )
    recon = recon[positions.to_numpy(dtype=np.int64)]
    if recon.shape != (adata.n_obs, len(genes)) or not np.isfinite(recon).all():
        raise RuntimeError(f"{model}/{dataset_id}: invalid reconstruction matrix")
    return ReconResult(model_name=model, recon=recon, gene_names=genes)


def select_contrasts(
    adata: ad.AnnData,
    *,
    dataset_id: str,
    contrast_key: str,
    cell_type_key: str,
    positive_label: str | None,
    negative_label: str | None,
    min_cells_per_group: int,
    max_contrasts: int,
) -> list[Contrast]:
    """Within-cell-type two-group contrasts with enough cells, largest groups first."""
    if contrast_key not in adata.obs.columns:
        raise ValueError(f"Missing contrast key '{contrast_key}' in obs.")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"Missing cell-type key '{cell_type_key}' in obs.")

    obs = adata.obs[[cell_type_key, contrast_key]].astype(str).copy()
    obs = obs[(obs[cell_type_key] != "unknown") & (obs[contrast_key] != "unknown")]
    contrasts: list[Contrast] = []

    for cell_type, group in obs.groupby(cell_type_key, sort=False):
        counts = group[contrast_key].value_counts()
        if positive_label is not None and negative_label is not None:
            pos_n = int(counts.get(positive_label, 0))
            neg_n = int(counts.get(negative_label, 0))
            if pos_n < min_cells_per_group or neg_n < min_cells_per_group:
                continue
            pos, neg = positive_label, negative_label
        else:
            eligible = counts[counts >= min_cells_per_group]
            if len(eligible) < 2:
                continue
            pos, neg = str(eligible.index[0]), str(eligible.index[1])
            pos_n, neg_n = int(eligible.iloc[0]), int(eligible.iloc[1])

        safe_ct = "".join(ch if ch.isalnum() else "_" for ch in str(cell_type))[:48]
        contrasts.append(
            Contrast(
                contrast_id=f"{dataset_id}::{safe_ct}::{pos}_vs_{neg}",
                dataset_id=dataset_id,
                cell_type=str(cell_type),
                positive_label=pos,
                negative_label=neg,
                positive_n=pos_n,
                negative_n=neg_n,
            )
        )

    contrasts.sort(key=lambda c: min(c.positive_n, c.negative_n), reverse=True)
    return contrasts[:max_contrasts] if max_contrasts > 0 else contrasts


def full_library_size(matrix) -> np.ndarray:
    """Per-cell total counts over all genes of a matrix (at least 1)."""
    sums = matrix.sum(axis=1)
    if sp.issparse(sums):
        sums = sums.A1
    else:
        sums = np.asarray(sums).reshape(-1)
    return np.maximum(sums.astype(np.float32), 1.0)


def log_cp10k_from_library(values: np.ndarray, library: np.ndarray) -> np.ndarray:
    """log1p(CP10K) of selected genes with an externally given library size."""
    denom = np.maximum(library.astype(np.float32), 1.0)[:, None]
    return np.log1p(values.astype(np.float32) * (10_000.0 / denom))


def align_common_matrices(
    adata: ad.AnnData,
    results: list[ReconResult],
    *,
    universe: GeneUniverse | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Raw counts and independently normalised log1p(CP10K) matrices on the common genes.

    Common genes are those present in the raw matrix and in every reconstruction cache,
    restricted to the paper's gene universe when ``universe`` is given (shared gene list
    minus excluded genes, then the pooled detection filter), kept in raw-matrix column order.
    Raw expression is normalised with the full-cell library; each reconstruction with the
    sum over all genes of its cache.
    """
    raw_lookup = feature_lookup(adata)
    common = set(raw_lookup)
    recon_maps: dict[str, dict[str, int]] = {}
    for result in results:
        gene_map = {normalize_gene(g): i for i, g in enumerate(result.gene_names)}
        recon_maps[result.model_name] = gene_map
        common &= set(gene_map)
    if universe is not None:
        common = universe.restrict(str(adata.uns.get("dataset_id", "")), common)

    if not common:
        raise RuntimeError("No common genes between raw counts and reconstructions.")

    ordered_genes = [
        gene
        for gene, _ in sorted(raw_lookup.items(), key=lambda item: item[1])
        if gene in common
    ]
    raw_cols = [raw_lookup[g] for g in ordered_genes]
    raw_counts = dense_columns(adata.X, raw_cols)
    min_rate = universe.min_sample_detection_rate if universe is not None else 0.0
    if min_rate > 0:
        detected = (np.asarray(raw_counts) > 0).mean(axis=0) >= min_rate
        ordered_genes = [g for g, keep in zip(ordered_genes, detected) if keep]
        raw_counts = np.asarray(raw_counts)[:, detected]
        if not ordered_genes:
            raise RuntimeError("Detection-rate prefilter removed every common gene.")
    raw_log = log_cp10k_from_library(raw_counts, full_library_size(adata.X))

    recon_log_by_model: dict[str, np.ndarray] = {}
    for result in results:
        idx = [recon_maps[result.model_name][g] for g in ordered_genes]
        recon_subset = result.recon[:, idx].astype(np.float32, copy=False)
        recon_library = np.maximum(result.recon.sum(axis=1).astype(np.float32), 1.0)
        recon_log_by_model[result.model_name] = log_cp10k_from_library(
            recon_subset, recon_library
        )

    return ordered_genes, raw_counts, raw_log, recon_log_by_model


def compute_logfc(matrix: np.ndarray, pos_mask: np.ndarray, neg_mask: np.ndarray) -> np.ndarray:
    """Mean difference of the (log-normalised) profiles between two cell groups."""
    return matrix[pos_mask].mean(axis=0) - matrix[neg_mask].mean(axis=0)


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Positions of the ``k`` largest |values| (all positions when fewer than ``k``)."""
    if len(values) <= k:
        return np.arange(len(values))
    return np.argpartition(-np.abs(values), k - 1)[:k]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two index sets."""
    set_a = set(map(int, a))
    set_b = set(map(int, b))
    union = len(set_a | set_b)
    return float(len(set_a & set_b) / union) if union else float("nan")


def evaluate_contrast(
    *,
    adata: ad.AnnData,
    genes: list[str],
    raw_counts: np.ndarray,
    raw_log: np.ndarray,
    recon_log_by_model: dict[str, np.ndarray],
    contrast: Contrast,
    contrast_key: str,
    cell_type_key: str,
    min_detection_rate: float,
    top_k: int,
) -> list[dict[str, object]]:
    """Long-format DEG concordance rows (three metrics per model) for one contrast."""
    obs = adata.obs
    pos_mask = (obs[cell_type_key].astype(str).to_numpy() == contrast.cell_type) & (
        obs[contrast_key].astype(str).to_numpy() == contrast.positive_label
    )
    neg_mask = (obs[cell_type_key].astype(str).to_numpy() == contrast.cell_type) & (
        obs[contrast_key].astype(str).to_numpy() == contrast.negative_label
    )
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return []

    detected = (
        (raw_counts[pos_mask] > 0).mean(axis=0) + (raw_counts[neg_mask] > 0).mean(axis=0)
    ) > min_detection_rate
    raw_logfc = compute_logfc(raw_log, pos_mask, neg_mask)
    keep = detected & np.isfinite(raw_logfc)
    if keep.sum() < max(20, top_k):
        return []

    raw_eval = raw_logfc[keep]
    rows: list[dict[str, object]] = []

    for model_name, recon_log in recon_log_by_model.items():
        recon_logfc = compute_logfc(recon_log, pos_mask, neg_mask)[keep]
        valid = np.isfinite(raw_eval) & np.isfinite(recon_logfc)
        if valid.sum() < max(20, top_k):
            continue

        raw_valid = raw_eval[valid]
        recon_valid = recon_logfc[valid]
        rho = spearmanr(raw_valid, recon_valid).statistic
        recon_top = topk_indices(recon_valid, top_k)
        raw_top_valid = topk_indices(raw_valid, top_k)
        sign_concordance = np.mean(
            np.sign(raw_valid[raw_top_valid]) == np.sign(recon_valid[raw_top_valid])
        )
        base = {
            "model": model_name,
            "dataset_id": contrast.dataset_id,
            "contrast_id": contrast.contrast_id,
            "cell_type": contrast.cell_type,
            "positive_label": contrast.positive_label,
            "negative_label": contrast.negative_label,
            "positive_n": int(pos_mask.sum()),
            "negative_n": int(neg_mask.sum()),
            "n_genes": int(valid.sum()),
        }
        rows.append(
            {
                **base,
                "metric": "logfc_spearman",
                "value": float(rho) if np.isfinite(rho) else np.nan,
            }
        )
        rows.append(
            {**base, "metric": f"top{top_k}_abs_jaccard", "value": jaccard(raw_top_valid, recon_top)}
        )
        rows.append(
            {**base, "metric": f"top{top_k}_sign_concordance", "value": float(sign_concordance)}
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Mean per (model, metric) over contrasts."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["model", "metric"], as_index=False)
        .agg(mean_value=("value", "mean"), n_contrasts=("contrast_id", "nunique"))
        .sort_values(["metric", "mean_value"], ascending=[True, False])
    )


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI options for the staged samples and the reconstruction caches."""
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument(
        "--recon-cache",
        action="append",
        metavar="NAME=DIR",
        help=f"Reconstruction cache directory of one model; default {MODEL_NAME}=<release cache>",
    )
    parser.add_argument("--dataset-ids", nargs="+", default=DEFAULT_DATASET_IDS)
    parser.add_argument("--dataset-ids-file", type=Path, default=None)
    parser.add_argument("--contrast-key", type=str, default="disease")
    parser.add_argument("--cell-type-key", type=str, default="cell_type")
    parser.add_argument("--positive-label", type=str, default="colorectal cancer")
    parser.add_argument("--negative-label", type=str, default="normal")
    parser.add_argument("--min-cells-per-group", type=int, default=40)
    parser.add_argument("--max-contrasts", type=int, default=6)
    parser.add_argument("--min-detection-rate", type=float, default=0.02)
    add_gene_universe_arguments(parser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cache_arguments(parser)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Only report selected contrasts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_ids = read_ids(args.dataset_ids_file) if args.dataset_ids_file else list(args.dataset_ids)
    cache_dirs = parse_model_dirs(args.recon_cache, {MODEL_NAME: DEFAULT_RECON_DIR})
    universe = GeneUniverse(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "recon_caches": {k: describe_path(v) for k, v in cache_dirs.items()},
                "samples_dir": describe_path(args.samples_dir),
                "dataset_ids": dataset_ids,
                "contrast_key": args.contrast_key,
                "cell_type_key": args.cell_type_key,
                "positive_label": args.positive_label,
                "negative_label": args.negative_label,
                "min_cells_per_group": args.min_cells_per_group,
                "max_contrasts": args.max_contrasts,
                "min_detection_rate": args.min_detection_rate,
                "top_k": args.top_k,
                **universe.config(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rows: list[dict[str, object]] = []
    all_contrasts: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        print(f"\n=== Dataset {dataset_id} ===", flush=True)
        adata = load_sampled_adata(args.samples_dir / f"{dataset_id}.h5ad", dataset_id)
        contrasts = select_contrasts(
            adata,
            dataset_id=dataset_id,
            contrast_key=args.contrast_key,
            cell_type_key=args.cell_type_key,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            min_cells_per_group=args.min_cells_per_group,
            max_contrasts=args.max_contrasts,
        )
        print(f"Sampled {adata.n_obs:,} cells; selected {len(contrasts)} contrasts")
        for contrast in contrasts:
            print(
                f"  {contrast.cell_type}: {contrast.positive_label} n={contrast.positive_n}, "
                f"{contrast.negative_label} n={contrast.negative_n}"
            )
            all_contrasts.append(asdict(contrast))
        if args.dry_run or not contrasts:
            continue

        results = [
            load_recon_cache(display_name(model), location, dataset_id, adata)
            for model, location in cache_dirs.items()
        ]
        genes, raw_counts, raw_log, recon_log_by_model = align_common_matrices(
            adata, results, universe=universe
        )
        print(f"Common DEG gene universe: {len(genes):,} genes")
        for contrast in contrasts:
            all_rows.extend(
                evaluate_contrast(
                    adata=adata,
                    genes=genes,
                    raw_counts=raw_counts,
                    raw_log=raw_log,
                    recon_log_by_model=recon_log_by_model,
                    contrast=contrast,
                    contrast_key=args.contrast_key,
                    cell_type_key=args.cell_type_key,
                    min_detection_rate=args.min_detection_rate,
                    top_k=args.top_k,
                )
            )

    pd.DataFrame(all_contrasts).to_csv(args.output_dir / "selected_contrasts.csv", index=False)
    pd.DataFrame(all_rows).to_csv(args.output_dir / "results_long.csv", index=False)
    summary = summarize(all_rows)
    if not summary.empty:
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        print("\nSummary")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nNo metric rows produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
