"""Training-style HVG Pearson and gene-mean correlations of the released checkpoint.

These are the expression-fidelity numbers of the ablation figure. Unlike Table 2 they are
computed on raw counts (no CP10K normalisation of the per-cell metric) and on the model's own
decode set, and they need the checkpoint: reconstructions are decoded here, not read from the
``generate.py`` caches. Per held-out dataset:

1. cells: every cell of the dataset (all shards under ``<data-root>/<dataset_id>/``), or a
   uniform random subset of ``--max-cells-per-dataset`` cells (20,000) drawn with the
   dataset-specific seed ``stable_seed(--sample-seed, dataset_id)``; rows keep the global
   order of the sorted shards;
2. genes: the default decode set of ``sctrilemma.inference.reconstruct_adata`` - the
   dataset-level top-4,096 vocabulary genes ranked by summed raw expression over the sampled
   cells - aligned to raw columns by var name (Ensembl ID);
3. ``training_style_recon_pearson_hvg``: raw counts vs ZINB means restricted to the
   ``--top-k-hvg`` (2,000) decoded genes with the largest raw-count variance; per-cell Pearson
   averaged over cells with a nonzero denominator (``n_cells`` in the output);
4. ``gene_mean_pearson`` / ``gene_mean_spearman``: raw counts and ZINB means independently
   log1p(CP10K)-normalised on the decoded genes, per-gene mean over cells, then Pearson /
   Spearman across genes (``n_hvg`` holds the number of decoded genes for these rows).

    pixi run python -m experiments.reconstruction.hvg_fidelity
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

from experiments.common import normalize_obs_metadata, read_ids, stable_seed
from experiments.prepare_samples import materialize_selected, read_dataset_obs
from experiments.reconstruction._util import (
    DATA_ROOT,
    DEFAULT_IDS_FILE,
    DEFAULT_RESULTS_ROOT,
    MODEL_NAME,
    add_model_arguments,
    load_inference,
)

DEFAULT_DATA_ROOT = DATA_ROOT / "20251108" / "by_dataset"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "hvg_fidelity"
FIELDS = ["run", "dataset", "metric", "value", "n_hvg", "n_cells", "sampled_cells", "sample_seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=MODEL_NAME, help="Label in the 'run' column and file name")
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+", help="Subset of dataset IDs to process")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N dataset IDs")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Census by_dataset root with one folder of h5ad shards per dataset",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-cells-per-dataset", type=int, default=20_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--sample-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Repeat the cell sample with these seeds (overrides --sample-seed)",
    )
    parser.add_argument("--top-k-hvg", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=2048)
    add_model_arguments(parser)
    return parser.parse_args()


def load_dataset_cells(
    dataset_id: str, *, data_root: Path, max_cells: int | None, seed: int
) -> tuple[ad.AnnData, int]:
    """Load all cells of a dataset or a uniform random subset, in global shard order.

    Only the selected rows are materialised; the result equals slicing the in-memory
    concatenation of the sorted shards with the same indices.
    """
    paths = sorted((data_root / dataset_id).glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No h5ad files under {data_root / dataset_id}")
    obs = read_dataset_obs(paths)
    n_total = len(obs)
    if max_cells is None or max_cells <= 0 or n_total <= max_cells:
        indices = np.arange(n_total, dtype=np.int64)
    else:
        rng = np.random.default_rng(stable_seed(seed, dataset_id))
        indices = np.sort(rng.choice(n_total, size=max_cells, replace=False))
    adata = materialize_selected(paths, obs.iloc[indices].copy(), dataset_id)
    normalize_obs_metadata(adata, ("donor_id",))
    return adata, n_total


def align_raw(adata: ad.AnnData, recon_gene_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Raw counts of the decoded genes (by var name) and the mask of decoded genes found."""
    gene_to_index = {str(gene): index for index, gene in enumerate(adata.var_names)}
    keep_recon: list[int] = []
    raw_columns: list[int] = []
    for index, gene in enumerate(recon_gene_names):
        raw_index = gene_to_index.get(gene)
        if raw_index is None:
            continue
        keep_recon.append(index)
        raw_columns.append(raw_index)
    matrix = adata.X
    if sp.issparse(matrix):
        raw = np.asarray(matrix[:, raw_columns].toarray(), dtype=np.float32)
    else:
        raw = np.asarray(matrix[:, raw_columns], dtype=np.float32)
    keep_mask = np.zeros(len(recon_gene_names), dtype=bool)
    keep_mask[keep_recon] = True
    return raw, keep_mask


def rowwise_pearson_mean(
    raw: np.ndarray, recon: np.ndarray, *, chunk_size: int, eps: float = 1e-8
) -> tuple[float, int]:
    """Mean per-cell Pearson over cells whose centred norms are both above ``eps``."""
    total = 0.0
    count = 0
    for start in range(0, raw.shape[0], chunk_size):
        end = min(start + chunk_size, raw.shape[0])
        raw_chunk = raw[start:end].astype(np.float32, copy=False)
        recon_chunk = recon[start:end].astype(np.float32, copy=False)
        raw_centered = raw_chunk - raw_chunk.mean(axis=1, keepdims=True)
        recon_centered = recon_chunk - recon_chunk.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(raw_centered, axis=1) * np.linalg.norm(recon_centered, axis=1)
        valid = denom > eps
        if np.any(valid):
            corr = (raw_centered[valid] * recon_centered[valid]).sum(axis=1) / denom[valid]
            corr = corr[np.isfinite(corr)]
            total += float(corr.sum())
            count += int(corr.size)
    if count == 0:
        return float("nan"), 0
    return total / count, count


def log1p_cp10k_f64(matrix: np.ndarray) -> np.ndarray:
    """Float64 log1p(CP10K) over the given genes (zero totals are left unscaled)."""
    dense = np.asarray(matrix, dtype=np.float64)
    totals = dense.sum(axis=1, keepdims=True)
    totals[totals <= 0] = 1.0
    return np.log1p(dense / totals * 10_000.0)


def compute_gene_mean_correlations(raw: np.ndarray, recon: np.ndarray) -> tuple[float, float]:
    """Gene-mean Pearson/Spearman across genes in log1p(CP10K) space."""
    real_mean = log1p_cp10k_f64(raw).mean(axis=0)
    gen_mean = log1p_cp10k_f64(recon).mean(axis=0)
    valid = np.isfinite(real_mean) & np.isfinite(gen_mean)
    if valid.sum() < 3:
        return 0.0, 0.0
    return (
        float(pearsonr(real_mean[valid], gen_mean[valid])[0]),
        float(spearmanr(real_mean[valid], gen_mean[valid])[0]),
    )


def compute_training_style_hvg(
    raw: np.ndarray, recon: np.ndarray, *, top_k_hvg: int, chunk_size: int
) -> tuple[float, int, int]:
    """Per-cell Pearson on raw counts vs means over the top-k raw-variance genes."""
    if raw.shape != recon.shape:
        raise ValueError(f"raw/recon shape mismatch: {raw.shape} vs {recon.shape}")
    gene_var = raw.astype(np.float64, copy=False).var(axis=0)
    valid = np.isfinite(gene_var) & (gene_var > 0)
    k = min(int(top_k_hvg), int(valid.sum()))
    if k < 2:
        return float("nan"), k, 0
    valid_index = np.where(valid)[0]
    order = np.argsort(gene_var[valid_index])[-k:]
    hvg_index = np.sort(valid_index[order])
    value, n_cells = rowwise_pearson_mean(
        raw[:, hvg_index], recon[:, hvg_index], chunk_size=chunk_size
    )
    return value, k, n_cells


def main() -> int:
    args = parse_args()
    dataset_ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    if args.limit is not None:
        dataset_ids = dataset_ids[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_seeds = args.sample_seeds if args.sample_seeds else [args.sample_seed]
    suffix = f"_repeat{len(sample_seeds)}" if args.sample_seeds else ""
    output_path = args.output_dir / f"{args.run}{suffix}.csv"

    wrapper = load_inference(args)
    rows: list[dict[str, object]] = []
    for index, dataset_id in enumerate(dataset_ids, start=1):
        for sample_seed in sample_seeds:
            adata, n_total = load_dataset_cells(
                dataset_id,
                data_root=args.data_root,
                max_cells=args.max_cells_per_dataset,
                seed=sample_seed,
            )
            recon, recon_gene_names = wrapper.reconstruct(adata, batch_size=args.batch_size)
            raw, keep_mask = align_raw(adata, recon_gene_names)
            if not keep_mask.all():
                recon = recon[:, keep_mask]
            value, n_hvg, n_cells = compute_training_style_hvg(
                raw, recon, top_k_hvg=args.top_k_hvg, chunk_size=args.chunk_size
            )
            rows.append(
                {
                    "run": args.run,
                    "dataset": dataset_id,
                    "metric": "training_style_recon_pearson_hvg",
                    "value": value,
                    "n_hvg": n_hvg,
                    "n_cells": n_cells,
                    "sampled_cells": int(adata.n_obs),
                    "sample_seed": sample_seed,
                }
            )
            gene_mean_pearson, gene_mean_spearman = compute_gene_mean_correlations(raw, recon)
            for metric_name, metric_value in (
                ("gene_mean_pearson", gene_mean_pearson),
                ("gene_mean_spearman", gene_mean_spearman),
            ):
                rows.append(
                    {
                        "run": args.run,
                        "dataset": dataset_id,
                        "metric": metric_name,
                        "value": metric_value,
                        "n_hvg": int(raw.shape[1]),
                        "n_cells": int(raw.shape[0]),
                        "sampled_cells": int(adata.n_obs),
                        "sample_seed": sample_seed,
                    }
                )
            print(
                f"[{index}/{len(dataset_ids)}] {dataset_id}: seed={sample_seed} "
                f"cells={adata.n_obs}/{n_total} decoded_genes={raw.shape[1]} "
                f"hvg_pearson={value:.4f} gene_mean_pearson={gene_mean_pearson:.4f}",
                flush=True,
            )
            del adata, recon, raw
            wrapper.clear_runtime_cache()

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: wrote {output_path} ({len(rows)} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
