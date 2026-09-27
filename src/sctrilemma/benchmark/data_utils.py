"""Data utilities for zero-shot benchmarking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def _default_data_root() -> Path:
    explicit = os.environ.get("SCTRILEMMA_DATA_ROOT") or os.environ.get("CELLXGENE_DATA_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    user = os.environ.get("USER")
    if user:
        return Path("/scratch") / user / "datasets" / "cellxgene"
    return Path("~/datasets/cellxgene").expanduser()


def _resolve_default_dataset_path(
    env_key: str,
    legacy_env_key: str,
    version: str,
    *candidates: str | Path,
) -> str:
    explicit = os.environ.get(env_key) or os.environ.get(legacy_env_key)
    if explicit:
        return str(Path(explicit).expanduser())

    expanded = [
        str(path.expanduser() if isinstance(path, Path) else Path(path).expanduser())
        for path in ((_default_data_root() / version / "by_dataset"), *candidates)
    ]
    for path in expanded:
        if os.path.exists(path):
            return path
    return expanded[0]


DEFAULT_BASELINE_PATH = _resolve_default_dataset_path(
    "SCTRILEMMA_BASELINE_PATH",
    "SCBENCH_BASELINE_PATH",
    "20250130",
    "~/research/datasets/cellxgene/20250130/by_dataset",
    "~/research/datasets/cellxgene_v2/20250130/by_dataset",
)
DEFAULT_TARGET_PATH = _resolve_default_dataset_path(
    "SCTRILEMMA_TARGET_PATH",
    "SCBENCH_TARGET_PATH",
    "20251108",
    "~/research/datasets/cellxgene/20251108/by_dataset",
    "~/research/datasets/cellxgene_v2/20251108/by_dataset",
)


def discover_new_datasets(
    baseline_path: str = DEFAULT_BASELINE_PATH,
    target_path: str = DEFAULT_TARGET_PATH,
) -> List[str]:
    """Discover dataset IDs in target but not in baseline for true zero-shot eval."""
    baseline_datasets = {d for d in os.listdir(baseline_path) if not d.startswith(".")}
    target_datasets = {d for d in os.listdir(target_path) if not d.startswith(".")}
    return sorted(target_datasets - baseline_datasets)


def list_baseline_datasets(
    baseline_path: str = DEFAULT_BASELINE_PATH,
) -> List[str]:
    """List all dataset IDs in the baseline (2025-01-30) Census."""
    return sorted(d for d in os.listdir(baseline_path) if not d.startswith("."))


def create_dataset_manifest(
    dataset_ids: List[str],
    target_path: str = DEFAULT_TARGET_PATH,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Create manifest with cell counts and paths for each dataset."""
    manifest_rows = []

    for dataset_id in dataset_ids:
        dataset_dir = Path(target_path) / dataset_id
        h5ad_files = sorted(dataset_dir.glob("*.h5ad"))

        n_chunks = len(h5ad_files)
        total_cells = 0
        n_genes = 0

        for h5ad_file in h5ad_files:
            try:
                adata = ad.read_h5ad(h5ad_file, backed="r")
                total_cells += adata.n_obs
                if n_genes == 0:
                    n_genes = adata.n_vars
                adata.file.close()
            except Exception as e:
                print(f"Warning: Could not read {h5ad_file}: {e}")
                continue

        manifest_rows.append(
            {
                "dataset_id": dataset_id,
                "n_cells": total_cells,
                "n_genes": n_genes,
                "n_chunks": n_chunks,
                "dataset_total_size_bytes": sum(f.stat().st_size for f in h5ad_files),
                "path": str(dataset_dir),
            }
        )

    manifest = pd.DataFrame(manifest_rows)

    if output_path:
        manifest.to_csv(output_path, index=False)
        print(f"Saved manifest to {output_path}")

    return manifest


def get_dataset_file_metadata(
    dataset_id: str,
    target_path: str = DEFAULT_TARGET_PATH,
) -> Dict[str, int]:
    """Return shard count and total on-disk size for a dataset."""
    dataset_dir = Path(target_path) / dataset_id
    h5ad_files = sorted(dataset_dir.glob("*.h5ad"))
    return {
        "n_shards": len(h5ad_files),
        "dataset_total_size_bytes": sum(f.stat().st_size for f in h5ad_files),
    }


def load_dataset(
    dataset_id: str,
    target_path: str = DEFAULT_TARGET_PATH,
    batch_key: str = "donor_id",
    label_key: str = "cell_type_ontology_term_id",
) -> ad.AnnData:
    """Load and concatenate h5ad chunks, validate batch/label columns."""
    dataset_dir = Path(target_path) / dataset_id
    h5ad_files = sorted(dataset_dir.glob("*.h5ad"))

    if not h5ad_files:
        raise ValueError(f"No h5ad files found in {dataset_dir}")

    adatas = [sc.read_h5ad(f) for f in h5ad_files]
    # `merge="same"` preserves var-level columns such as `feature_name`,
    # `feature_id`, and `feature_type` when they are identical across parts.
    adata = (
        adatas[0]
        if len(adatas) == 1
        else ad.concat(adatas, join="outer", index_unique="-", merge="same")
    )

    if batch_key not in adata.obs.columns:
        if "donor_id" in adata.obs.columns:
            batch_key = "donor_id"
        elif "dataset_id" in adata.obs.columns:
            batch_key = "dataset_id"
        else:
            adata.obs["batch"] = dataset_id
            batch_key = "batch"

    if label_key not in adata.obs.columns:
        raise ValueError(
            f"Label key '{label_key}' not found. Available: {list(adata.obs.columns)}"
        )

    adata.uns["batch_key"] = batch_key
    adata.uns["label_key"] = label_key
    adata.uns["dataset_id"] = dataset_id

    if "feature_id" in adata.var.columns:
        adata.var.index = pd.Index(adata.var["feature_id"].astype(str).values)
        adata.var_names_make_unique()

    adata.obs_names_make_unique()

    return adata


def get_gene_mapping(
    source_genes: List[str],
    target_genes: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get index arrays mapping common genes between source and target vocabularies."""
    source_to_idx = {g: i for i, g in enumerate(source_genes)}
    target_to_idx = {g: i for i, g in enumerate(target_genes)}

    common_genes = set(source_genes) & set(target_genes)

    source_indices = [source_to_idx[g] for g in common_genes]
    target_indices = [target_to_idx[g] for g in common_genes]

    return np.array(source_indices), np.array(target_indices)


def align_genes_to_model(
    adata: ad.AnnData,
    model_genes: List[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Reorder/pad expression matrix to match model's gene vocabulary."""
    source_genes = list(adata.var_names)
    source_idx, target_idx = get_gene_mapping(source_genes, model_genes)

    n_cells = adata.n_obs
    n_model_genes = len(model_genes)

    aligned = np.full((n_cells, n_model_genes), fill_value, dtype=np.float32)

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()  # type: ignore[union-attr]
    X = np.asarray(X, dtype=np.float32)

    aligned[:, target_idx] = X[:, source_idx]

    return aligned


def normalize_counts(
    X: np.ndarray,
    target_sum: float = 10000.0,
    log1p: bool = True,
) -> np.ndarray:
    """Library size normalize to target_sum, optionally apply log1p."""
    totals = X.sum(axis=1, keepdims=True)
    X_norm = X / np.maximum(totals, 1e-8) * target_sum

    if log1p:
        X_norm = np.log1p(X_norm)

    return X_norm.astype(np.float32)


if __name__ == "__main__":
    print("Discovering new datasets...")
    new_datasets = discover_new_datasets()
    print(f"Found {len(new_datasets)} new datasets")

    if new_datasets:
        print(f"\nFirst 5: {new_datasets[:5]}")
        print("\nCreating manifest for first 5 datasets...")
        manifest = create_dataset_manifest(
            new_datasets[:5],
            output_path="outputs/benchmark/manifest_sample.csv",
        )
        print(manifest)
