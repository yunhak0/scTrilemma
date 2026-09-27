"""Shared helpers for the paper analyses: dataset sampling, gene alignment, reconstruction scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from sctrilemma.utils.metrics import cell_pearson_dense, cell_spearman_dense


def read_ids(path: Path) -> list[str]:
    """Read unique non-comment identifiers in file order."""
    values: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        values.append(value)
        seen.add(value)
    if not values:
        raise ValueError(f"No dataset IDs found in {path}")
    return values


def stable_seed(seed: int, value: str) -> int:
    """Derive a deterministic dataset-specific NumPy seed."""
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return (seed + int(digest[:8], 16)) % (2**32)


def stratified_sample_indices(
    obs: pd.DataFrame,
    *,
    max_cells: int,
    seed: int,
    keys: tuple[str, ...] = ("cell_type", "disease"),
) -> np.ndarray:
    """Sample at most ``max_cells`` while preserving small metadata strata."""
    if max_cells <= 0 or len(obs) <= max_cells:
        return np.arange(len(obs), dtype=np.int64)

    rng = np.random.default_rng(seed)
    present = [key for key in keys if key in obs.columns]
    if not present:
        return np.sort(rng.choice(len(obs), size=max_cells, replace=False))

    group_key = obs[present].astype(str).agg("||".join, axis=1)
    groups = group_key.groupby(group_key).indices
    base = max(1, max_cells // max(1, len(groups)))
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for values in groups.values():
        indices = np.asarray(values, dtype=np.int64)
        rng.shuffle(indices)
        take = min(base, len(indices))
        selected.append(indices[:take])
        if len(indices) > take:
            leftovers.append(indices[take:])

    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    if len(chosen) < max_cells and leftovers:
        remaining = np.concatenate(leftovers)
        rng.shuffle(remaining)
        chosen = np.concatenate([chosen, remaining[: max_cells - len(chosen)]])
    if len(chosen) > max_cells:
        chosen = rng.choice(chosen, size=max_cells, replace=False)
    return np.sort(chosen.astype(np.int64))


def normalize_obs_metadata(adata: ad.AnnData, columns: Iterable[str]) -> None:
    """Make selected metadata columns stable strings."""
    for column in columns:
        if column in adata.obs.columns:
            adata.obs[column] = (
                adata.obs[column].astype("string").fillna("unknown").astype(str)
            )


def load_sampled_adata(path: Path, dataset_id: str) -> ad.AnnData:
    """Load and validate one staged sampled dataset."""
    adata = ad.read_h5ad(path)
    if "soma_joinid" not in adata.obs.columns:
        raise RuntimeError(f"{path} lacks obs['soma_joinid']")
    soma = adata.obs["soma_joinid"].to_numpy().astype(np.int64)
    if np.unique(soma).size != soma.size:
        raise RuntimeError(f"{path} contains duplicate soma_joinid values")
    normalize_obs_metadata(
        adata,
        (
            "dataset_id",
            "donor_id",
            "disease",
            "tissue",
            "tissue_general",
            "assay",
            "cell_type",
            "cell_type_ontology_term_id",
        ),
    )
    adata.uns["dataset_id"] = dataset_id
    if "dataset_id" not in adata.obs.columns:
        adata.obs["dataset_id"] = dataset_id
    if "organism_ontology_term_id" not in adata.obs.columns:
        adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    return adata


def cache_payload(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Load and minimally validate a reconstruction cache."""
    with np.load(path, allow_pickle=False) as cached:
        recon = np.asarray(cached["recon"], dtype=np.float32)
        genes = [str(value) for value in cached["gene_names"]]
        soma = np.asarray(cached["soma_joinid"], dtype=np.int64)
    if recon.ndim != 2 or recon.shape != (soma.size, len(genes)):
        raise RuntimeError(
            f"Malformed cache {path}: recon={recon.shape}, cells={soma.size}, genes={len(genes)}"
        )
    if np.unique(soma).size != soma.size:
        raise RuntimeError(f"Duplicate soma_joinid values in {path}")
    if not np.isfinite(recon).all() or np.any(recon < 0):
        raise RuntimeError(f"Non-finite or negative reconstruction values in {path}")
    return recon, genes, soma


def normalize_gene(value: object) -> str:
    """Normalize gene identifiers without changing identifier semantics."""
    return str(value).strip().upper()


def feature_lookup(adata: ad.AnnData) -> dict[str, int]:
    """Map var names and standard feature columns to raw-matrix columns."""
    lookup: dict[str, int] = {}
    for index, value in enumerate(adata.var_names):
        lookup.setdefault(normalize_gene(value), index)
    for column in ("feature_id", "feature_name"):
        if column in adata.var.columns:
            for index, value in enumerate(adata.var[column].astype(str)):
                lookup.setdefault(normalize_gene(value), index)
    return lookup


def dense_columns(matrix, columns: list[int]) -> np.ndarray:
    """Materialize selected columns as float32."""
    selected = matrix[:, columns]
    if sp.issparse(selected):
        return np.asarray(selected.toarray(), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32)


def log1p_cp10k(values: np.ndarray) -> np.ndarray:
    """Independently normalize a nonnegative matrix on its current genes."""
    totals = np.asarray(values.sum(axis=1), dtype=np.float32).reshape(-1, 1)
    scale = 10_000.0 / np.maximum(totals, 1.0)
    return np.log1p(values.astype(np.float32, copy=False) * scale).astype(np.float32)


def metadata_json(**values: object) -> np.ndarray:
    """Encode metadata for storage in an allow_pickle=False NPZ."""
    return np.asarray(json.dumps(values, sort_keys=True))


def score_matrix(
    raw: np.ndarray,
    reconstructed: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Per-cell Pearson/Spearman (averaged over cells) plus MAE/MSE of a reconstruction."""
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw)).to(device)
    recon_tensor = torch.from_numpy(np.ascontiguousarray(reconstructed)).to(device)
    mask = torch.ones_like(raw_tensor, dtype=torch.bool)
    pearson = cell_pearson_dense(raw_tensor, recon_tensor, mask)
    spearman = cell_spearman_dense(raw_tensor, recon_tensor, mask)
    difference = recon_tensor - raw_tensor
    mae = float(difference.abs().mean().item())
    mse = float(difference.square().mean().item())
    del raw_tensor, recon_tensor, mask, difference
    return {
        "recon_pearson": pearson,
        "recon_spearman": spearman,
        "recon_mae": mae,
        "recon_mse": mse,
    }
