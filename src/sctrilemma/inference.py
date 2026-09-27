"""Shared inference path for scTrilemma models (ZSB / stand-alone benchmarks).

Single source of truth for "AnnData → embedding" and "AnnData → ZINB μ
reconstruction" using the preprocessing convention stored with compatible
``ScTrilemmaModule`` checkpoints:

* per-cell ``library_size = sum(raw counts over ALL genes)`` (full cell)
* ``expr = log1p((raw / library_size) * target_sum)`` (CP10K + log1p)
* ``build_gene_mapping`` to translate ``adata.var_names`` (ENSEMBL) into the
  model's gene-vocab indices
* top-K per-cell crop for encoder; top-K dataset-level crop for decoder
* optional per-cell ``tissue_code`` / ``pseudo_bulk`` lookups keyed by
  ``f"{dataset_id}_{donor_id}"``
* ``tissue_code`` forwarded to BOTH ``model.encode`` and ``model.decode``;
  the released model consumes it only in the conditional prior, and
  ``decode`` ignores it.

Any ZSB harness should delegate to :func:`embed_adata` and
:func:`reconstruct_adata` so preprocessing drift across benchmark entrypoints
is avoided by construction.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

__all__ = ["embed_adata", "reconstruct_adata"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_gene_vocab(
    gene_vocab_path: Optional[str], gene_vocab: Optional[Dict[str, int]]
) -> Dict[str, int]:
    if gene_vocab is not None:
        return gene_vocab
    if gene_vocab_path is None:
        raise ValueError("gene_vocab or gene_vocab_path required")
    with open(gene_vocab_path) as f:
        return json.load(f)


def _build_col_to_vocab(adata, gene_vocab: Dict[str, int]) -> Dict[int, int]:
    from sctrilemma.data.gene_mapping import build_gene_mapping

    vocab_indices, file_indices = build_gene_mapping(
        list(adata.var_names), gene_vocab
    )
    return {int(fi): int(vi) for fi, vi in zip(file_indices, vocab_indices)}


def _build_donor_context(
    adata,
    tissue_code_dict: Optional[Dict[str, Any]],
    pseudo_bulk_dict: Optional[Dict[str, torch.Tensor]],
    model,
    gene_vocab_size: int,
) -> Tuple[np.ndarray, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Build per-donor lookup tables for tissue_code + pseudo_bulk.

    Returns
    -------
    donor_codes : (N,) int64 numpy array mapping cell idx → donor row
    tissue_code_per_cell : (N, tc_dim) tensor or None
    pb_table : (n_donors, vocab_size) tensor or None (slice with donor_codes)
    """
    dataset_id = str(adata.uns.get("dataset_id", "unknown"))
    if "donor_id" in adata.obs.columns:
        donor_ids = (
            adata.obs["donor_id"]
            .astype("string")
            .fillna("unknown")
            .astype(str)
            .to_numpy()
        )
    else:
        donor_ids = np.full(adata.n_obs, "unknown", dtype=object)
    unique_donors, donor_codes = np.unique(donor_ids, return_inverse=True)

    tissue_code_per_cell: Optional[torch.Tensor] = None
    tc_dim = int(getattr(model, "tissue_code_dim", 0))
    if tissue_code_dict is not None and tc_dim > 0:
        unknown_tc = tissue_code_dict.get(f"{dataset_id}_unknown")
        tc_rows = []
        for donor in unique_donors:
            key = f"{dataset_id}_{donor}"
            entry = tissue_code_dict.get(key, unknown_tc)
            if entry is None:
                entry = torch.zeros(tc_dim, dtype=torch.float32)
            tc_rows.append(torch.as_tensor(entry, dtype=torch.float32).view(tc_dim))
        tc_table = torch.stack(tc_rows, dim=0)  # (n_donors, tc_dim)
        tissue_code_per_cell = tc_table[torch.from_numpy(donor_codes).long()]

    pb_table: Optional[torch.Tensor] = None
    if pseudo_bulk_dict is not None:
        zero_pb = torch.zeros(gene_vocab_size, dtype=torch.float32)
        unknown_pb = pseudo_bulk_dict.get(f"{dataset_id}_unknown", zero_pb)
        pb_rows = []
        for donor in unique_donors:
            key = f"{dataset_id}_{donor}"
            entry = pseudo_bulk_dict.get(key, unknown_pb)
            pb_rows.append(torch.as_tensor(entry, dtype=torch.float32).view(-1))
        pb_table = torch.stack(pb_rows, dim=0)  # (n_donors, vocab_size)

    return donor_codes, tissue_code_per_cell, pb_table


def _preprocess_cell_batch(
    X_slice,
    adata_col_to_vocab: Dict[int, int],
    crop: int,
    target_sum: float,
    device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """log1p(CP10K) + vocab map + top-K crop for one mini-batch.

    ``X_slice`` is ``adata.X[start:end]`` (sparse or dense). Library size is
    computed per cell over the slice's full row sum. Crop selects top-K by
    log1p(CP10K) value (monotonic, so top-K by raw counts gives same genes).

    Returns ``(padded_idx, padded_val, mask)`` on ``device``.
    """
    import scipy.sparse as sp_mod

    X_csr = X_slice.tocsr() if sp_mod.issparse(X_slice) else None
    n = X_slice.shape[0]

    batch_indices: List[List[int]] = []
    batch_values: List[List[float]] = []

    for i in range(n):
        if X_csr is not None:
            row_start, row_end = X_csr.indptr[i], X_csr.indptr[i + 1]
            cols = X_csr.indices[row_start:row_end]
            vals = X_csr.data[row_start:row_end]
            cell_raw_sum = float(vals.sum())
        else:
            row = np.asarray(X_slice[i]).ravel()
            cell_raw_sum = float(row.sum())
            cols = np.where(row > 0)[0]
            vals = row[cols]

        lib_denom = cell_raw_sum if cell_raw_sum > 0.0 else 1.0

        mapped_idx: List[int] = []
        mapped_val: List[float] = []
        for j, c in enumerate(cols):
            vi = adata_col_to_vocab.get(int(c))
            if vi is None:
                continue
            expr = (float(vals[j]) / lib_denom) * target_sum
            mapped_idx.append(vi)
            mapped_val.append(float(np.log1p(expr)))

        if not mapped_idx:
            mapped_idx = [0]
            mapped_val = [0.0]

        if len(mapped_idx) > crop:
            top_k = np.argpartition(-np.array(mapped_val), crop)[:crop]
            mapped_idx = [mapped_idx[k] for k in top_k]
            mapped_val = [mapped_val[k] for k in top_k]

        batch_indices.append(mapped_idx)
        batch_values.append(mapped_val)

    max_len = max(len(x) for x in batch_indices)
    padded_idx = torch.zeros(n, max_len, dtype=torch.long, device=device)
    padded_val = torch.zeros(n, max_len, dtype=torch.float32, device=device)
    mask = torch.zeros(n, max_len, dtype=torch.bool, device=device)
    for i, (idxs, vs) in enumerate(zip(batch_indices, batch_values)):
        L = len(idxs)
        padded_idx[i, :L] = torch.tensor(idxs, dtype=torch.long)
        padded_val[i, :L] = torch.tensor(vs, dtype=torch.float32)
        mask[i, :L] = True

    return padded_idx, padded_val, mask


def _compute_full_cell_library(adata) -> torch.Tensor:
    """Per-cell raw count sum over **all** genes in ``adata.X``.

    Matches the checkpoint preprocessing convention: full cell raw sum used
    as ``library_size`` for the decoder.

    Prior ``ScTrilemmaInference._compute_full_library_size_all`` summed only over
    ``matched_file_indices`` (the cropped decode gene set), which diverged
    from this convention.
    """
    import scipy.sparse as sp_mod

    X = adata.X
    if sp_mod.issparse(X):
        lib = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    else:
        lib = np.asarray(X, dtype=np.float32).sum(axis=1)
    return torch.from_numpy(lib.astype(np.float32))


def _select_decode_gene_set(
    adata,
    adata_col_to_vocab: Dict[int, int],
    crop: int,
) -> Tuple[List[int], List[int], List[str]]:
    """Choose the gene set to reconstruct.

    Dataset-level top-K (by summed expression across cells, bounded by
    ``crop``) — every cell is decoded over this same fixed gene set.
    Matches the prior ``ScTrilemmaInference.reconstruct`` convention.

    Returns ``(matched_file_indices, matched_vocab_indices, decode_gene_names)``.
    """
    import scipy.sparse as sp_mod

    matched_file_indices = sorted(adata_col_to_vocab.keys())
    if crop and len(matched_file_indices) > crop:
        X = adata.X
        if sp_mod.issparse(X):
            gene_sums = np.array(
                X[:, matched_file_indices].sum(axis=0)
            ).ravel()
        else:
            gene_sums = np.asarray(
                X[:, matched_file_indices].sum(axis=0)
            ).ravel()
        top_k = np.argsort(gene_sums)[::-1][:crop]
        matched_file_indices = [matched_file_indices[i] for i in sorted(top_k)]
    matched_vocab_indices = [adata_col_to_vocab[fi] for fi in matched_file_indices]
    decode_gene_names = [str(adata.var_names[fi]) for fi in matched_file_indices]
    return matched_file_indices, matched_vocab_indices, decode_gene_names


# ---------------------------------------------------------------------------
# Public inference entry points
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_adata(
    pl_module,
    adata,
    *,
    gene_vocab_path: Optional[str] = None,
    gene_vocab: Optional[Dict[str, int]] = None,
    tissue_code_dict: Optional[Dict[str, Any]] = None,
    pseudo_bulk_dict: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 256,
    crop_size: Optional[int] = None,
    target_sum: float = 10000.0,
) -> np.ndarray:
    """Embed cells in ``adata`` using the checkpoint preprocessing convention.

    Returns ``np.ndarray`` of shape ``(adata.n_obs, d_model)``.
    """
    pl_module.eval()
    model = pl_module.model
    dev = next(model.parameters()).device

    cfg = getattr(pl_module, "cfg", None)
    cfg_train = (cfg.get("training", {}) if cfg is not None else {}) or {}
    crop = int(crop_size or cfg_train.get("crop_size", 4096) or 4096)
    d_model = (
        int(cfg.model.d_model)
        if cfg is not None and hasattr(cfg, "model")
        else int(getattr(pl_module, "embedding_dim", 256))
    )

    gene_vocab = _resolve_gene_vocab(gene_vocab_path, gene_vocab)
    adata_col_to_vocab = _build_col_to_vocab(adata, gene_vocab)
    donor_codes, tissue_code_per_cell, pb_table = _build_donor_context(
        adata, tissue_code_dict, pseudo_bulk_dict, model, len(gene_vocab)
    )

    n_cells = adata.n_obs
    embeddings = np.zeros((n_cells, d_model), dtype=np.float32)

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        X_slice = adata.X[start:end]

        padded_idx, padded_val, mask = _preprocess_cell_batch(
            X_slice, adata_col_to_vocab, crop, target_sum, dev
        )

        tissue_code_batch = (
            tissue_code_per_cell[start:end].to(dev)
            if tissue_code_per_cell is not None
            else None
        )
        pseudo_bulk_batch = None
        if pb_table is not None:
            pb_idx = torch.from_numpy(donor_codes[start:end]).long()
            pseudo_bulk_batch = pb_table[pb_idx].to(dev)

        _, vae_mu, _, _, _ = model.encode(
            meta_features={},
            raw_input=padded_val,
            mask=mask,
            gene_indices=padded_idx,
            pseudo_bulk=pseudo_bulk_batch,
            tissue_code=tissue_code_batch,
        )
        cell_repr = model.get_representation(vae_mu)
        embeddings[start:end] = cell_repr.float().cpu().numpy()

    return embeddings


@torch.no_grad()
def reconstruct_adata(
    pl_module,
    adata,
    *,
    gene_vocab_path: Optional[str] = None,
    gene_vocab: Optional[Dict[str, int]] = None,
    tissue_code_dict: Optional[Dict[str, Any]] = None,
    pseudo_bulk_dict: Optional[Dict[str, torch.Tensor]] = None,
    batch_size: int = 256,
    crop_size: Optional[int] = None,
    target_sum: float = 10000.0,
) -> Tuple[np.ndarray, List[str]]:
    """Reconstruct ZINB μ for cells in ``adata``.

    Pipeline:
        1. Preprocess encoder input (shared with :func:`embed_adata`):
           log1p(CP10K) + vocab map + per-cell top-K crop.
        2. Select decoder gene set (dataset-level top-K) — every cell is
           reconstructed over the same gene set.
        3. Full cell library ``(N,)`` from **all** genes in ``adata.X`` used
           as ``library_size`` for the decoder (matches training convention).
        4. ``tissue_code`` and ``pseudo_bulk`` forwarded to **both**
           ``model.encode`` and ``model.decode`` so decoder modes like
           ``tissue_code_mode="memory"`` activate.

    Returns
    -------
    recon_mu : np.ndarray ``(adata.n_obs, n_decode_genes)`` ZINB μ values
    decode_gene_names : ``List[str]`` of length ``n_decode_genes``
    """
    pl_module.eval()
    model = pl_module.model
    dev = next(model.parameters()).device

    cfg = getattr(pl_module, "cfg", None)
    cfg_train = (cfg.get("training", {}) if cfg is not None else {}) or {}
    crop = int(crop_size or cfg_train.get("crop_size", 4096) or 4096)

    gene_vocab = _resolve_gene_vocab(gene_vocab_path, gene_vocab)
    adata_col_to_vocab = _build_col_to_vocab(adata, gene_vocab)
    donor_codes, tissue_code_per_cell, pb_table = _build_donor_context(
        adata, tissue_code_dict, pseudo_bulk_dict, model, len(gene_vocab)
    )

    # Decoder gene set (fixed for the whole adata)
    _, matched_vocab_indices, decode_gene_names = _select_decode_gene_set(
        adata, adata_col_to_vocab, crop
    )
    n_decode_genes = len(matched_vocab_indices)
    decode_vocab_t = torch.tensor(
        matched_vocab_indices, dtype=torch.long, device=dev
    )

    # Full cell library (all genes, per cell) — used as decoder library_size
    full_library_all = _compute_full_cell_library(adata)

    n_cells = adata.n_obs
    recon_all = np.zeros((n_cells, n_decode_genes), dtype=np.float32)

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        B = end - start
        X_slice = adata.X[start:end]

        full_lib = full_library_all[start:end].to(dev)

        tissue_code_batch = (
            tissue_code_per_cell[start:end].to(dev)
            if tissue_code_per_cell is not None
            else None
        )
        pseudo_bulk_batch = None
        if pb_table is not None:
            pb_idx = torch.from_numpy(donor_codes[start:end]).long()
            pseudo_bulk_batch = pb_table[pb_idx].to(dev)

        # Encode (per-cell top-K crop via shared helper)
        padded_idx, padded_val, mask = _preprocess_cell_batch(
            X_slice, adata_col_to_vocab, crop, target_sum, dev
        )
        z, _, _, _, z_gene = model.encode(
            meta_features={},
            raw_input=padded_val,
            mask=mask,
            gene_indices=padded_idx,
            pseudo_bulk=pseudo_bulk_batch,
            tissue_code=tissue_code_batch,
        )

        # Decode (fixed decode gene set, full library, tissue_code passed so
        # decoder modes like tissue_code_mode="memory" activate).
        decode_idx = decode_vocab_t.unsqueeze(0).expand(B, -1)
        decode_mask = torch.ones(
            B, n_decode_genes, dtype=torch.bool, device=dev
        )
        gene_embs = model._build_gene_embeddings({}, decode_idx)
        zinb_mu, _, _, _ = model.decode(
            z,
            gene_embs,
            context=pseudo_bulk_batch,
            gene_indices=decode_idx,
            padding_mask=decode_mask,
            library_size=full_lib,
            z_gene=z_gene,
            tissue_code=tissue_code_batch,
        )
        recon_all[start:end] = zinb_mu.float().cpu().numpy()

    return recon_all, decode_gene_names
