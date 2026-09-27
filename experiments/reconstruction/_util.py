"""Helpers shared by the reconstruction-fidelity modules.

They are specific to decoding the released checkpoint over a fixed gene list and to the
result layout of this folder; general sampling, alignment and scoring helpers live in
``experiments/common.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from sctrilemma.benchmark.models import ScTrilemmaInference
from sctrilemma.inference import (
    _build_col_to_vocab,
    _build_donor_context,
    _compute_full_cell_library,
    _preprocess_cell_batch,
    _resolve_gene_vocab,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"
    )
)
DEFAULT_IDS_FILE = ROOT / "configs/zsb/full_89_ids.txt"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/sctrilemma/final.ckpt"
DEFAULT_GENE_VOCAB = Path(
    os.environ.get(
        "SCTRILEMMA_GENE_VOCAB",
        str(DATA_ROOT / "20250130/gene_vocab_homo_sapiens_20250130.json"),
    )
)
DEFAULT_SAMPLES_DIR = ROOT / "outputs/experiments/samples"
DEFAULT_GENELIST_DIR = ROOT / "experiments/data/genelists"
DEFAULT_RESULTS_ROOT = ROOT / "outputs/experiments/reconstruction"
MODEL_NAME = "sctrilemma"
DISPLAY_NAMES = {MODEL_NAME: "scTrilemma"}
TARGET_SUM = 10_000.0


def read_genelist(path: Path) -> list[str]:
    """Read one shared per-dataset gene list (one identifier per line, file order kept)."""
    genes = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not genes:
        raise ValueError(f"Empty gene list: {path}")
    return genes


def display_name(model_name: str, override: str | None = None) -> str:
    """Table label of a model name."""
    return override or DISPLAY_NAMES.get(model_name, model_name)


def describe_path(path: Path | None) -> str | None:
    """Repository-relative string for run metadata (absolute only outside the repository)."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI options selecting the checkpoint, its gene vocabulary and the compute device."""
    group = parser.add_argument_group("model")
    group.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    group.add_argument(
        "--gene-vocab",
        type=Path,
        default=DEFAULT_GENE_VOCAB,
        help="Census gene vocabulary JSON used by the checkpoint "
        "(default: $SCTRILEMMA_GENE_VOCAB or <data root>/20250130/...)",
    )
    group.add_argument(
        "--pseudo-bulk",
        type=Path,
        default=None,
        help="Optional donor pseudo-bulk dictionary (the released checkpoint does not use it)",
    )
    group.add_argument(
        "--tissue-code",
        type=Path,
        default=None,
        help="Optional donor tissue-code dictionary (enters only the KL prior; decoding ignores it)",
    )
    group.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto = CUDA when available",
    )


def load_inference(args: argparse.Namespace) -> ScTrilemmaInference:
    """Build and load the public inference wrapper for the released checkpoint."""
    wrapper = ScTrilemmaInference(
        checkpoint_path=str(args.checkpoint),
        gene_vocab_path=str(args.gene_vocab),
        pseudo_bulk_path=str(args.pseudo_bulk) if args.pseudo_bulk else None,
        tissue_code_path=str(args.tissue_code) if args.tissue_code else None,
    )
    if args.device != "auto":
        # The wrapper picks CUDA whenever it is available; honour an explicit choice.
        wrapper._device = args.device
    wrapper.load_model()
    wrapper.pl_module.eval()
    return wrapper


@torch.no_grad()
def reconstruct_shared_genes(
    wrapper: ScTrilemmaInference,
    adata: ad.AnnData,
    decode_genes: list[str],
    *,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Decode ZINB means over an explicit gene list instead of the dataset-level top-K crop.

    Mirrors ``sctrilemma.inference.reconstruct_adata`` (encoder preprocessing with the
    checkpoint's crop, full-cell library size, donor context forwarded to encode and decode)
    but keeps ``decode_genes`` in the given order, dropping only genes absent from ``adata``
    or the vocabulary. Returns raw-count-scale means ``(N, G)`` and the decoded gene names.
    """
    pl_module = wrapper.pl_module
    model = wrapper.model
    if pl_module is None or model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    dev = next(model.parameters()).device
    gene_vocab = _resolve_gene_vocab(None, wrapper._gene_vocab)
    crop = int(wrapper._encode_crop_size)
    col_to_vocab = _build_col_to_vocab(adata, gene_vocab)
    donor_codes, tissue_code_per_cell, pb_table = _build_donor_context(
        adata, wrapper._tissue_code_dict, wrapper._pseudo_bulk_dict, model, len(gene_vocab)
    )
    var_to_col = {str(name): index for index, name in enumerate(adata.var_names)}
    file_indices = [
        var_to_col[gene]
        for gene in decode_genes
        if gene in var_to_col and var_to_col[gene] in col_to_vocab
    ]
    if not file_indices:
        raise ValueError("decode_genes contains no gene present in adata and the vocabulary")
    decode_gene_names = [str(adata.var_names[index]) for index in file_indices]
    decode_vocab = torch.tensor(
        [col_to_vocab[index] for index in file_indices], dtype=torch.long, device=dev
    )
    full_library = _compute_full_cell_library(adata)

    n_cells, n_genes = adata.n_obs, len(file_indices)
    recon = np.zeros((n_cells, n_genes), dtype=np.float32)
    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        size = end - start
        tissue = (
            tissue_code_per_cell[start:end].to(dev) if tissue_code_per_cell is not None else None
        )
        pseudo_bulk = None
        if pb_table is not None:
            pseudo_bulk = pb_table[torch.from_numpy(donor_codes[start:end]).long()].to(dev)
        idx, val, mask = _preprocess_cell_batch(
            adata.X[start:end], col_to_vocab, crop, TARGET_SUM, dev
        )
        z, _, _, _, z_gene = model.encode(
            meta_features={},
            raw_input=val,
            mask=mask,
            gene_indices=idx,
            pseudo_bulk=pseudo_bulk,
            tissue_code=tissue,
        )
        decode_idx = decode_vocab.unsqueeze(0).expand(size, -1)
        decode_mask = torch.ones(size, n_genes, dtype=torch.bool, device=dev)
        gene_embs = model._build_gene_embeddings({}, decode_idx)
        zinb_mu, _, _, _ = model.decode(
            z,
            gene_embs,
            context=pseudo_bulk,
            gene_indices=decode_idx,
            padding_mask=decode_mask,
            library_size=full_library[start:end].to(dev),
            z_gene=z_gene,
            tissue_code=tissue,
        )
        recon[start:end] = zinb_mu.float().cpu().numpy()
    return recon, decode_gene_names
