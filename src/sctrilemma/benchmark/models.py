"""Model inference modules for zero-shot benchmarking."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

torch.serialization.add_safe_globals([DictConfig, ListConfig])


class BaseModelInference(ABC):
    """Base class for model inference modules."""

    model_name: str = "base"
    embedding_dim: int = 0

    @abstractmethod
    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Load model weights from path."""
        pass

    @abstractmethod
    def get_gene_vocabulary(self) -> List[str]:
        """Return expected gene vocabulary (Ensembl IDs)."""
        pass

    @abstractmethod
    def get_embeddings(
        self,
        adata: ad.AnnData,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Compute embeddings for cells in adata."""
        pass

    def embedding_key(self) -> str:
        return f"X_{self.model_name}"

    def clear_runtime_cache(self) -> None:
        """Release per-dataset temporary state while keeping weights loaded."""
        return None

    def is_generative(self) -> bool:
        """Return True for generation models (no embeddings, only generate counts).

        Generation-only models produce new cells from noise rather than encoding
        existing cells. The official reproduction registry does not include such
        models, but the hook is kept for metric dispatch compatibility.
        """
        return False

    def can_reconstruct(self) -> bool:
        """Return whether the model supports reconstruction metrics."""
        return False

    def reconstruct(
        self,
        adata: ad.AnnData,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, List[str]]:
        """Reconstruct gene expression. Returns (matrix, gene_names)."""
        raise NotImplementedError(f"{self.model_name} does not support reconstruction")


class ScTrilemmaInference(BaseModelInference):
    """scTrilemma Cross-Attention VAE inference for zero-shot benchmarking.

    Requires a compatible ScTrilemmaModule Lightning checkpoint and the
    CellxGene Census gene vocabulary JSON used by that checkpoint.
    """

    model_name = "sctrilemma"
    embedding_dim = 256

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        gene_vocab_path: Optional[str] = None,
        pseudo_bulk_path: Optional[str] = None,
        tissue_code_path: Optional[str] = None,
        crop_size: int = 4096,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._gene_vocab_path = gene_vocab_path
        # None → auto-load from ckpt cfg.data on load_model(); explicit path
        # overrides.
        self._pseudo_bulk_path = pseudo_bulk_path
        self._tissue_code_path = tissue_code_path
        self._crop_size = crop_size
        self._encode_crop_size = crop_size
        self.pl_module = None
        self.model = None
        self._gene_vocab: Optional[dict] = None
        self._gene_vocab_list: Optional[List[str]] = None
        self._pseudo_bulk_dict: Optional[dict[str, torch.Tensor]] = None
        self._tissue_code_dict: Optional[dict] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:

        from sctrilemma.model.vae_module import ScTrilemmaModule

        ckpt_path = model_path or self._checkpoint_path
        if ckpt_path is None:
            raise ValueError("checkpoint_path is required for ScTrilemmaInference")

        # Gene vocab: {gene_name: global_index}
        if self._gene_vocab_path is None:
            raise ValueError("gene_vocab_path is required for ScTrilemmaInference")
        with open(self._gene_vocab_path) as f:
            self._gene_vocab = json.load(f)
        self._gene_vocab_list = sorted(
            self._gene_vocab.keys(), key=lambda g: self._gene_vocab[g]
        )

        # Load Lightning checkpoint
        pl_module = ScTrilemmaModule.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", weights_only=False
        )
        pl_module.to(self._device)
        pl_module.eval()
        self.pl_module = pl_module
        self.model = pl_module.model
        self.embedding_dim = pl_module.cfg.model.d_model
        self._crop_size = pl_module.cfg.training.get("crop_size", 4096) or 4096
        self._encode_crop_size = (
            pl_module.cfg.training.get("encode_crop_size", None) or self._crop_size
        )

        # Auto-inherit tissue_code / pseudo_bulk paths from the ckpt's cfg.data
        # unless the caller passed an explicit override. This keeps ZSB from
        # drifting when a new training config changes the dict paths.
        cfg_data = pl_module.cfg.get("data", {}) if pl_module.cfg is not None else {}
        if self._tissue_code_path is None:
            self._tissue_code_path = cfg_data.get("tissue_code_path")
        if self._pseudo_bulk_path is None:
            self._pseudo_bulk_path = cfg_data.get("pseudo_bulk_path")

        if self._tissue_code_path:
            tc_path = Path(self._tissue_code_path)
            if not tc_path.exists():
                raise FileNotFoundError(f"scTrilemma tissue_code file not found: {tc_path}")
            self._tissue_code_dict = torch.load(
                tc_path, map_location="cpu", weights_only=True
            )

        if self._pseudo_bulk_path:
            pb_path = Path(self._pseudo_bulk_path)
            if not pb_path.exists():
                raise FileNotFoundError(f"scTrilemma pseudo-bulk file not found: {pb_path}")
            loaded_pb = torch.load(pb_path, map_location="cpu", weights_only=True)
            self._pseudo_bulk_dict = {
                str(key): torch.as_tensor(value, dtype=torch.float32).cpu()
                for key, value in loaded_pb.items()
            }
        if self._requires_pseudo_bulk() and self._pseudo_bulk_dict is None:
            raise ValueError(
                "This scTrilemma checkpoint requires pseudo-bulk conditioning. "
                "Pass --sctrilemma-pseudo-bulk pointing to pseudo_bulk_dict.pt "
                "(or add cfg.data.pseudo_bulk_path to the training config)."
            )
        print(f"scTrilemma model loaded from {ckpt_path} (d_model={self.embedding_dim})")

    def get_gene_vocabulary(self) -> List[str]:
        if self._gene_vocab_list is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._gene_vocab_list

    def _prepare_batch(
        self,
        X_slice,
        adata_gene_to_vocab: dict,
        max_genes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert sparse rows to (gene_indices, log1p(CP10K) values, mask, library_size).

        Input format matches the checkpoint preprocessing convention: values are
        ``log1p((raw / lib_size_full) * 10000)`` where ``lib_size_full`` is the
        per-cell raw count sum over ALL genes (not just mapped/cropped ones).
        """
        import scipy.sparse as sp_mod

        if sp_mod.issparse(X_slice):
            X_csr = X_slice.tocsr()
        else:
            X_csr = None

        batch_indices: List[list] = []
        batch_values: List[list] = []
        batch_lib: List[float] = []

        n = X_slice.shape[0]
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

            mapped_idx = []
            mapped_val_raw = []
            for j, c in enumerate(cols):
                vi = adata_gene_to_vocab.get(int(c))
                if vi is not None:
                    mapped_idx.append(vi)
                    mapped_val_raw.append(float(vals[j]))

            if not mapped_idx:
                mapped_idx = [0]
                mapped_val_raw = [0.0]

            # CP10K normalize then log1p:
            # expr = (raw / lib) * target_sum; expr = log1p(expr).
            mapped_val = [
                float(np.log1p((v / lib_denom) * 10000.0)) for v in mapped_val_raw
            ]

            # Crop to top-expressed if needed (log1p is monotonic, so cropping by
            # log1p(CP10K) picks the same genes as cropping by raw counts).
            if len(mapped_idx) > max_genes:
                top_k = np.argpartition(-np.array(mapped_val), max_genes)[:max_genes]
                mapped_idx = [mapped_idx[k] for k in top_k]
                mapped_val = [mapped_val[k] for k in top_k]

            batch_lib.append(cell_raw_sum)
            batch_indices.append(mapped_idx)
            batch_values.append(mapped_val)

        # Pad to max length in this mini-batch
        max_len = max(len(x) for x in batch_indices)
        B = len(batch_indices)
        dev = self._device

        padded_idx = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        padded_val = torch.zeros(B, max_len, dtype=torch.float32, device=dev)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=dev)

        for i in range(B):
            L = len(batch_indices[i])
            padded_idx[i, :L] = torch.tensor(batch_indices[i], dtype=torch.long)
            padded_val[i, :L] = torch.tensor(batch_values[i], dtype=torch.float32)
            mask[i, :L] = True

        library_size = torch.tensor(batch_lib, dtype=torch.float32, device=dev)
        return padded_idx, padded_val, mask, library_size

    def _build_gene_map(self, adata: ad.AnnData) -> dict:
        """Map adata column index → scTrilemma vocab index for matched genes."""
        from sctrilemma.data.gene_mapping import build_gene_mapping

        vocab_indices, file_indices = build_gene_mapping(
            list(adata.var_names), self._gene_vocab  # type: ignore[arg-type]
        )
        return {int(fi): int(vi) for fi, vi in zip(file_indices, vocab_indices)}

    def _requires_pseudo_bulk(self) -> bool:
        if self.model is None:
            return False
        return bool(
            getattr(self.model, "encoder_pb_injection_mode", "")
            or getattr(self.model, "decoder_pb_placement_mode", "")
            or getattr(self.model, "decoder_global_pb_prior_mode", "")
        )

    def _zero_pseudo_bulk(self) -> torch.Tensor:
        return torch.zeros(len(self.get_gene_vocabulary()), dtype=torch.float32)

    def _prepare_pseudo_bulk_table(
        self,
        adata: ad.AnnData,
    ) -> tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        if not self._requires_pseudo_bulk():
            return None, None
        if self._pseudo_bulk_dict is None:
            raise RuntimeError("Pseudo-bulk conditioning is required but not loaded.")

        dataset_id = str(adata.uns.get("dataset_id", "unknown"))
        if "donor_id" in adata.obs.columns:
            donor_ids = adata.obs["donor_id"].fillna("unknown").astype(str).to_numpy()
        else:
            donor_ids = np.full(adata.n_obs, "unknown", dtype=object)

        unique_donors, donor_codes = np.unique(donor_ids, return_inverse=True)
        unknown_key = f"{dataset_id}_unknown"
        unknown_pb = self._pseudo_bulk_dict.get(unknown_key, self._zero_pseudo_bulk())

        pb_rows: list[torch.Tensor] = []
        for donor in unique_donors:
            key = f"{dataset_id}_{donor}"
            pb_rows.append(self._pseudo_bulk_dict.get(key, unknown_pb))
        pb_table = torch.stack([row.float().cpu() for row in pb_rows], dim=0)
        return donor_codes.astype(np.int64), pb_table

    def _compute_full_library_size_all(
        self, adata: ad.AnnData, matched_file_indices: List[int]
    ) -> torch.Tensor:
        """Compute per-cell library size from ALL matched genes, once for the
        whole dataset.  Called outside the batch loop to avoid repeated
        sparse-matrix format conversions.

        Returns a CPU tensor of shape (n_cells,).
        """
        import scipy.sparse as sp_mod

        matched_cols = np.asarray(matched_file_indices, dtype=np.intp)
        X = adata.X

        if sp_mod.issparse(X):
            # Slice all matched columns once (CSR → CSC slice → sum)
            lib = np.asarray(
                X[:, matched_cols].sum(axis=1), dtype=np.float32
            ).ravel()
        else:
            lib = np.asarray(X, dtype=np.float32)[:, matched_cols].sum(axis=1)

        return torch.from_numpy(lib.astype(np.float32))

    def get_embeddings(self, adata: ad.AnnData, batch_size: int = 256) -> np.ndarray:
        """Delegate to the shared ``embed_adata`` helper — single source of
        truth for scTrilemma preprocessing (log1p(CP10K), vocab mapping, crop,
        tissue_code / pseudo_bulk lookup). Keeps this class aligned with the
        public inference helper.
        """
        if self.pl_module is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        from sctrilemma.inference import embed_adata

        return embed_adata(
            self.pl_module,
            adata,
            gene_vocab=self._gene_vocab,
            tissue_code_dict=self._tissue_code_dict,
            pseudo_bulk_dict=self._pseudo_bulk_dict,
            batch_size=batch_size,
            crop_size=self._encode_crop_size,
        )

    def _legacy_get_embeddings_reference(self, adata: ad.AnnData, batch_size: int = 256) -> np.ndarray:
        # Retained only as a reference for debugging; no longer on the hot
        # path. Matches the pre-refactor embedding procedure (missing log1p
        # and tissue_code by construction — kept so regressions can be
        # reproduced A/B if needed).
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gene_map = self._build_gene_map(adata)
        donor_codes, pb_table = self._prepare_pseudo_bulk_table(adata)
        n_cells = adata.n_obs
        embeddings = np.zeros((n_cells, self.embedding_dim), dtype=np.float32)

        with torch.no_grad():
            for start in tqdm(range(0, n_cells, batch_size), desc="scTrilemma embed"):
                end = min(start + batch_size, n_cells)
                X_slice = adata.X[start:end]
                gene_idx, raw_val, mask, _ = self._prepare_batch(
                    X_slice, gene_map, self._encode_crop_size
                )
                pseudo_bulk = None
                if donor_codes is not None and pb_table is not None:
                    pb_idx = torch.from_numpy(donor_codes[start:end]).long()
                    pseudo_bulk = pb_table[pb_idx].to(self._device)
                _, mu, _, _, _ = self.model.encode(
                    meta_features={},
                    raw_input=raw_val,
                    mask=mask,
                    gene_indices=gene_idx,
                    pseudo_bulk=pseudo_bulk,
                )
                cell_repr = self.model.get_representation(mu)
                embeddings[start:end] = cell_repr.cpu().numpy()

        return embeddings

    def can_reconstruct(self) -> bool:
        return True

    def reconstruct(
        self,
        adata: ad.AnnData,
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, List[str]]:
        """Delegate to shared ``reconstruct_adata`` helper — single source of
        truth for preprocessing, decoder gene selection, and tissue_code /
        pseudo_bulk forwarding (encode + decode). Keeps this class from
        drifting vs the public inference helper.
        """
        if self.pl_module is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        from sctrilemma.inference import reconstruct_adata

        return reconstruct_adata(
            self.pl_module,
            adata,
            gene_vocab=self._gene_vocab,
            tissue_code_dict=self._tissue_code_dict,
            pseudo_bulk_dict=self._pseudo_bulk_dict,
            batch_size=batch_size,
            crop_size=self._encode_crop_size,
        )

    def clear_runtime_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


MODEL_REGISTRY = {
    "sctrilemma": ScTrilemmaInference,
}


def get_model(model_name: str, **kwargs) -> BaseModelInference:
    """Factory for benchmark models."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return ScTrilemmaInference(
        checkpoint_path=kwargs.get("checkpoint_path"),
        gene_vocab_path=kwargs.get("gene_vocab_path"),
        pseudo_bulk_path=kwargs.get("pseudo_bulk_path"),
    )
