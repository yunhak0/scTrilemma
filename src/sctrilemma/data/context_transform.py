"""Per-cell expression transform for scTrilemma datasets."""

from __future__ import annotations

from typing import Any

import torch


class ContextTransform:
    """Build the expression tensor consumed by the VAE encoder.

    The official scTrilemma VAE consumes expression-derived gene tokens, so this
    transform only computes the expression input and pseudo-bulk values.
    """

    def __init__(
        self,
        gene_metadata: object | None = None,
        vocab: dict[str, int] | None = None,
        max_len: int | None = None,
        use_raw_expr: bool = False,
    ) -> None:
        del gene_metadata, vocab
        self.max_len = max_len
        self.use_raw_expr = bool(use_raw_expr)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        gene_indices = sample["gene_indices"]
        expr_values = sample["expr_values"]

        if self.use_raw_expr:
            raw_input = expr_values
            pb_values = None
        else:
            pseudo_bulk = sample["pseudo_bulk"]
            pb_values = pseudo_bulk[gene_indices]
            raw_input = expr_values - pb_values

        raw_counts = sample.get("raw_counts")
        if raw_counts is not None:
            nonzero_mask = raw_counts > 0
        else:
            nonzero_mask = expr_values > 0
        raw_input = torch.where(nonzero_mask, raw_input, torch.zeros_like(raw_input))

        if "library_size" in sample and "library_size_full" not in sample:
            sample["library_size_full"] = float(sample["library_size"])

        if self.max_len is not None and len(gene_indices) > self.max_len:
            perm = torch.randperm(len(gene_indices))[: self.max_len]
            gene_indices = gene_indices[perm]
            raw_input = raw_input[perm]
            if pb_values is not None:
                pb_values = pb_values[perm]
            if raw_counts is not None:
                sample["raw_counts"] = raw_counts[perm]
                if "library_size" in sample:
                    sample["library_size"] = float(sample["raw_counts"].sum())
            count_weights = sample.get("count_weights")
            if count_weights is not None:
                if isinstance(count_weights, torch.Tensor):
                    sample["count_weights"] = count_weights[perm]
                else:
                    sample["count_weights"] = count_weights[perm.numpy()]

        sample["gene_indices"] = gene_indices
        sample["raw_input"] = raw_input
        sample["residual"] = raw_input
        if pb_values is not None:
            sample["pseudo_bulk_values"] = pb_values
        sample.pop("pseudo_bulk", None)
        return sample
