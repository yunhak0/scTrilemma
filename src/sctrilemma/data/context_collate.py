from __future__ import annotations

from typing import Any

import torch


def context_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate already-transformed ContextDataset items with padding."""
    if not batch:
        return {}

    lengths = [len(item["gene_indices"]) for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    gene_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
    raw_input = torch.zeros(batch_size, max_len, dtype=torch.float)
    raw_counts = torch.zeros(batch_size, max_len, dtype=torch.float)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    library_size = torch.zeros(batch_size, dtype=torch.float)
    library_size_full = torch.zeros(batch_size, dtype=torch.float)
    has_pseudo_bulk_values = any(item.get("pseudo_bulk_values") is not None for item in batch)
    pseudo_bulk_values = (
        torch.zeros(batch_size, max_len, dtype=torch.float)
        if has_pseudo_bulk_values
        else None
    )
    has_count_weights = any(item.get("count_weights") is not None for item in batch)
    count_weights = (
        torch.zeros(batch_size, max_len, dtype=torch.float)
        if has_count_weights
        else None
    )

    first_tc = batch[0].get("tissue_code")
    tissue_code_dense = (
        torch.zeros(batch_size, first_tc.numel(), dtype=torch.float)
        if first_tc is not None
        else None
    )

    cell_type_labels = torch.full((batch_size,), -1, dtype=torch.long)

    if "batch_label" in batch[0]:
        batch_labels = torch.tensor(
            [int(item.get("batch_label", -1)) for item in batch],
            dtype=torch.long,
        )
    else:
        batch_id_strs = [str(item.get("batch_id", "unknown")) for item in batch]
        batch_id_to_idx = {bid: idx for idx, bid in enumerate(sorted(set(batch_id_strs)))}
        batch_labels = torch.tensor(
            [batch_id_to_idx[bid] for bid in batch_id_strs],
            dtype=torch.long,
        )

    for i, item in enumerate(batch):
        length = lengths[i]
        gene_indices[i, :length] = item["gene_indices"]
        raw_input_i = item.get("raw_input", item.get("residual"))
        if raw_input_i is None:
            raise KeyError("ContextDataset item must contain raw_input or residual")
        raw_input[i, :length] = raw_input_i

        if "raw_counts" in item:
            raw_counts[i, :length] = item["raw_counts"]
        if "library_size" in item:
            library_size[i] = float(item["library_size"])
        elif "raw_counts" in item:
            library_size[i] = float(item["raw_counts"].sum())
        library_size_full[i] = float(item.get("library_size_full", library_size[i]))

        if count_weights is not None:
            item_weights = item.get("count_weights")
            if item_weights is not None:
                count_weights[i, :length] = torch.as_tensor(item_weights, dtype=torch.float)
            else:
                count_weights[i, :length] = 1.0

        if pseudo_bulk_values is not None and item.get("pseudo_bulk_values") is not None:
            pseudo_bulk_values[i, :length] = item["pseudo_bulk_values"]
        padding_mask[i, :length] = True

        if tissue_code_dense is not None and item.get("tissue_code") is not None:
            tissue_code_dense[i] = item["tissue_code"]
        if "cell_type_label" in item:
            cell_type_labels[i] = int(item["cell_type_label"])

    out: dict[str, Any] = {
        "gene_indices": gene_indices,
        "raw_input": raw_input,
        "residual": raw_input,
        "raw_counts": raw_counts,
        "padding_mask": padding_mask,
        "library_size": library_size,
        "library_size_full": library_size_full,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "cell_type_labels": cell_type_labels,
        "batch_labels": batch_labels,
        "dataset_id": [str(item.get("dataset_id", "")) for item in batch],
        "_is_padding": batch[0].get("_is_padding", False),
    }
    if pseudo_bulk_values is not None:
        out["pseudo_bulk_values"] = pseudo_bulk_values
    if count_weights is not None:
        out["count_weights"] = count_weights
    if tissue_code_dense is not None:
        out["tissue_code"] = tissue_code_dense
    return out
