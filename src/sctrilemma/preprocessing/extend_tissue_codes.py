"""Extend pseudo-bulk profiles and tissue codes to a second Census release.

The training configuration validates on datasets from a later Census release,
so the group-level inputs must cover both releases. This script merges the
pseudo-bulk dictionaries of the two releases (the training release takes
precedence for shared keys) and soft-assigns the new groups to the fixed
centroids fitted on the training release; the centroids are not refitted.

Inputs (under ``<data_root>``):
    <census_version>/pseudo_bulk_dict.pt
    <census_version>/tissue_codes_k{K}.pt
    <census_version>/tissue_code_centroids_k{K}.pt
    <extra_census_version>/pseudo_bulk_dict.pt

Outputs (under ``<data_root>/<census_version>``):
    pseudo_bulk_dict_with_<extra_census_version>.pt
    tissue_codes_k{K}_with_<extra_census_version>.pt

Usage:
    python -m sctrilemma.preprocessing.extend_tissue_codes \
        --data_root /path/to/cellxgene --census_version 20250130 \
        --extra_census_version 20251108 --k 32
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from sctrilemma.preprocessing.build_tissue_codes import softmax


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge pseudo-bulk dictionaries across releases and extend tissue codes."
    )
    parser.add_argument(
        "--data_root",
        default=os.environ.get("SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"),
    )
    parser.add_argument("--census_version", default="20250130", help="Release the codes were fitted on")
    parser.add_argument("--extra_census_version", default="20251108", help="Release to add")
    parser.add_argument("--pb_file", default="pseudo_bulk_dict.pt")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0, help="Must match build_tissue_codes")
    args = parser.parse_args()

    base_dir = Path(args.data_root) / args.census_version
    extra_dir = Path(args.data_root) / args.extra_census_version
    codes_path = base_dir / f"tissue_codes_k{args.k}.pt"
    centroids_path = base_dir / f"tissue_code_centroids_k{args.k}.pt"

    print("Loading inputs")
    pb_base: dict[str, torch.Tensor] = torch.load(base_dir / args.pb_file, map_location="cpu", weights_only=True)
    pb_extra: dict[str, torch.Tensor] = torch.load(extra_dir / args.pb_file, map_location="cpu", weights_only=True)
    codes: dict[str, torch.Tensor] = torch.load(codes_path, map_location="cpu", weights_only=True)
    centroids = torch.load(centroids_path, map_location="cpu", weights_only=True).numpy().astype(np.float32)
    print(f"  base groups: {len(pb_base)}, extra groups: {len(pb_extra)}, codes: {len(codes)}, centroids: {centroids.shape}")
    if centroids.shape[0] != next(iter(codes.values())).shape[0]:
        raise ValueError("centroid count does not match the code dimension")

    # Merge pseudo-bulk profiles; the base release wins for shared keys.
    pb_merged = dict(pb_base)
    new_keys = [k for k in pb_extra if k not in pb_base]
    for k in new_keys:
        pb_merged[k] = pb_extra[k]
    print(f"  added {len(new_keys)} groups -> {len(pb_merged)} total")

    # Use the same distance normalization as the fit: the distance std over
    # the groups the centroids were fitted on.
    base_keys = [k for k in codes if k in pb_merged]
    if not base_keys:
        raise RuntimeError("no overlap between the tissue-code dictionary and the pseudo-bulk profiles")
    d_base = cdist(np.stack([pb_merged[k].numpy() for k in base_keys]).astype(np.float32), centroids)
    dist_std = float(d_base.std())
    print(f"  distance std on the base release: {dist_std:.4f}")

    codes_merged = dict(codes)
    if new_keys:
        d_new = cdist(np.stack([pb_merged[k].numpy() for k in new_keys]).astype(np.float32), centroids)
        soft_new = softmax(-d_new / (dist_std * args.temperature + 1e-8), axis=-1)
        for i, k in enumerate(new_keys):
            codes_merged[k] = torch.from_numpy(soft_new[i]).float()
    print(f"  tissue codes: {len(codes_merged)} total")

    pb_out = base_dir / f"pseudo_bulk_dict_with_{args.extra_census_version}.pt"
    codes_out = base_dir / f"tissue_codes_k{args.k}_with_{args.extra_census_version}.pt"
    torch.save(pb_merged, pb_out)
    torch.save(codes_merged, codes_out)
    print(f"Saved {pb_out} ({pb_out.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {codes_out} ({codes_out.stat().st_size / 1e6:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
