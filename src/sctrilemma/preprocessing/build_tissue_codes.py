"""Build group-level tissue codes from pseudo-bulk profiles.

The pseudo-bulk vectors of all dataset--donor groups (see
``calculate_pseudobulk.py``) are clustered with k-means. Each group is then
represented by a soft assignment over the ``K`` centroids, which the model
uses to parameterize its conditional prior (PB-Cond).

Outputs (written next to the pseudo-bulk file):
    tissue_codes_k{K}.pt            dict[group_key] -> (K,) soft assignment
    tissue_code_centroids_k{K}.pt   (K, vocab_size) centroid matrix

Usage:
    python -m sctrilemma.preprocessing.build_tissue_codes \
        --data_root /path/to/cellxgene --census_version 20250130 --k 32
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def soft_assign(pb_matrix: np.ndarray, centroids: np.ndarray, temperature: float) -> np.ndarray:
    """Soft-assign rows of ``pb_matrix`` to ``centroids``.

    Euclidean distances are scaled by their standard deviation over the whole
    matrix before the softmax, so the assignment sharpness does not depend on
    the absolute expression scale.
    """
    distances = cdist(pb_matrix, centroids)  # (N, K)
    distances_norm = distances / (distances.std() + 1e-8)
    return softmax(-distances_norm / temperature, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build k-means tissue codes from pseudo-bulk profiles.")
    parser.add_argument(
        "--data_root",
        default=os.environ.get("SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"),
        help="Root directory containing <census_version>/pseudo_bulk_dict.pt",
    )
    parser.add_argument("--census_version", default="20250130")
    parser.add_argument("--pb_file", default="pseudo_bulk_dict.pt")
    parser.add_argument("--k", type=int, default=32, help="Number of k-means centroids")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_init", type=int, default=5)
    args = parser.parse_args()

    version_dir = Path(args.data_root) / args.census_version
    pb_path = version_dir / args.pb_file

    print(f"Loading pseudo-bulk profiles from {pb_path}")
    pb_dict: dict[str, torch.Tensor] = torch.load(pb_path, map_location="cpu", weights_only=True)
    keys = list(pb_dict.keys())
    pb_matrix = torch.stack([pb_dict[k] for k in keys]).numpy().astype(np.float32)
    n_groups, vocab_size = pb_matrix.shape
    print(f"  {n_groups} groups x {vocab_size} genes")

    print(f"Running k-means with K={args.k} (seed={args.seed}, n_init={args.n_init})")
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=args.n_init, verbose=0)
    kmeans.fit(pb_matrix)
    centroids = kmeans.cluster_centers_.astype(np.float32)  # (K, vocab_size)

    soft_codes = soft_assign(pb_matrix, centroids, args.temperature)
    code_dict = {keys[i]: torch.from_numpy(soft_codes[i]).float() for i in range(n_groups)}

    out_codes = version_dir / f"tissue_codes_k{args.k}.pt"
    torch.save(code_dict, out_codes)
    print(f"  Saved {out_codes} ({out_codes.stat().st_size / 1e6:.1f} MB)")

    out_centroids = version_dir / f"tissue_code_centroids_k{args.k}.pt"
    torch.save(torch.from_numpy(centroids).float(), out_centroids)
    print(f"  Saved {out_centroids}")

    sizes = np.bincount(np.argmax(soft_codes, axis=1), minlength=args.k)
    print(f"  Cluster sizes (argmax): min={sizes.min()} max={sizes.max()} std={sizes.std():.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
