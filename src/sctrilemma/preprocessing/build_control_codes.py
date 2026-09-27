"""Build control versions of the group tissue codes for the PB-Cond ablations.

Given the tissue-code dictionary used for training (``tissue_codes_k{K}_with_<release>.pt``),
write two controls next to it:

* ``*_shuffled_seed{S}.pt`` -- the same code vectors with their assignment to groups permuted
  (the prior still varies across groups, but no longer matches their pseudo-bulk profiles);
* ``*_constant_mean.pt`` -- every group receives the mean code (a learned, unconditional prior).

Usage:
    python -m sctrilemma.preprocessing.build_control_codes \
        --codes /path/to/20250130/tissue_codes_k32_with_20251108.pt --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch


def build_controls(codes_path: Path, seed: int) -> tuple[Path, Path]:
    source: dict[str, torch.Tensor] = torch.load(codes_path, map_location="cpu", weights_only=True)
    keys = [key for key in source if key != "__unknown__"]
    values = [source[key].clone().to(torch.float32) for key in keys]

    rng = random.Random(seed)
    perm = list(range(len(keys)))
    rng.shuffle(perm)
    shuffled = {key: values[idx] for key, idx in zip(keys, perm)}
    shuffled_path = codes_path.with_name(f"{codes_path.stem}_shuffled_seed{seed}.pt")
    torch.save(shuffled, shuffled_path)

    mean = torch.stack(values).mean(dim=0)
    constant = {key: mean.clone() for key in keys}
    constant_path = codes_path.with_name(f"{codes_path.stem}_constant_mean.pt")
    torch.save(constant, constant_path)
    return shuffled_path, constant_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shuffled and constant control tissue codes.")
    parser.add_argument("--codes", type=Path, required=True, help="Tissue-code dictionary used for training")
    parser.add_argument("--seed", type=int, default=42, help="Permutation seed for the shuffled control")
    args = parser.parse_args()
    shuffled_path, constant_path = build_controls(args.codes, args.seed)
    print(f"Saved {shuffled_path}")
    print(f"Saved {constant_path}")


if __name__ == "__main__":
    main()
