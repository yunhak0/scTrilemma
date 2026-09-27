# ruff: noqa: E402
"""Export scTrilemma embeddings of the held-out datasets to per-dataset caches.

Each dataset is loaded from the held-out Census release, capped at 100,000 cells with the
deterministic dataset-hash sampler at seed 0 (the benchmark's input protocol), embedded
with the posterior mean of the checkpoint's latent encoder, and written to
``<output-dir>/embeddings/<dataset>.npz`` with aligned arrays ``embeddings`` (N, 512)
float32, ``labels`` (cell_type_ontology_term_id) and ``batches`` (donor_id, or dataset_id
when absent). Caches are the input of ``experiments/scoring`` (Table 1); no metric is
computed here. A ``cache_manifest.csv`` and ``metadata.json`` record the export.

    python -m sctrilemma.benchmark.export_embeddings --checkpoint checkpoints/sctrilemma/final.ckpt
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, os.environ.get("CPU_THREADS", "4"))

import numpy as np
import torch

from sctrilemma.benchmark.data_utils import (
    DEFAULT_TARGET_PATH,
    _default_data_root,
    load_dataset,
)
from sctrilemma.benchmark.models import ScTrilemmaInference
from sctrilemma.benchmark.run import _batch_key_for_adata, _sample_cells_for_analysis

LABEL_KEY = "cell_type_ontology_term_id"
MAX_CELLS_PER_DATASET = 100_000
SAMPLE_SEED = 0
BATCH_SIZE = 256
EMBEDDING_DIM = 512
EMBEDDING_MODE = "mu"  # posterior mean; the only mode of ScTrilemmaInference.get_embeddings
DEFAULT_IDS_FILE = Path("configs/zsb/full_89_ids.txt")
DEFAULT_CHECKPOINT = Path("checkpoints/sctrilemma/final.ckpt")
DEFAULT_OUTPUT_DIR = Path("outputs/experiments/embeddings/sctrilemma")
MANIFEST_FIELDS = [
    "dataset",
    "cache_path",
    "n_cells",
    "embedding_dim",
    "batch_key",
    "n_labels",
    "n_batches",
    "input_cap",
    "sample_seed",
    "status",
]


def _default_gene_vocab() -> Path:
    explicit = os.environ.get("SCTRILEMMA_GENE_VOCAB")
    if explicit:
        return Path(explicit).expanduser()
    return _default_data_root() / "20250130" / "gene_vocab_homo_sapiens_20250130.json"


def _read_dataset_ids(ids_file: Path, explicit_ids: list[str]) -> list[str]:
    """Read unique dataset IDs while preserving manifest order."""
    if explicit_ids:
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("--dataset-id values must be unique")
        return explicit_ids
    dataset_ids = [
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not dataset_ids:
        raise ValueError(f"No dataset IDs found in {ids_file}")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError(f"Dataset manifest contains duplicate IDs: {ids_file}")
    return dataset_ids


def _git_revision() -> dict[str, str]:
    """Return the current revision, retaining an explicit failure reason."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
        return {"git_revision": completed.stdout.strip(), "git_revision_status": "available"}
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        return {"git_revision": "unavailable", "git_revision_status": detail.strip()}


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed cache atomically so interrupted exports are never reused."""
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def _write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    """Persist cache integrity metadata with stable column order."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _cache_is_valid(cache_path: Path, *, embedding_dim: int, max_cells: int) -> bool:
    """Return whether a completed cache has the required aligned arrays."""
    if not cache_path.exists():
        return False
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            embeddings = cached["embeddings"]
            labels = cached["labels"]
            batches = cached["batches"]
        return (
            embeddings.ndim == 2
            and embeddings.shape[1] == embedding_dim
            and embeddings.dtype == np.float32
            and labels.ndim == 1
            and batches.ndim == 1
            and len(embeddings) == len(labels) == len(batches)
            and len(embeddings) <= max_cells
        )
    except (KeyError, OSError, ValueError):
        return False


def _load_model(args: argparse.Namespace) -> ScTrilemmaInference:
    model = ScTrilemmaInference(
        checkpoint_path=str(args.checkpoint),
        gene_vocab_path=str(args.gene_vocab),
        pseudo_bulk_path=str(args.pseudo_bulk) if args.pseudo_bulk else None,
        tissue_code_path=str(args.tissue_code) if args.tissue_code else None,
    )
    model.load_model()
    if model.embedding_dim != args.embedding_dim:
        raise RuntimeError(
            f"checkpoint exports d_model={model.embedding_dim}, expected {args.embedding_dim} "
            "(pass --embedding-dim to export another checkpoint)"
        )
    return model


def export_dataset(
    dataset_id: str,
    model: ScTrilemmaInference,
    *,
    target_path: Path,
    cache_path: Path,
    max_cells: int,
    sample_seed: int,
    batch_size: int,
    embedding_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Embed one dataset under the input protocol and write its cache."""
    adata = load_dataset(dataset_id, str(target_path))
    adata = _sample_cells_for_analysis(adata, max_cells=max_cells, seed=sample_seed, dataset_id=dataset_id)
    batch_key = _batch_key_for_adata(adata)
    labels = adata.obs[LABEL_KEY].astype("string").fillna("unknown").astype(str).to_numpy()
    batches = adata.obs[batch_key].astype("string").fillna("unknown").astype(str).to_numpy()
    embeddings = np.asarray(model.get_embeddings(adata, batch_size=batch_size), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != embedding_dim:
        raise RuntimeError(f"{dataset_id}: expected (N, {embedding_dim}) embeddings, got {embeddings.shape}")
    if not (len(embeddings) == len(labels) == len(batches) == adata.n_obs):
        raise RuntimeError(
            f"{dataset_id}: row alignment failed: embeddings={len(embeddings)}, "
            f"labels={len(labels)}, batches={len(batches)}, adata={adata.n_obs}"
        )
    if len(embeddings) > max_cells:
        raise RuntimeError(f"{dataset_id}: input cap violated ({len(embeddings)} cells)")
    labels = np.asarray(labels, dtype=str)
    batches = np.asarray(batches, dtype=str)
    soma = (np.asarray(adata.obs["soma_joinid"], dtype=np.int64) if "soma_joinid" in adata.obs
            else np.full(len(embeddings), -1, dtype=np.int64))
    _atomic_save_npz(cache_path, embeddings=embeddings, labels=labels, batches=batches, soma_joinid=soma)
    del adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, labels, batches, batch_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-id", action="append", default=[], help="Export one or more IDs only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--target-path", type=Path, default=Path(DEFAULT_TARGET_PATH),
                        help="Held-out Census release, <root>/20251108/by_dataset")
    parser.add_argument("--gene-vocab", type=Path, default=_default_gene_vocab())
    parser.add_argument("--tissue-code", type=Path, default=None,
                        help="Optional tissue-code dict; defaults to the path stored in the checkpoint config")
    parser.add_argument("--pseudo-bulk", type=Path, default=None,
                        help="Optional pseudo-bulk dict; defaults to the path stored in the checkpoint config")
    parser.add_argument("--max-cells-per-dataset", type=int, default=MAX_CELLS_PER_DATASET)
    parser.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM,
                        help="Expected d_model of the checkpoint (cache validation)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.gene_vocab.is_file():
        raise FileNotFoundError(f"gene vocabulary not found: {args.gene_vocab}")
    if not args.target_path.is_dir():
        raise FileNotFoundError(f"target path not found: {args.target_path}")

    dataset_ids = _read_dataset_ids(args.dataset_ids_file, args.dataset_id)
    output_dir = args.output_dir.resolve()
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "analysis": "scTrilemma zero-shot embedding export",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint_path": str(args.checkpoint),
        "target_path": str(args.target_path),
        "gene_vocab_path": str(args.gene_vocab),
        "tissue_code_path": str(args.tissue_code) if args.tissue_code else "from checkpoint config",
        "pseudo_bulk_path": str(args.pseudo_bulk) if args.pseudo_bulk else "from checkpoint config",
        "dataset_ids_file": str(args.dataset_ids_file),
        "dataset_ids_requested": dataset_ids,
        "max_cells_per_dataset": args.max_cells_per_dataset,
        "sample_seed": args.sample_seed,
        "batch_size": args.batch_size,
        "expected_embedding_dim": args.embedding_dim,
        "embedding_mode": EMBEDDING_MODE,
        "label_key": LABEL_KEY,
        "inference_only": True,
        **_git_revision(),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def cache_ok(path: Path) -> bool:
        return _cache_is_valid(path, embedding_dim=args.embedding_dim, max_cells=args.max_cells_per_dataset)

    missing_ids = [dataset_id for dataset_id in dataset_ids if not cache_ok(embedding_dir / f"{dataset_id}.npz")]
    model: ScTrilemmaInference | None = _load_model(args) if missing_ids else None

    rows: list[dict[str, Any]] = []
    for position, dataset_id in enumerate(dataset_ids, start=1):
        cache_path = embedding_dir / f"{dataset_id}.npz"
        if cache_ok(cache_path):
            with np.load(cache_path, allow_pickle=False) as cached:
                embeddings = cached["embeddings"]
                labels = cached["labels"]
                batches = cached["batches"]
            batch_key = "cached"
            status = "reused"
        else:
            assert model is not None
            embeddings, labels, batches, batch_key = export_dataset(
                dataset_id,
                model,
                target_path=args.target_path,
                cache_path=cache_path,
                max_cells=args.max_cells_per_dataset,
                sample_seed=args.sample_seed,
                batch_size=args.batch_size,
                embedding_dim=args.embedding_dim,
            )
            status = "exported"

        rows.append({
            "dataset": dataset_id,
            "cache_path": str(cache_path),
            "n_cells": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "batch_key": batch_key,
            "n_labels": int(np.unique(labels).size),
            "n_batches": int(np.unique(batches).size),
            "input_cap": args.max_cells_per_dataset,
            "sample_seed": args.sample_seed,
            "status": status,
        })
        _write_manifest(rows, output_dir / "cache_manifest.csv")
        print(f"[{position}/{len(dataset_ids)}] {dataset_id}: {status}, {len(embeddings):,} x {embeddings.shape[1]}", flush=True)

    metadata["completed_at_utc"] = datetime.now(UTC).isoformat()
    metadata["n_caches_requested"] = len(rows)
    metadata["n_caches_exported"] = sum(row["status"] == "exported" for row in rows)
    metadata["n_caches_reused"] = sum(row["status"] == "reused" for row in rows)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {len(rows)} cache records at {embedding_dir}", flush=True)


if __name__ == "__main__":
    main()
