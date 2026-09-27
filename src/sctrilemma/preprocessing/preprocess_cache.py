"""Preprocess h5ad files into vocab-mapped sparse CSR .npz caches.

Converts raw h5ad files into numpy-native .npz files that
load 100-1000x faster than sc.read_h5ad(), eliminating HDF5 decompression
and AnnData object creation overhead during DDP training.

Usage:
    python -m sctrilemma.preprocessing.preprocess_cache \
        --data_root /scratch/${USER}/datasets/cellxgene \
        --census_version 20250130 \
        --vocab_path .../gene_vocab_homo_sapiens_20250130.json \
        --cell_type_vocab_path .../cell_type_vocab_homo_sapiens_20250130.json \
        --num_workers 32

Pattern: follows calculate_pseudo_bulk.py (ProcessPoolExecutor, incremental).
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scanpy as sc
from scipy import sparse

from sctrilemma.data.gene_mapping import build_gene_mapping

logger = logging.getLogger(__name__)

CACHE_VERSION = 1


def _compute_vocab_hash(vocab_path: str | Path) -> str:
    """SHA-256 of the gene vocab JSON file contents."""
    h = hashlib.sha256()
    with open(vocab_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_cache_fresh(
    meta_path: Path,
    h5ad_path: Path,
    gene_vocab_hash: str,
) -> bool:
    """Check if an existing cache file is still valid."""
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("cache_version") != CACHE_VERSION:
            return False
        if meta.get("gene_vocab_hash") != gene_vocab_hash:
            return False
        stat = h5ad_path.stat()
        if meta.get("source_h5ad_size") != stat.st_size:
            return False
        return True
    except Exception:
        return False


def preprocess_single_file(args: tuple[str, str, dict[str, int], dict[str, int] | None, int, str, bool]) -> dict[str, str | int | float]:
    """Convert one h5ad file to a .npz cache.

    Parameters (packed as tuple for ProcessPoolExecutor):
        h5ad_path, cache_dir, gene_vocab, cell_type_vocab,
        vocab_size, gene_vocab_hash, compress

    Returns dict with status info: {dataset_id, shard, status, n_cells, elapsed}.
    """
    (
        h5ad_path,
        cache_dir,
        gene_vocab,
        cell_type_vocab,
        vocab_size,
        gene_vocab_hash,
        compress,
    ) = args

    h5ad_path = Path(h5ad_path)
    cache_dir = Path(cache_dir)
    dataset_id = h5ad_path.parent.name
    shard_name = h5ad_path.stem  # e.g. "part_000"

    result: dict[str, str | int | float] = {
        "dataset_id": dataset_id,
        "shard": shard_name,
        "status": "skipped",
        "n_cells": 0,
        "elapsed": 0.0,
    }

    t0 = time.monotonic()

    # Output paths
    ds_cache_dir = cache_dir / dataset_id
    cache_path = ds_cache_dir / f"{shard_name}.npz"
    meta_path = ds_cache_dir / f"{shard_name}.meta.json"

    # Staleness check
    if _is_cache_fresh(meta_path, h5ad_path, gene_vocab_hash):
        result["elapsed"] = time.monotonic() - t0
        return result

    try:
        # 1. Load h5ad
        adata = sc.read_h5ad(h5ad_path)

        # 2. Gene mapping
        vocab_indices_map, valid_file_indices = build_gene_mapping(
            list(adata.var_names), gene_vocab
        )

        if len(valid_file_indices) == 0:
            result["status"] = "no_valid_genes"
            result["elapsed"] = time.monotonic() - t0
            del adata
            gc.collect()
            return result

        # 3. CSR remapping: select valid gene columns, then remap indices to vocab space
        expr_raw = adata.X
        if expr_raw is None:
            result["status"] = "error: adata.X is None"
            result["elapsed"] = time.monotonic() - t0
            del adata
            gc.collect()
            return result

        # Convert to csr_matrix for uniform typing (csr_matrix(x) is a no-copy view when already CSR)
        expr_matrix = sparse.csr_matrix(expr_raw)

        # Select only valid gene columns and ensure CSR
        X_valid = sparse.csr_matrix(expr_matrix[:, valid_file_indices])

        n_cells: int = X_valid.shape[0]  # pyright: ignore[reportOptionalSubscript]

        # Remap CSR indices from file-local valid-gene positions to global vocab indices
        new_indices: npt.NDArray[np.int32] = vocab_indices_map[X_valid.indices]
        X_remapped = sparse.csr_matrix(
            (X_valid.data.astype(np.float32), new_indices.astype(np.int32), X_valid.indptr.astype(np.int64)),
            shape=(n_cells, vocab_size),
        )

        # 4. Per-cell metadata
        # Donor codes
        donor_ids_raw: npt.NDArray[np.str_]
        if "donor_id" in adata.obs.columns:
            donor_ids_raw = np.asarray(adata.obs["donor_id"].astype(str).fillna("unknown").values)
        else:
            donor_ids_raw = np.array(["unknown"] * n_cells)

        unique_donors, donor_codes_raw = np.unique(donor_ids_raw, return_inverse=True)
        donor_codes = donor_codes_raw.astype(np.int32)

        # Cell type labels
        ct_col = (
            "cell_type_ontology_term_id"
            if "cell_type_ontology_term_id" in adata.obs.columns
            else "cell_type" if "cell_type" in adata.obs.columns
            else None
        )
        if cell_type_vocab and ct_col:
            cts = adata.obs[ct_col].astype(str).values
            ct_labels = np.array(
                [cell_type_vocab.get(ct, -1) for ct in cts], dtype=np.int32
            )
        else:
            ct_labels = np.full(n_cells, -1, dtype=np.int32)

        # Library sizes (raw sums per cell) – expr_matrix already validated as non-None
        library_sizes = np.asarray(expr_matrix.sum(axis=1)).flatten().astype(np.float32)

        # Nonzero counts per cell (from remapped matrix)
        nnz_counts = np.diff(X_remapped.indptr).astype(np.int32)

        # 5. Save .npz
        ds_cache_dir.mkdir(parents=True, exist_ok=True)

        save_fn = np.savez_compressed if compress else np.savez
        save_fn(
            cache_path,
            csr_data=X_remapped.data,
            csr_indices=X_remapped.indices,
            csr_indptr=X_remapped.indptr,
            donor_codes=donor_codes,
            donor_id_map=np.array(list(unique_donors), dtype=object),
            cell_type_labels=ct_labels,
            library_sizes=library_sizes,
            nnz_counts=nnz_counts,
            dataset_id=np.array(dataset_id),
            n_cells=np.array(n_cells),
            vocab_size=np.array(vocab_size),
        )

        # 6. Save meta.json
        stat = h5ad_path.stat()
        meta = {
            "cache_version": CACHE_VERSION,
            "gene_vocab_hash": gene_vocab_hash,
            "source_h5ad_mtime": stat.st_mtime,
            "source_h5ad_size": stat.st_size,
            "n_cells": n_cells,
            "n_valid_genes": len(valid_file_indices),
            "created_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        result["status"] = "ok"
        result["n_cells"] = n_cells

        del adata, expr_raw, expr_matrix, X_valid, X_remapped
        gc.collect()

    except Exception as e:
        result["status"] = f"error: {e}"
        gc.collect()

    result["elapsed"] = time.monotonic() - t0
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess h5ad files into .npz tensor caches"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to data root (containing census_version)",
    )
    parser.add_argument(
        "--census_version", type=str, required=True,
        help="Census version date (e.g. 20250130)",
    )
    parser.add_argument(
        "--vocab_path", type=str, required=True,
        help="Path to gene vocab JSON",
    )
    parser.add_argument(
        "--cell_type_vocab_path", type=str, default=None,
        help="Path to cell type vocab JSON (optional)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--compress", action="store_true",
        help="Use np.savez_compressed instead of np.savez (slower load, smaller files)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration of all caches (ignore staleness check)",
    )
    parser.add_argument(
        "--dataset_ids_file", type=str, default=None,
        help="Optional file with dataset UUIDs (one per line, '#' comments OK). If provided, only those datasets are cached.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_root = Path(args.data_root)
    by_dataset_path = data_root / args.census_version / "by_dataset"
    cache_dir = data_root / args.census_version / "by_dataset_cache"

    if not by_dataset_path.exists():
        print(f"Error: Data directory not found: {by_dataset_path}", file=sys.stderr)
        sys.exit(1)

    # Load vocabs
    print(f"Loading gene vocab from {args.vocab_path}...")
    with open(args.vocab_path) as f:
        gene_vocab = json.load(f)
    vocab_size = len(gene_vocab)
    gene_vocab_hash = _compute_vocab_hash(args.vocab_path)
    print(f"Vocab size: {vocab_size}, hash: {gene_vocab_hash[:12]}...")

    cell_type_vocab = None
    if args.cell_type_vocab_path and Path(args.cell_type_vocab_path).exists():
        with open(args.cell_type_vocab_path) as f:
            cell_type_vocab = json.load(f)
        print(f"Cell type vocab size: {len(cell_type_vocab)}")

    # If force mode, remove existing cache
    if args.force and cache_dir.exists():
        import shutil
        print(f"Force mode: removing existing cache at {cache_dir}")
        shutil.rmtree(cache_dir)

    # Check vocab hash consistency with existing cache
    if cache_dir.exists():
        existing_metas = list(cache_dir.rglob("*.meta.json"))
        if existing_metas:
            sample_meta = json.loads(existing_metas[0].read_text())
            if sample_meta.get("gene_vocab_hash") != gene_vocab_hash:
                print(
                    "WARNING: Gene vocab hash mismatch! "
                    + f"Existing cache uses {sample_meta.get('gene_vocab_hash', 'unknown')[:12]}..., "
                    + f"current vocab is {gene_vocab_hash[:12]}... "
                    + "All caches will be regenerated."
                )

    # Discover all h5ad files; optionally filter by dataset id list.
    dataset_dirs = sorted([d for d in by_dataset_path.iterdir() if d.is_dir()])
    if args.dataset_ids_file:
        with open(args.dataset_ids_file) as f:
            wanted = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
        before = len(dataset_dirs)
        dataset_dirs = [d for d in dataset_dirs if d.name in wanted]
        missing = wanted - {d.name for d in dataset_dirs}
        print(
            f"Filtered datasets via {args.dataset_ids_file}: "
            f"{before} → {len(dataset_dirs)} (missing on disk: {len(missing)})"
        )
        if missing:
            for m in sorted(missing)[:5]:
                print(f"  missing: {m}")
    all_h5ad_files = []
    for d in dataset_dirs:
        all_h5ad_files.extend(sorted(d.glob("*.h5ad")))

    print(f"Found {len(all_h5ad_files)} h5ad files across {len(dataset_dirs)} datasets")

    # Prepare work items
    work_items = []
    for h5ad_path in all_h5ad_files:
        work_items.append((
            str(h5ad_path),
            str(cache_dir),
            gene_vocab,
            cell_type_vocab,
            vocab_size,
            gene_vocab_hash,
            args.compress,
        ))

    num_workers = args.num_workers or os.cpu_count() or 1
    print(f"Processing with {num_workers} workers...")

    # Process
    total = len(work_items)
    done = 0
    ok_count = 0
    skip_count = 0
    error_count = 0
    total_cells = 0
    t_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(preprocess_single_file, item): item[0]
            for item in work_items
        }

        for future in as_completed(future_map):
            h5ad_path = future_map[future]
            done += 1
            try:
                result = future.result()
                status = result["status"]
                if status == "ok":
                    ok_count += 1
                    total_cells += int(result["n_cells"])
                elif status == "skipped":
                    skip_count += 1
                else:
                    error_count += 1
                    print(
                        f"  [{done}/{total}] ERROR {result['dataset_id']}/{result['shard']}: {status}",
                        file=sys.stderr,
                    )

                if done % 50 == 0 or done == total:
                    elapsed = time.monotonic() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{done}/{total}] ok={ok_count} skip={skip_count} err={error_count} "
                        + f"cells={total_cells:,} rate={rate:.1f} files/s"
                    )

            except Exception as e:
                error_count += 1
                done += 1
                print(f"  [{done}/{total}] FATAL {h5ad_path}: {e}", file=sys.stderr)

    elapsed = time.monotonic() - t_start
    print(
        f"\nDone in {elapsed:.1f}s. "
        + f"ok={ok_count}, skipped={skip_count}, errors={error_count}, "
        + f"total_cells={total_cells:,}"
    )

    # Summary: cache size
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        print(f"Cache directory: {cache_dir}")
        print(f"Total cache size: {total_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
