"""Decode the sampled cells of every held-out dataset over the dataset's shared gene list.

For each dataset the staged sample (``experiments/prepare_samples.py``, at most 2,500 cells)
is encoded with the released checkpoint and decoded over the shared per-dataset gene list in
``experiments/data/genelists`` (the gene universe of Table 2 / Table 19). One NPZ cache per
dataset is written with ``recon`` (N, G) float32 ZINB means on the raw-count scale,
``gene_names`` (G), ``soma_joinid`` (N, sample order), ``model`` and a JSON ``metadata``
string; ``score_agreement.py`` scores these caches.

    pixi run python -m experiments.reconstruction.generate
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import load_sampled_adata, metadata_json, read_ids
from experiments.reconstruction._util import (
    DEFAULT_GENELIST_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_SAMPLES_DIR,
    MODEL_NAME,
    add_model_arguments,
    describe_path,
    load_inference,
    read_genelist,
    reconstruct_shared_genes,
)

DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "cache" / MODEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+", help="Subset of dataset IDs to process")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Sampled datasets written by experiments/prepare_samples.py",
    )
    parser.add_argument("--genelist-dir", type=Path, default=DEFAULT_GENELIST_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8, help="Cells per forward pass")
    parser.add_argument("--force", action="store_true", help="Rewrite valid existing caches")
    add_model_arguments(parser)
    return parser.parse_args()


def cache_is_valid(path: Path, expected_soma: np.ndarray) -> bool:
    """True when ``path`` holds a well-formed cache for exactly the staged cell order."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cached:
            recon = cached["recon"]
            genes = cached["gene_names"]
            soma = np.asarray(cached["soma_joinid"], dtype=np.int64)
        return (
            recon.ndim == 2
            and recon.shape == (soma.size, len(genes))
            and np.array_equal(soma, expected_soma)
        )
    except (OSError, ValueError, KeyError):
        return False


def main() -> int:
    args = parse_args()
    dataset_ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model": MODEL_NAME,
        "dataset_ids": dataset_ids,
        "samples_dir": describe_path(args.samples_dir),
        "genelist_dir": describe_path(args.genelist_dir),
        "checkpoint": describe_path(args.checkpoint),
        "gene_vocab": describe_path(args.gene_vocab),
        "pseudo_bulk": describe_path(args.pseudo_bulk),
        "tissue_code": describe_path(args.tissue_code),
        "batch_size": args.batch_size,
        "device": args.device,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    started = time.perf_counter()
    wrapper = load_inference(args)
    print(f"Loaded {MODEL_NAME} in {time.perf_counter() - started:.1f}s", flush=True)
    written = skipped = failures = 0
    for index, dataset_id in enumerate(dataset_ids, start=1):
        output_path = args.output_dir / f"{dataset_id}.npz"
        adata = None
        try:
            adata = load_sampled_adata(args.samples_dir / f"{dataset_id}.h5ad", dataset_id)
            soma = adata.obs["soma_joinid"].to_numpy().astype(np.int64)
            if not args.force and cache_is_valid(output_path, soma):
                skipped += 1
                print(f"[{index}/{len(dataset_ids)}] {dataset_id}: cached", flush=True)
                continue
            dataset_started = time.perf_counter()
            genelist = read_genelist(args.genelist_dir / f"{dataset_id}.txt")
            recon, gene_names = reconstruct_shared_genes(
                wrapper, adata, genelist, batch_size=args.batch_size
            )
            recon = np.asarray(recon, dtype=np.float32)
            genes = np.asarray([str(value) for value in gene_names], dtype=str)
            if recon.shape != (adata.n_obs, genes.size):
                raise RuntimeError(
                    f"Unexpected reconstruction shape {recon.shape}; "
                    f"expected {(adata.n_obs, genes.size)}"
                )
            if not np.isfinite(recon).all() or np.any(recon < 0):
                raise RuntimeError("Reconstruction contains non-finite or negative values")
            temporary = output_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                recon=recon,
                gene_names=genes,
                soma_joinid=soma,
                model=np.asarray(MODEL_NAME),
                metadata=metadata_json(
                    model=MODEL_NAME,
                    dataset_id=dataset_id,
                    checkpoint=run_config["checkpoint"],
                    genelist=describe_path(args.genelist_dir / f"{dataset_id}.txt"),
                    n_cells=int(adata.n_obs),
                    n_genes=int(genes.size),
                    n_genelist=len(genelist),
                    batch_size=args.batch_size,
                    values="ZINB mean on the raw-count scale (full-cell library size)",
                ),
            )
            temporary.replace(output_path)
            written += 1
            print(
                f"[{index}/{len(dataset_ids)}] {dataset_id}: cells={adata.n_obs}, "
                f"genes={genes.size}/{len(genelist)}, "
                f"size={output_path.stat().st_size / 1e6:.1f} MB, "
                f"time={time.perf_counter() - dataset_started:.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{index}/{len(dataset_ids)}] {dataset_id}: FAILED {exc}", flush=True)
        finally:
            wrapper.clear_runtime_cache()
            del adata
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(
        f"Done: model={MODEL_NAME}, written={written}, skipped={skipped}, failures={failures}",
        flush=True,
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
