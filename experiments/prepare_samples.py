"""Stage one deterministic, metadata-stratified sample (<= 2,500 cells) per held-out dataset.

The samples are the shared input of the matched-cohort analyses (Table 2 and its appendix
tables) and of the latent interventions (Figure 4 / Table 20).

    pixi run python -m experiments.prepare_samples
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments.common import (
    read_ids,
    stable_seed,
    stratified_sample_indices,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"))
DEFAULT_DATA_ROOT = DATA_ROOT / "20251108" / "by_dataset"
DEFAULT_IDS_FILE = ROOT / "configs/zsb/full_89_ids.txt"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/experiments/samples"


def read_dataset_obs(paths: list[Path]) -> pd.DataFrame:
    """Read only obs tables and attach source-row coordinates."""
    frames: list[pd.DataFrame] = []
    offset = 0
    for shard_index, path in enumerate(paths):
        backed = ad.read_h5ad(path, backed="r")
        try:
            obs = backed.obs.copy()
            obs["__shard_index"] = shard_index
            obs["__row_index"] = np.arange(backed.n_obs, dtype=np.int64)
            obs["__global_index"] = np.arange(
                offset, offset + backed.n_obs, dtype=np.int64
            )
            offset += backed.n_obs
            frames.append(obs)
        finally:
            backed.file.close()
    return pd.concat(frames, axis=0, ignore_index=True)


def materialize_selected(
    paths: list[Path],
    selected_obs: pd.DataFrame,
    dataset_id: str,
) -> ad.AnnData:
    """Load only selected rows from each shard and restore global sample order."""
    pieces: list[ad.AnnData] = []
    selected_obs = selected_obs.copy()
    selected_obs["__sample_order"] = np.arange(len(selected_obs), dtype=np.int64)
    for shard_index, group in selected_obs.groupby("__shard_index", sort=True):
        local_rows = group["__row_index"].to_numpy(dtype=np.int64)
        backed = ad.read_h5ad(paths[int(shard_index)], backed="r")
        try:
            piece = backed[local_rows].to_memory()
        finally:
            backed.file.close()
        piece.obs["__sample_order"] = group["__sample_order"].to_numpy(dtype=np.int64)
        pieces.append(piece)

    combined = (
        pieces[0]
        if len(pieces) == 1
        else ad.concat(
            pieces,
            axis=0,
            join="outer",
            index_unique="-",
            merge="same",
            fill_value=0,
        )
    )
    order = np.argsort(combined.obs["__sample_order"].to_numpy(dtype=np.int64))
    combined = combined[order].copy()
    combined.obs.drop(columns="__sample_order", inplace=True)
    if "feature_id" in combined.var.columns:
        combined.var_names = combined.var["feature_id"].astype(str).to_numpy()
        combined.var_names_make_unique()
    combined.obs_names_make_unique()
    combined.uns["dataset_id"] = dataset_id
    if "dataset_id" not in combined.obs.columns:
        combined.obs["dataset_id"] = dataset_id
    if "organism_ontology_term_id" not in combined.obs.columns:
        combined.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    if "soma_joinid" not in combined.obs.columns:
        raise RuntimeError(f"{dataset_id} lacks obs['soma_joinid']")
    soma = combined.obs["soma_joinid"].to_numpy().astype(np.int64)
    if np.unique(soma).size != soma.size:
        raise RuntimeError(f"{dataset_id} sampled duplicate soma_joinid values")
    return combined


def stage_dataset(
    dataset_id: str,
    *,
    data_root: Path,
    output_dir: Path,
    max_cells: int,
    seed: int,
    force: bool,
) -> dict[str, object]:
    """Create one resumable sampled H5AD and return its audit row."""
    output_path = output_dir / f"{dataset_id}.h5ad"
    if output_path.exists() and not force:
        cached = ad.read_h5ad(output_path, backed="r")
        try:
            return {
                "dataset_id": dataset_id,
                "status": "cached",
                "n_total": int(cached.uns.get("source_n_cells", -1)),
                "n_sampled": int(cached.n_obs),
                "n_genes": int(cached.n_vars),
                "n_shards": int(cached.uns.get("source_n_shards", -1)),
                "sample_path": str(output_path),
                "seconds": 0.0,
            }
        finally:
            cached.file.close()

    started = time.perf_counter()
    paths = sorted((data_root / dataset_id).glob("*.h5ad"))
    if not paths:
        raise FileNotFoundError(f"No h5ad files under {data_root / dataset_id}")
    obs = read_dataset_obs(paths)
    sample_indices = stratified_sample_indices(
        obs,
        max_cells=max_cells,
        seed=stable_seed(seed, dataset_id),
        keys=("cell_type", "disease"),
    )
    selected_obs = obs.iloc[sample_indices].copy()
    sampled = materialize_selected(paths, selected_obs, dataset_id)
    expected_soma = selected_obs["soma_joinid"].to_numpy().astype(np.int64)
    actual_soma = sampled.obs["soma_joinid"].to_numpy().astype(np.int64)
    if not np.array_equal(expected_soma, actual_soma):
        raise RuntimeError(f"{dataset_id} sampled soma_joinid order mismatch")

    sampled.uns["source_n_cells"] = int(len(obs))
    sampled.uns["source_n_shards"] = int(len(paths))
    sampled.uns["sampling_seed"] = int(seed)
    sampled.uns["sampling_keys"] = ["cell_type", "disease"]
    sampled.uns["max_cells"] = int(max_cells)
    temporary = output_path.with_suffix(".tmp.h5ad")
    sampled.write_h5ad(temporary, compression="gzip", compression_opts=4)
    temporary.replace(output_path)
    seconds = time.perf_counter() - started
    return {
        "dataset_id": dataset_id,
        "status": "written",
        "n_total": int(len(obs)),
        "n_sampled": int(sampled.n_obs),
        "n_genes": int(sampled.n_vars),
        "n_shards": int(len(paths)),
        "sample_path": str(output_path),
        "seconds": seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-cells", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "dataset_ids_file": str(args.dataset_ids_file),
                "dataset_ids": dataset_ids,
                "data_root": str(args.data_root),
                "max_cells": args.max_cells,
                "seed": args.seed,
                "sampling_keys": ["cell_type", "disease"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    failures = 0
    for index, dataset_id in enumerate(dataset_ids, start=1):
        try:
            row = stage_dataset(
                dataset_id,
                data_root=args.data_root,
                output_dir=args.output_dir,
                max_cells=args.max_cells,
                seed=args.seed,
                force=args.force,
            )
            rows.append(row)
            print(
                f"[{index}/{len(dataset_ids)}] {dataset_id}: {row['status']} "
                f"{row['n_sampled']}/{row['n_total']} cells, {row['n_genes']} genes, "
                f"{float(row['seconds']):.1f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
            print(f"[{index}/{len(dataset_ids)}] {dataset_id}: FAILED {exc}", flush=True)

        temporary_csv = args.output_dir / ".sample_manifest.tmp.csv"
        pd.DataFrame(rows).to_csv(temporary_csv, index=False)
        temporary_csv.replace(args.output_dir / "sample_manifest.csv")

    print(f"Done: datasets={len(dataset_ids)}, failures={failures}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
