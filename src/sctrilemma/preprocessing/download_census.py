"""Download CellxGene Census expression data grouped by dataset.

The output layout matches the training dataloader and cache preprocessor:

    {output_dir}/{version}/by_dataset/{dataset_id}/part_000.h5ad
    {output_dir}/{version}/by_dataset/{dataset_id}/.completed
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cellxgene_census
import pandas as pd

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message=".*is_categorical_dtype.*"
)
try:
    from anndata import ImplicitModificationWarning

    warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
except ImportError:
    pass

CENSUS_VERSION = "2025-01-30"
DEFAULT_ORGANISM = "homo_sapiens"
DEFAULT_OUTPUT_DIR = "/scratch/${USER}/datasets/cellxgene"
MAX_CELLS_PER_FILE = 100_000


def _version_tag(census_version: str) -> str:
    return census_version.replace("-", "")


def _output_root(output_dir: str | Path) -> Path:
    return Path(os.path.expandvars(str(output_dir))).expanduser()


def load_grouped_soma_idx(
    organism: str,
    census_version: str,
    output_dir: str | Path,
    min_cells: int = 2,
) -> dict[str, list[int]]:
    """Load downloaded cell metadata and group soma_joinid values by dataset."""
    version = _version_tag(census_version)
    metadata_path = (
        _output_root(output_dir)
        / version
        / f"cell_metadata_{organism}_{version}.parquet"
    )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cell metadata not found at {metadata_path}. "
            "Run sctrilemma-download-metadata first."
        )

    df = pd.read_parquet(metadata_path, columns=["dataset_id", "soma_joinid"])
    if isinstance(df["dataset_id"].dtype, pd.CategoricalDtype):
        df["dataset_id"] = df["dataset_id"].astype(str)

    grouped = df.groupby("dataset_id")["soma_joinid"].apply(list).to_dict()
    filtered = {dataset_id: ids for dataset_id, ids in grouped.items() if len(ids) >= min_cells}
    print(f"Found {len(filtered):,} datasets with at least {min_cells} cells.", flush=True)
    return filtered


def download_dataset_group(
    census_version: str,
    dataset_id: str,
    soma_ids: list[int],
    output_dir: str | Path,
    organism: str,
    max_cells_per_file: int,
) -> str:
    """Download one dataset into one or more h5ad shards."""
    version = _version_tag(census_version)
    dataset_dir = _output_root(output_dir) / version / "by_dataset" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if (dataset_dir / ".completed").exists() and list(dataset_dir.glob("*.h5ad")):
        return f"Skipped {dataset_id} (already completed)"

    n_parts = (len(soma_ids) + max_cells_per_file - 1) // max_cells_per_file
    try:
        with cellxgene_census.open_soma(census_version=census_version) as census:
            for part_idx in range(n_parts):
                start = part_idx * max_cells_per_file
                end = min(start + max_cells_per_file, len(soma_ids))
                chunk_ids = soma_ids[start:end]
                if not chunk_ids:
                    continue

                file_path = dataset_dir / f"part_{part_idx:03d}.h5ad"
                if file_path.exists():
                    continue

                adata = cellxgene_census.get_anndata(
                    census,
                    organism=organism,
                    obs_coords=chunk_ids,
                )
                if "feature_id" in adata.var.columns:
                    adata.var.index = pd.Index(adata.var["feature_id"])
                    adata.var_names_make_unique()
                adata.obs_names_make_unique()
                adata.uns["dataset_id"] = dataset_id
                adata.write_h5ad(file_path)

        (dataset_dir / ".completed").touch()
        return f"Completed {dataset_id} ({len(soma_ids):,} cells)"
    except Exception as exc:
        return f"Failed {dataset_id}: {exc}"


def _download_wrapper(args: tuple[str, str, list[int], str, str, int]) -> str:
    return download_dataset_group(*args)


def run_download(
    census_version: str = CENSUS_VERSION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    organism: str = DEFAULT_ORGANISM,
    max_workers: int = 4,
    test_limit: int = 0,
    min_cells: int = 2,
    max_cells_per_file: int = MAX_CELLS_PER_FILE,
) -> None:
    """Download all metadata-selected datasets in parallel."""
    grouped_ids = load_grouped_soma_idx(
        organism=organism,
        census_version=census_version,
        output_dir=output_dir,
        min_cells=min_cells,
    )
    datasets = sorted(grouped_ids.items(), key=lambda item: len(item[1]), reverse=True)
    if test_limit > 0:
        datasets = datasets[:test_limit]

    if not datasets:
        print("No datasets to download.", flush=True)
        return

    tasks = [
        (census_version, dataset_id, ids, str(output_dir), organism, max_cells_per_file)
        for dataset_id, ids in datasets
    ]
    print(f"Starting download for {len(tasks):,} datasets with {max_workers} workers.", flush=True)

    success_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_download_wrapper, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result.startswith("Failed"):
                fail_count += 1
            else:
                success_count += 1
            print(f"[{idx}/{len(tasks)}] {result}", flush=True)

    print(f"Finished. Success: {success_count}, failed: {fail_count}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CellxGene Census h5ad shards grouped by dataset"
    )
    parser.add_argument("--census_version", default=CENSUS_VERSION)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--organism", default=DEFAULT_ORGANISM)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--test_limit", type=int, default=0)
    parser.add_argument("--min_cells", type=int, default=2)
    parser.add_argument("--max_cells_per_file", type=int, default=MAX_CELLS_PER_FILE)
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    run_download(
        census_version=args.census_version,
        output_dir=args.output_dir,
        organism=args.organism,
        max_workers=args.max_workers,
        test_limit=args.test_limit,
        min_cells=args.min_cells,
        max_cells_per_file=args.max_cells_per_file,
    )


if __name__ == "__main__":
    cli()
