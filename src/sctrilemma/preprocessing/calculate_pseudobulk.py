import argparse
import gc
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import numpy.typing as npt
import scanpy as sc
import torch
from scipy import sparse


def load_gene_vocab(file_path: str | Path) -> dict[str, int]:
    file_path = Path(file_path)
    with open(file_path) as f:
        gene_vocab: dict[str, int] = json.load(f)
    return gene_vocab

def calculate_donor_pseudo_bulk(
    dataset_path: Path,
    dataset_id: str,
    vocab: dict[str, int],
    vocab_size: int,
    target_sum: float = 10000.0
) -> dict[str, torch.Tensor]:
    """
    Compute donor-level pseudo-bulk profiles for one dataset directory.
    Cells are grouped by (dataset_id, donor_id).

    Returns:
        dict: {group_key: pseudo_bulk_tensor}, or an empty dict
    """
    h5ad_files = sorted(list(dataset_path.glob("*.h5ad")))
    if not h5ad_files:
        return {}

    # Pre-compile regex for feature_id sanitization (remove suffixes like .1, -1)
    # Note: var_names in census data are feature_id (Ensembl ID), vocab also uses feature_id
    gene_regex = re.compile(r"[\.-]\d+$")

    # Accumulate per-donor sums and cell counts.
    # {donor_id: (sum_array, count)}
    donor_accumulators = {}

    for h5_path in h5ad_files:
        try:
            # 1. Load Data
            adata = sc.read_h5ad(h5_path)

            # 2. Check for donor_id column
            if "donor_id" not in adata.obs.columns:
                print(f"Warning: 'donor_id' not found in {h5_path}, skipping...")
                del adata
                gc.collect()
                continue

            # Get unique donor_ids in this file
            donor_ids: npt.NDArray[np.str_] = np.asarray(adata.obs["donor_id"].astype(str).fillna("unknown").values)

            # 3. Gene Mapping (Optimized)
            var_names = adata.var_names
            current_gene_indices: list[int] = []
            valid_gene_pos: list[int] = []

            for i, g in enumerate(var_names):
                sanitized_name = gene_regex.sub("", g)
                if sanitized_name in vocab:
                    current_gene_indices.append(vocab[sanitized_name])
                    valid_gene_pos.append(i)

            if not valid_gene_pos:
                del adata
                gc.collect()
                continue

            X_raw = adata.X
            if X_raw is None:
                del adata
                gc.collect()
                continue

            # Filter X to only include valid genes
            valid_gene_idx = np.array(valid_gene_pos, dtype=np.intp)
            if sparse.issparse(X_raw):
                X_filtered = sparse.csr_matrix(X_raw[:, valid_gene_idx])
            else:
                X_filtered = np.asarray(X_raw)[:, valid_gene_idx]

            # 4. Normalization
            tmp_adata = sc.AnnData(X=X_filtered)
            sc.pp.normalize_total(tmp_adata, target_sum=target_sum)
            X_norm_raw = tmp_adata.X
            if X_norm_raw is None:
                del adata, tmp_adata
                gc.collect()
                continue

            # Ensure we have a workable type (sparse or ndarray)
            X_norm: sparse.csr_matrix | npt.NDArray[np.float64]
            if sparse.issparse(X_norm_raw):
                X_norm = sparse.csr_matrix(X_norm_raw)
            else:
                X_norm = np.asarray(X_norm_raw)

            # 5. Group by donor_id and accumulate
            unique_donors: npt.NDArray[np.str_] = np.unique(donor_ids)
            for donor_id in unique_donors:
                donor_mask: npt.NDArray[np.bool_] = donor_ids == donor_id
                donor_X = X_norm[donor_mask]

                n_donor_cells: int = donor_X.shape[0]
                if n_donor_cells == 0:
                    continue

                # Initialize accumulator for this donor if needed
                if donor_id not in donor_accumulators:
                    donor_accumulators[donor_id] = (
                        np.zeros(vocab_size, dtype=np.float64),
                        0
                    )

                # Sum along cells for this donor
                batch_sum_raw = donor_X.sum(axis=0)

                # Handle different types: sparse matrix, numpy matrix, numpy array
                batch_sum: npt.NDArray[np.float64]
                if sparse.issparse(batch_sum_raw):
                    batch_sum = np.asarray(batch_sum_raw).flatten()
                elif isinstance(batch_sum_raw, np.matrix):
                    batch_sum = np.asarray(batch_sum_raw).flatten()
                elif isinstance(batch_sum_raw, np.ndarray):
                    batch_sum = batch_sum_raw.flatten()
                else:
                    batch_sum = np.asarray(batch_sum_raw).flatten()

                # Ensure batch_sum is 1D and has correct length
                if len(batch_sum) != len(current_gene_indices):
                    print(f"Error: batch_sum length {len(batch_sum)} != current_gene_indices length {len(current_gene_indices)}", file=sys.stderr, flush=True)
                    continue

                # Update accumulator
                donor_sum, donor_count = donor_accumulators[donor_id]
                donor_sum[current_gene_indices] += batch_sum
                donor_count += n_donor_cells
                donor_accumulators[donor_id] = (donor_sum, donor_count)

            # Cleanup
            del adata, tmp_adata, X_filtered, X_norm
            gc.collect()

        except Exception as e:
            import traceback
            print(f"Error processing {h5_path}: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            gc.collect()
            continue

    # 6. Calculate pseudo-bulk for each donor
    result = {}
    for donor_id, (donor_sum, donor_count) in donor_accumulators.items():
        if donor_count == 0:
            continue

        mean_expr = donor_sum / donor_count
        pseudo_bulk = np.log1p(mean_expr)

        # Key format: "dataset_id_donor_id"
        key = f"{dataset_id}_{donor_id}"
        result[key] = torch.from_numpy(pseudo_bulk).float()

    return result

def process_single_dataset(args_tuple: tuple[Path, str, dict[str, int], int, float]) -> dict[str, torch.Tensor]:
    """
    Wrapper function for parallel processing.
    Args:
        args_tuple: (dataset_dir, dataset_id, vocab, vocab_size, target_sum)
    Returns:
        dict: {dataset_id_donor_id: pseudo_bulk_tensor} or empty dict on error
    """
    dataset_dir, dataset_id, vocab, vocab_size, target_sum = args_tuple
    try:
        donor_pseudo_bulks = calculate_donor_pseudo_bulk(dataset_dir, dataset_id, vocab, vocab_size, target_sum)
        return donor_pseudo_bulks
    except Exception as e:
        import traceback
        print(f"Error processing {dataset_id}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return {}

def main():
    parser = argparse.ArgumentParser(description="Calculate donor-level pseudo-bulk profiles")
    parser.add_argument("--data_root", type=str, required=True, help="Path to data root (containing census_version)")
    parser.add_argument("--census_version", type=str, required=True, help="Census version date")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to gene vocab JSON")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: data_root/census_version)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--target_sum", type=float, default=10000.0, help="Target sum for normalization")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    by_dataset_path = data_root / args.census_version / "by_dataset"

    if not by_dataset_path.exists():
        print(f"Error: Data directory not found: {by_dataset_path}")
        sys.exit(1)

    print(f"Loading vocab from {args.vocab_path}...")
    vocab = load_gene_vocab(args.vocab_path)
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    output_dir = Path(args.output_dir) if args.output_dir else data_root / args.census_version
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pseudo_bulk_dict.pt"

    # Check for existing
    results = {}
    if output_path.exists():
        print(f"Found existing pseudo-bulk file at {output_path}. Loading...")
        try:
            results = torch.load(output_path)
            print(f"Loaded {len(results)} existing profiles.")
        except Exception as e:
            print(f"Failed to load existing file, starting fresh: {e}")

    dataset_dirs = sorted([d for d in by_dataset_path.iterdir() if d.is_dir()])
    total_datasets = len(dataset_dirs)
    print(f"Found {total_datasets} datasets to process.")

    # Filter out already processed datasets
    # Check if any donor from this dataset is already processed
    datasets_to_process = []
    skipped_count = 0
    for dataset_dir in dataset_dirs:
        dataset_id = dataset_dir.name
        # Check if any key starting with dataset_id exists
        already_processed = any(k.startswith(f"{dataset_id}_") for k in results.keys())
        if already_processed:
            skipped_count += 1
        else:
            datasets_to_process.append((dataset_dir, dataset_id))

    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed datasets.")

    if not datasets_to_process:
        print("All datasets already processed!")
        return

    # Prepare arguments for parallel processing
    num_workers = args.num_workers if args.num_workers else os.cpu_count()
    print(f"Using {num_workers} parallel workers.")

    process_args = [
        (dataset_dir, dataset_id, vocab, vocab_size, args.target_sum)
        for dataset_dir, dataset_id in datasets_to_process
    ]

    processed_count = 0
    total_to_process = len(datasets_to_process)

    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_dataset = {
            executor.submit(process_single_dataset, work_args): work_args[1]  # work_args[1] is dataset_id
            for work_args in process_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_dataset):
            dataset_id = future_to_dataset[future]
            try:
                donor_pseudo_bulks = future.result()

                if donor_pseudo_bulks:
                    # Update results with all donor pseudo-bulks
                    results.update(donor_pseudo_bulks)
                    processed_count += 1

                    # Periodic save (every 10 datasets) to prevent data loss
                    if processed_count % 10 == 0:
                        print(f"[{processed_count}/{total_to_process}] Saving intermediate results...")
                        torch.save(results, output_path)
                else:
                    print(f"Warning: No valid data found for {dataset_id}")

            except Exception as e:
                import traceback
                print(f"Failed to get result for {dataset_id}: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                # Check if it's a process termination issue
                if "terminated abruptly" in str(e).lower():
                    print(f"Process for {dataset_id} was terminated. This might indicate:", file=sys.stderr, flush=True)
                    print("  1. An unhandled exception in the worker process", file=sys.stderr, flush=True)
                    print("  2. Memory issues (less likely with 2 workers)", file=sys.stderr, flush=True)
                    print("  3. Data corruption or missing files", file=sys.stderr, flush=True)

    # Final Save
    print(f"Saving final results to {output_path}...")
    torch.save(results, output_path)
    print(f"Done! Processed {processed_count} datasets (skipped {skipped_count}).")

if __name__ == "__main__":
    main()
