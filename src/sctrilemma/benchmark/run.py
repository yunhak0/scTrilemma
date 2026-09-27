"""Zero-shot benchmark of scTrilemma on held-out CELLxGENE Census datasets.

Usage: python -m sctrilemma.benchmark.run --help
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("CPU_THREADS", "4"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("CPU_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("CPU_THREADS", "4"))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time  # noqa: I001
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import scipy.sparse as sp

from sctrilemma.benchmark.data_utils import (  # noqa: E402, I001
    DEFAULT_BASELINE_PATH as DATA_UTILS_DEFAULT_BASELINE_PATH,
    DEFAULT_TARGET_PATH as DATA_UTILS_DEFAULT_TARGET_PATH,
    create_dataset_manifest,
    discover_new_datasets,
    get_dataset_file_metadata,
    load_dataset,
)
from sctrilemma.benchmark.metrics import (  # noqa: E402
    BenchmarkResults,
    compute_reconstruction_metrics,
    flatten_benchmark_results,
    run_benchmark,
    summarize_results_csv,
)
from sctrilemma.benchmark.models import (  # noqa: E402
    MODEL_REGISTRY,
    BaseModelInference,
    get_model,
)
from sctrilemma.benchmark.profiling import (  # noqa: E402
    BenchmarkProfiler,
    PairStatus,
    cleanup_memory,
    peak_gpu_mem_gb,
    reset_peak_gpu_mem,
    write_completion_csv,
    write_profile_csv,
)

DEFAULT_BASELINE_PATH = Path(DATA_UTILS_DEFAULT_BASELINE_PATH)
DEFAULT_TARGET_PATH = Path(DATA_UTILS_DEFAULT_TARGET_PATH)
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark")

BATCH_KEY = "donor_id"
LABEL_KEY = "cell_type_ontology_term_id"


def write_results_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Append result rows to CSV with file locking."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(output_path, "a", encoding="utf-8", newline="") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass

        handle.seek(0, os.SEEK_END)
        is_empty = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if is_empty:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass


def _reason_code_from_exception(exc: Exception) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "oom" in message:
        return "oom"
    if "timeout" in message:
        return "timeout"
    return "unknown_error"


def _batch_key_for_adata(adata: ad.AnnData) -> str:
    if BATCH_KEY in adata.obs.columns:
        return BATCH_KEY
    if "dataset_id" in adata.obs.columns:
        return "dataset_id"
    adata.obs["dataset_id"] = str(adata.uns.get("dataset_id", "unknown"))
    return "dataset_id"


def _pair_status_base(
    dataset_id: str,
    model_name: str,
    adata: Optional[ad.AnnData],
    dataset_meta: Dict[str, int],
) -> PairStatus:
    batch_key = _batch_key_for_adata(adata) if adata is not None else BATCH_KEY
    n_batches = int(adata.obs[batch_key].nunique()) if adata is not None else 0
    n_labels = (
        int(adata.obs[LABEL_KEY].nunique()) if adata is not None and LABEL_KEY in adata.obs.columns else 0
    )
    return PairStatus(
        dataset_id=dataset_id,
        model_name=model_name,
        status="failed",
        reason="",
        n_cells=int(adata.n_obs) if adata is not None else 0,
        n_genes=int(adata.n_vars) if adata is not None else 0,
        n_batches=n_batches,
        n_labels=n_labels,
        n_shards=dataset_meta.get("n_shards", 0),
        dataset_total_size_bytes=dataset_meta.get("dataset_total_size_bytes", 0),
    )


def _align_raw_to_reconstructed(
    adata: ad.AnnData,
    recon_gene_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice raw X to match reconstructed gene subset. Returns (raw_aligned, keep_mask)."""
    adata_gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}

    keep_recon: List[int] = []
    raw_col: List[int] = []
    for i, gene in enumerate(recon_gene_names):
        raw_i = adata_gene_to_idx.get(gene)
        if raw_i is not None:
            keep_recon.append(i)
            raw_col.append(raw_i)

    raw_X = adata.X
    if sp.issparse(raw_X):
        raw_aligned = np.asarray(raw_X[:, raw_col].toarray(), dtype=np.float32)
    else:
        raw_aligned = np.asarray(raw_X[:, raw_col], dtype=np.float32)

    keep_mask = np.zeros(len(recon_gene_names), dtype=bool)
    keep_mask[keep_recon] = True
    return raw_aligned, keep_mask


def _sample_cells_for_analysis(
    adata: ad.AnnData,
    *,
    max_cells: Optional[int],
    seed: int,
    dataset_id: str,
) -> ad.AnnData:
    """Return a deterministic cell subsample for compute-controlled analyses."""
    if max_cells is None or max_cells <= 0 or adata.n_obs <= max_cells:
        return adata
    digest = hashlib.sha1(dataset_id.encode("utf-8")).hexdigest()
    dataset_seed = (seed + int(digest[:8], 16)) % (2**32)
    rng = np.random.default_rng(dataset_seed)
    indices = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    sampled = adata[indices].copy()
    if "donor_id" in sampled.obs.columns:
        sampled.obs["donor_id"] = sampled.obs["donor_id"].astype("string").fillna("unknown").astype(str)
    return sampled


def run_single_benchmark(
    dataset_id: str,
    model_name: str,
    target_path: Path,
    output_dir: Path,
    *,
    adata: Optional[ad.AnnData] = None,
    dataset_load_seconds: float = 0.0,
    dataset_meta: Optional[Dict[str, int]] = None,
    k: int = 50,
    use_faiss: bool = True,
    batch_size: Optional[int] = None,
    n_jobs: int = -1,
    model: Optional[BaseModelInference] = None,
    model_kwargs: Optional[Dict] = None,
    enable_profiling: bool = False,
    disable_reconstruction: bool = False,
    enable_bio_graph_connectivity: bool = False,
    enable_scgraph: bool = True,
    oversize_serialized: bool = False,
) -> Tuple[List[Dict[str, str]], PairStatus]:
    """Run benchmark for a single dataset/model pair and persist outputs."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_id}")
    print(f"Model: {model_name}")
    print(f"{'=' * 60}")

    dataset_meta = dataset_meta or get_dataset_file_metadata(dataset_id, str(target_path))
    profiler = BenchmarkProfiler(dataset_id=dataset_id, model_name=model_name)
    if dataset_load_seconds > 0:
        profiler.record_stage("load_dataset", dataset_load_seconds)

    reset_peak_gpu_mem()
    start_time = time.perf_counter()
    loaded_locally = False
    rows: List[Dict[str, str]] = []
    embedding_key: Optional[str] = None
    embeddings = None
    reconstructed = None

    try:
        if adata is None:
            loaded_locally = True
            with profiler.time_stage("load_dataset"):
                adata = load_dataset(dataset_id, str(target_path))

        assert adata is not None
        status = _pair_status_base(dataset_id, model_name, adata, dataset_meta)
        batch_key = _batch_key_for_adata(adata)

        print(
            f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}, "
            f"Batches ({batch_key}): {status.n_batches}, Labels: {status.n_labels}"
        )

        if status.n_labels < 2:
            status.status = "skipped"
            status.reason = "fewer than 2 cell types"
            return rows, status

        if model is None:
            with profiler.time_stage("load_model"):
                model = get_model(model_name, **(model_kwargs or {}))
                model.load_model()
        else:
            print(f"Using pre-loaded {model_name} model")

        # Generation-only models skip embeddings, classification, and bio metrics.
        is_generative = model.is_generative()

        if is_generative:
            print("  Generative model — skipping embeddings, classification, bio metrics")
            results = BenchmarkResults()
            status.bio_skipped = True
            status.bio_skip_reason = "generative_model"
        else:
            with profiler.time_stage("compute_embeddings"):
                if batch_size is not None:
                    embeddings = model.get_embeddings(adata, batch_size=batch_size)
                else:
                    embeddings = model.get_embeddings(adata)

            embedding_key = model.embedding_key()
            adata.obsm[embedding_key] = embeddings
            print(f"  Embedding shape: {embeddings.shape}")

            with profiler.time_stage("run_metrics"):
                results = run_benchmark(
                    adata=adata,
                    embedding_key=embedding_key,
                    batch_key=batch_key,
                    label_key=LABEL_KEY,
                    k=k,
                    use_faiss=use_faiss,
                    n_jobs=n_jobs,
                    enable_bio_graph_connectivity=enable_bio_graph_connectivity,
                    enable_scgraph=enable_scgraph,
                )

            if results.scib_skip_reason is not None:
                status.bio_skipped = True
                status.bio_skip_reason = results.scib_skip_reason

        reconstruction_enabled = model.can_reconstruct() and not disable_reconstruction
        status.reconstruction_enabled = reconstruction_enabled
        if not reconstruction_enabled:
            status.recon_skipped = True
            status.recon_skip_reason = (
                "disabled_by_flag" if disable_reconstruction and model.can_reconstruct() else "unsupported_model"
            )
        else:
            try:
                with profiler.time_stage("compute_reconstruction"):
                    recon_kwargs = {}
                    if batch_size is not None:
                        recon_kwargs["batch_size"] = batch_size
                    reconstructed, recon_gene_names = model.reconstruct(adata, **recon_kwargs)
                    raw_aligned, keep_mask = _align_raw_to_reconstructed(adata, recon_gene_names)
                    if not keep_mask.all():
                        reconstructed = reconstructed[:, keep_mask]
                    reconstruction_metrics, gene_skip_reason = compute_reconstruction_metrics(
                        raw_aligned,
                        reconstructed,
                    )
                    results.reconstruction = reconstruction_metrics
                    results.gene_skip_reason = gene_skip_reason
                if gene_skip_reason is not None:
                    status.gene_skipped = True
                    status.gene_skip_reason = gene_skip_reason
            except Exception as exc:
                status.recon_skipped = True
                status.recon_skip_reason = _reason_code_from_exception(exc)
                print(f"  Warning: reconstruction skipped: {exc}")

        rows = flatten_benchmark_results(
            results,
            {
                "dataset": dataset_id,
                "model": model_name,
                "embedding_key": embedding_key,
            },
        )
        write_results_csv(rows, output_dir / "results.csv")
        status.status = "completed"
        status.reason = "oversize_serialized" if oversize_serialized else ""

        print("Results:")
        if results.classification is not None:
            print(f"  Accuracy: {results.classification.accuracy:.4f}")
            print(f"  F1 (weighted): {results.classification.f1_weighted:.4f}")
            print(f"  F1 (macro): {results.classification.f1_macro:.4f}")
        else:
            print("  (generative model — no classification metrics)")
        if results.reconstruction is not None:
            print(f"  W2: {results.reconstruction.wasserstein2:.4f}")
            print(f"  FD: {results.reconstruction.frechet_distance:.4f}")

        return rows, status

    except NotImplementedError as exc:
        status = _pair_status_base(dataset_id, model_name, adata, dataset_meta)
        status.status = "skipped"
        status.reason = str(exc)
        return rows, status
    except Exception as exc:
        status = _pair_status_base(dataset_id, model_name, adata, dataset_meta)
        status.status = "failed"
        status.reason = str(exc)
        import traceback

        traceback.print_exc()
        return rows, status
    finally:
        total_seconds = time.perf_counter() - start_time
        profiler.record_total(total_seconds)

        try:
            completion_status = locals().get("status")
            if isinstance(completion_status, PairStatus):
                completion_status.load_dataset_seconds = profiler.stage_seconds("load_dataset")
                completion_status.load_model_seconds = profiler.stage_seconds("load_model")
                completion_status.embedding_seconds = profiler.stage_seconds("compute_embeddings")
                completion_status.reconstruction_seconds = profiler.stage_seconds("compute_reconstruction")
                completion_status.metrics_seconds = profiler.stage_seconds("run_metrics")
                completion_status.total_seconds = profiler.stage_seconds("total")
                completion_status.peak_gpu_mem_gb = peak_gpu_mem_gb()
                write_completion_csv([completion_status.to_row()], output_dir / "completion_report.csv")
                if enable_profiling:
                    write_profile_csv(profiler.to_rows(), output_dir / "profile_report.csv")
        finally:
            if embedding_key is not None and adata is not None and embedding_key in adata.obsm:
                del adata.obsm[embedding_key]
            if embeddings is not None:
                del embeddings
            if reconstructed is not None:
                del reconstructed
            if model is not None:
                model.clear_runtime_cache()
            if loaded_locally and adata is not None:
                del adata
            cleanup_memory()


def preload_models(
    model_names: List[str],
    model_kwargs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, BaseModelInference]:
    """Pre-load all requested models once."""
    model_kwargs = model_kwargs or {}
    loaded_models: Dict[str, BaseModelInference] = {}
    for model_name in model_names:
        print(f"Pre-loading {model_name}...")
        try:
            model = get_model(model_name, **model_kwargs.get(model_name, {}))
            model.load_model()
            loaded_models[model_name] = model
            print(f"  {model_name} loaded successfully")
        except Exception as exc:
            print(f"  Warning: Failed to load {model_name}: {exc}")
    return loaded_models


AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def _default_cpu_threads() -> int:
    for key in ("CPU_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                continue
    return 4


def _print_summary(summary_df) -> None:
    if summary_df is None:
        print("No results to summarize")
        return

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    for model_name in summary_df["model"].unique():
        model_df = summary_df[summary_df["model"] == model_name]
        print(f"\n{model_name}:")
        for _, row in model_df.iterrows():
            print(
                f"  {row['metric']}: {row['value_mean']} +/- "
                f"{row['value_std']} (n={row['n_datasets']})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot benchmark of scTrilemma on held-out datasets"
    )
    parser.add_argument("--models", nargs="+", default=["sctrilemma"], choices=AVAILABLE_MODELS)
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument(
        "--dataset-ids",
        type=str,
        nargs="+",
        default=None,
        help="Multiple specific dataset IDs to run (space-separated). "
             "Takes precedence over --dataset-id and --datasets.",
    )
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--target-path", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=_default_cpu_threads())
    parser.add_argument(
        "--max-cells-per-dataset",
        type=int,
        default=None,
        help="Optional deterministic cell subsample for compute-controlled analysis runs.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for --max-cells-per-dataset sampling.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-cache-models", action="store_true")
    parser.add_argument("--disable-reconstruction", action="store_true")
    parser.add_argument("--enable-bio-graph-connectivity", action="store_true")
    parser.add_argument("--disable-scgraph", action="store_true", help="Skip scGraph metric")
    # scTrilemma-specific arguments
    parser.add_argument("--sctrilemma-checkpoint", type=str, default=None, help="scTrilemma checkpoint path")
    parser.add_argument("--sctrilemma-gene-vocab", type=str, default=None, help="scTrilemma gene vocab JSON")
    parser.add_argument("--sctrilemma-pseudo-bulk", type=str, default=None, help="scTrilemma pseudo-bulk dict")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-model kwargs (scTrilemma needs checkpoint/vocab/pb paths)
    model_kwargs: Dict[str, Dict] = {}
    if "sctrilemma" in args.models:
        if args.sctrilemma_checkpoint is None:
            parser.error("--sctrilemma-checkpoint is required when using the sctrilemma model")
        if args.sctrilemma_gene_vocab is None:
            parser.error("--sctrilemma-gene-vocab is required when using the sctrilemma model")
        model_kwargs["sctrilemma"] = {
            "checkpoint_path": args.sctrilemma_checkpoint,
            "gene_vocab_path": args.sctrilemma_gene_vocab,
            "pseudo_bulk_path": args.sctrilemma_pseudo_bulk,
        }
    if args.dataset_ids:
        dataset_ids = list(args.dataset_ids)
        print(f"Evaluating {len(dataset_ids)} specified datasets")
    elif args.dataset_id:
        dataset_ids = [args.dataset_id]
        print(f"Evaluating single dataset: {args.dataset_id}")
    else:
        print("Discovering new datasets...")
        dataset_ids = discover_new_datasets(str(args.baseline_path), str(args.target_path))
        print(f"Found {len(dataset_ids)} new datasets")
        if args.datasets != "all":
            dataset_ids = dataset_ids[: int(args.datasets)]

    manifest_path = args.output_dir / "dataset_manifest.csv"
    if not manifest_path.exists():
        create_dataset_manifest(dataset_ids, str(args.target_path), str(manifest_path))

    loaded_models: Dict[str, BaseModelInference] = {}
    if not args.no_cache_models:
        loaded_models = preload_models(args.models, model_kwargs=model_kwargs)

    completed = 0
    skipped = 0
    failed = 0

    print(
        f"\nStarting benchmark: {len(dataset_ids)} datasets x {len(args.models)} "
        f"models = {len(dataset_ids) * len(args.models)} pairs"
    )

    for dataset_id in dataset_ids:
        dataset_meta = get_dataset_file_metadata(dataset_id, str(args.target_path))
        adata: Optional[ad.AnnData] = None
        dataset_load_seconds = 0.0

        try:
            load_start = time.perf_counter()
            adata = load_dataset(dataset_id, str(args.target_path))
            dataset_load_seconds = time.perf_counter() - load_start
            adata = _sample_cells_for_analysis(
                adata,
                max_cells=args.max_cells_per_dataset,
                seed=args.sample_seed,
                dataset_id=dataset_id,
            )
            if args.max_cells_per_dataset is not None:
                print(
                    f"Using {adata.n_obs:,} cells for {dataset_id} "
                    f"(max_cells_per_dataset={args.max_cells_per_dataset})"
                )
        except Exception as exc:
            for model_name in args.models:
                status = PairStatus(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    status="failed",
                    reason=str(exc),
                    n_shards=dataset_meta.get("n_shards", 0),
                    dataset_total_size_bytes=dataset_meta.get("dataset_total_size_bytes", 0),
                )
                write_completion_csv([status.to_row()], args.output_dir / "completion_report.csv")
                failed += 1
            continue

        try:
            for model_name in args.models:
                if not args.no_cache_models and model_name not in loaded_models:
                    status = _pair_status_base(dataset_id, model_name, adata, dataset_meta)
                    status.status = "failed"
                    status.reason = "model preload failed"
                    write_completion_csv([status.to_row()], args.output_dir / "completion_report.csv")
                    failed += 1
                    continue

                _rows, status = run_single_benchmark(
                    dataset_id=dataset_id,
                    model_name=model_name,
                    target_path=args.target_path,
                    output_dir=args.output_dir,
                    adata=adata,
                    dataset_load_seconds=dataset_load_seconds,
                    dataset_meta=dataset_meta,
                    k=args.k,
                    use_faiss=not args.no_faiss,
                    batch_size=args.batch_size,
                    n_jobs=args.n_jobs,
                    model=loaded_models.get(model_name),
                    model_kwargs=model_kwargs.get(model_name),
                    enable_profiling=args.profile,
                    disable_reconstruction=args.disable_reconstruction,
                    enable_bio_graph_connectivity=args.enable_bio_graph_connectivity,
                    enable_scgraph=not args.disable_scgraph,
                )
                if status.status == "completed":
                    completed += 1
                elif status.status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                print(
                    f"Progress: {completed + skipped + failed}/"
                    f"{len(dataset_ids) * len(args.models)} "
                    f"({completed} completed, {skipped} skipped, {failed} failed)"
                )
        finally:
            if adata is not None:
                del adata
            cleanup_memory()

    summary_df = summarize_results_csv(
        args.output_dir / "results.csv",
        args.output_dir / "results_summary.csv",
    )
    _print_summary(summary_df)

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results: {args.output_dir / 'results.csv'}")
    print(f"Completion: {args.output_dir / 'completion_report.csv'}")
    print(f"Summary: {args.output_dir / 'results_summary.csv'}")
    if args.profile:
        print(f"Profile: {args.output_dir / 'profile_report.csv'}")


if __name__ == "__main__":
    main()
