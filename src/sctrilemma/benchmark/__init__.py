"""Zero-shot benchmarking utilities for scTrilemma."""

from .data_utils import (
    create_dataset_manifest,
    discover_new_datasets,
    get_dataset_file_metadata,
    load_dataset,
    normalize_counts,
)
from .metrics import (
    BenchmarkResults,
    ClassificationMetrics,
    ReconstructionMetrics,
    compute_knn_classification,
    compute_reconstruction_metrics,
    flatten_benchmark_results,
    run_benchmark,
    summarize_results_csv,
)
from .models import (
    MODEL_REGISTRY,
    BaseModelInference,
    ScTrilemmaInference,
    get_model,
)
from .profiling import (
    BenchmarkProfiler,
    PairStatus,
    TimingResult,
    cleanup_memory,
    write_completion_csv,
    write_profile_csv,
)

__all__ = [
    "create_dataset_manifest",
    "discover_new_datasets",
    "get_dataset_file_metadata",
    "load_dataset",
    "normalize_counts",
    "BenchmarkResults",
    "ClassificationMetrics",
    "ReconstructionMetrics",
    "compute_knn_classification",
    "compute_reconstruction_metrics",
    "flatten_benchmark_results",
    "run_benchmark",
    "summarize_results_csv",
    "MODEL_REGISTRY",
    "ScTrilemmaInference",
    "BaseModelInference",
    "get_model",
    "BenchmarkProfiler",
    "PairStatus",
    "TimingResult",
    "cleanup_memory",
    "write_completion_csv",
    "write_profile_csv",
]
