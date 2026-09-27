"""Profiling and status utilities for zero-shot benchmarking."""

from __future__ import annotations

import csv
import gc
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import torch

PROFILE_FIELDS = ["dataset_id", "model_name", "stage", "elapsed_seconds"]
COMPLETION_FIELDS = [
    "dataset_id",
    "model_name",
    "status",
    "reason",
    "n_cells",
    "n_genes",
    "n_batches",
    "n_labels",
    "n_shards",
    "dataset_total_size_bytes",
    "load_dataset_seconds",
    "load_model_seconds",
    "embedding_seconds",
    "reconstruction_seconds",
    "metrics_seconds",
    "total_seconds",
    "peak_gpu_mem_gb",
    "bio_skipped",
    "bio_skip_reason",
    "recon_skipped",
    "recon_skip_reason",
    "gene_skipped",
    "gene_skip_reason",
    "reconstruction_enabled",
]


@dataclass
class TimingResult:
    """A single profiled stage result."""

    stage: str
    elapsed_seconds: float


@dataclass
class PairStatus:
    """Pair-level completion report row."""

    dataset_id: str
    model_name: str
    status: str
    reason: str
    n_cells: int = 0
    n_genes: int = 0
    n_batches: int = 0
    n_labels: int = 0
    n_shards: int = 0
    dataset_total_size_bytes: int = 0
    load_dataset_seconds: float = 0.0
    load_model_seconds: float = 0.0
    embedding_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    metrics_seconds: float = 0.0
    total_seconds: float = 0.0
    peak_gpu_mem_gb: float = 0.0
    bio_skipped: bool = False
    bio_skip_reason: str = ""
    recon_skipped: bool = False
    recon_skip_reason: str = ""
    gene_skipped: bool = False
    gene_skip_reason: str = ""
    reconstruction_enabled: bool = False

    def to_row(self) -> Dict[str, str]:
        """Serialize to a CSV-safe row dictionary."""
        row = asdict(self)
        for key in ("bio_skipped", "recon_skipped", "gene_skipped", "reconstruction_enabled"):
            row[key] = "true" if row[key] else "false"
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.6f}"
            else:
                row[key] = str(value)
        return row


class BenchmarkProfiler:
    """Collect stage timings for a dataset/model pair."""

    def __init__(self, dataset_id: str, model_name: str) -> None:
        self.dataset_id = dataset_id
        self.model_name = model_name
        self._timings: List[TimingResult] = []

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        """Record elapsed wall time for a stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._timings.append(TimingResult(stage=stage, elapsed_seconds=elapsed))

    def record_total(self, elapsed_seconds: float) -> None:
        """Record total elapsed wall time."""
        self._timings = [t for t in self._timings if t.stage != "total"]
        self._timings.append(TimingResult(stage="total", elapsed_seconds=elapsed_seconds))

    def record_stage(self, stage: str, elapsed_seconds: float) -> None:
        """Record an externally measured stage duration."""
        self._timings.append(TimingResult(stage=stage, elapsed_seconds=elapsed_seconds))

    def stage_seconds(self, stage: str) -> float:
        """Return the summed elapsed seconds for a stage."""
        return sum(t.elapsed_seconds for t in self._timings if t.stage == stage)

    def print_summary(self) -> None:
        """Print a compact profiling summary."""
        print("Profiling summary:")
        for timing in self._timings:
            print(f"  {timing.stage}: {timing.elapsed_seconds:.2f}s")

    def to_rows(self) -> List[Dict[str, str]]:
        """Convert all timing results to CSV rows."""
        rows: List[Dict[str, str]] = []
        for timing in self._timings:
            rows.append(
                {
                    "dataset_id": self.dataset_id,
                    "model_name": self.model_name,
                    "stage": timing.stage,
                    "elapsed_seconds": f"{timing.elapsed_seconds:.6f}",
                }
            )
        return rows


def cleanup_memory() -> None:
    """Trigger Python and CUDA memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def peak_gpu_mem_gb() -> float:
    """Return peak allocated GPU memory in GB for the current device."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        return float(torch.cuda.max_memory_allocated() / 1e9)
    except Exception:
        return 0.0


def reset_peak_gpu_mem() -> None:
    """Reset peak GPU memory stats if available."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _append_rows_csv(
    rows: List[Dict[str, str]],
    output_path: Path,
    fieldnames: List[str],
) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def write_profile_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Append profiling rows to CSV."""
    _append_rows_csv(rows, output_path, PROFILE_FIELDS)


def write_completion_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Append completion report rows to CSV."""
    _append_rows_csv(rows, output_path, COMPLETION_FIELDS)
