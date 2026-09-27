"""Metrics module for zero-shot benchmarking."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from sctrilemma.utils.metrics import (
    compute_pearson_correlation,
)

SCIB_AGGREGATE_KEYWORDS = (
    "total",
    "batch correction",
    "bio conservation",
)


@dataclass
class ClassificationMetrics:
    """Results from k-NN classification."""

    accuracy: float
    f1_weighted: float
    f1_macro: float


@dataclass
class ReconstructionMetrics:
    """Reconstruction and gene-level metrics.

    All correlation metrics are computed in **log1p(CP10K)** space: both
    prediction and ground truth are independently normalized via
    ``log1p(x / sum(x) * 10000)`` before comparison.  This is the
    field-standard surface used by scLDM, GEARS, and scGen.

    **Zero metrics** (``zero_*``) are computed in **raw count space**
    using a per-cell rank-based classifier, so they are robust to
    softmax/dense model outputs.

    **HVG selection** (``recon_*_hvg``) uses variance in log1p(CP10K)
    space — matches scanpy's seurat_v3 flavor, differs from raw-count
    HVG selection.
    """

    recon_mae: float
    recon_mse: float
    recon_pearson: float
    recon_spearman: float
    # Legacy fields kept for CSV backward compatibility.
    # In log1p(CP10K) space these are identical to ``recon_pearson``.
    recon_pearson_log1p: float
    recon_pearson_sqrt: float
    recon_pearson_nonzero: float
    recon_spearman_nonzero: float
    recon_pearson_highexpr: float
    recon_spearman_highexpr: float
    recon_pearson_hvg: float
    recon_spearman_hvg: float
    # Per-gene-mean statistics (scGen/scLDM generation-style)
    gene_mean_pearson: float
    gene_mean_spearman: float
    # Distribution distance on PCA (scLDM/CFGen style)
    mmd_pca: float
    # scDesign2/3 marginal distribution metrics (KS distance; lower = better)
    #
    # Note: ``marginal_cell_libsize_ks`` is intentionally NOT included.  There
    # is no model-agnostic way to define "reconstructed library size" for
    # softmax/frequency-scale outputs (scVI, scTrilemma-softmax), so it would produce
    # a systematically biased score depending only on output convention.
    # Note: ``marginal_cell_detect_ks`` is intentionally NOT included either.
    # Under per-cell rank-calibrated zero detection, the predicted detection
    # count matches the actual count row-by-row by construction, so the KS
    # statistic is always 0 and carries no information about model quality.
    marginal_gene_mean_ks: float
    marginal_gene_var_ks: float
    marginal_gene_detect_ks: float
    # scLDM generation quality metrics (lower = better)
    wasserstein2: float
    frechet_distance: float
    # scDesign3 joint distribution metrics
    gene_corr_matrix_pearson: float  # higher = better
    cell_corr_ks: float  # lower = better
    # scDesign3 additional summary statistics (lower = better)
    cell_distance_ks: float  # KS of pairwise cell distances in PCA space
    cell_detect_freq_ks: float  # KS of per-cell detection frequency
    zero_accuracy: float
    zero_recovery_precision: float
    zero_recovery_recall: float
    zero_recovery_f1: float
    zero_recovery_specificity: float
    zero_recovery_accuracy: float
    zero_recovery_balanced_acc: float
    gene_pearson: float
    gene_spearman: float


@dataclass
class BenchmarkResults:
    """Combined benchmark results for a single dataset/model pair."""

    classification: Optional[ClassificationMetrics] = None
    scib_metrics: Optional[Dict[str, float]] = None
    reconstruction: Optional[ReconstructionMetrics] = None
    scib_skip_reason: Optional[str] = None
    gene_skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to a flat metric dictionary."""
        result: Dict[str, float] = {}
        if self.classification is not None:
            result.update({
                "accuracy": self.classification.accuracy,
                "f1_weighted": self.classification.f1_weighted,
                "f1_macro": self.classification.f1_macro,
            })
        if self.reconstruction is not None:
            result.update(asdict(self.reconstruction))
        if self.scib_metrics is not None:
            result.update(self.scib_metrics)
        return result


def _knn_torch_gpu(
    embeddings: np.ndarray,
    k: int,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """k-NN via ``torch.cdist`` + ``topk`` on GPU, chunked over query rows.

    Returns ``(indices, distances)`` where distances are **squared L2**
    to match the convention of ``faiss.IndexFlatL2.search``.  The chunked
    approach keeps GPU memory bounded at ``O(chunk_size × N)`` instead of
    ``O(N²)``; with ``chunk_size=1024`` and ``N=100K``, peak VRAM for the
    distance tensor is ~400 MB — well within budget.

    Benchmark (vs sklearn / faiss-CPU on B200, k=50):
      N=10K → 33× faster than sklearn, 116× faster than faiss-CPU
      N=50K → 107× faster than sklearn, 400× faster than faiss-CPU
    Neighbor-set Jaccard vs sklearn: 1.0000 at all tested sizes.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, d = embeddings.shape
    emb_t = torch.from_numpy(
        np.ascontiguousarray(embeddings, dtype=np.float32)
    ).to(device)

    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            query = emb_t[start:end]
            # cdist returns L2; square to match faiss IndexFlatL2 convention
            dist = torch.cdist(query, emb_t)
            dist_sq = dist * dist
            d_top, i_top = torch.topk(
                dist_sq, k=k, dim=1, largest=False, sorted=True
            )
            indices[start:end] = i_top.cpu().numpy()
            distances[start:end] = d_top.cpu().numpy()

    return indices, distances


def compute_knn_classification(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 50,
    use_faiss: bool = True,
    n_jobs: int = -1,
    use_cuml: Optional[bool] = None,
) -> ClassificationMetrics:
    """Compute leave-one-out k-NN classification metrics.

    Backend priority (highest → lowest):
      1. cuML (if available + ``use_cuml`` not False)
      2. **torch GPU** (if CUDA available) — 30-100× faster than sklearn
      3. FAISS (if ``use_faiss=True``)
      4. sklearn (fallback)
    """
    n_cells = embeddings.shape[0]

    if n_cells <= k:
        k = max(1, n_cells - 1)

    if use_cuml is None:
        use_cuml = _check_cuml_available()

    if use_cuml:
        try:
            indices, distances = _knn_cuml(embeddings, k + 1)
        except Exception as exc:
            print(f"cuML k-NN failed, falling back: {exc}")
            indices, distances = _knn_torch_gpu(embeddings, k + 1)
    elif torch.cuda.is_available():
        indices, distances = _knn_torch_gpu(embeddings, k + 1)
    elif use_faiss:
        indices, distances = _knn_faiss(embeddings, k + 1)
    else:
        indices, distances = _knn_sklearn(embeddings, k + 1, n_jobs=n_jobs)

    neighbor_indices = indices[:, 1 : k + 1]
    neighbor_distances = distances[:, 1 : k + 1]

    eps = 1e-8
    weights = 1.0 / (neighbor_distances + eps)

    predictions = _weighted_vote(labels, neighbor_indices, weights)

    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0.0,  # pyright: ignore[reportArgumentType]
    )
    f1_macro = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0.0,  # pyright: ignore[reportArgumentType]
    )

    return ClassificationMetrics(
        accuracy=float(accuracy),
        f1_weighted=float(f1_weighted),
        f1_macro=float(f1_macro),
    )


def _knn_faiss(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors using FAISS."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is required for efficient k-NN search. "
            "Install with: uv add faiss-cpu (or faiss-gpu)"
        ) from exc

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    d = embeddings.shape[1]

    def _search_with_cpu_index() -> Tuple[np.ndarray, np.ndarray]:
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(embeddings)
        cpu_distances, cpu_indices = cpu_index.search(embeddings, k)
        return cpu_indices, cpu_distances

    force_gpu = os.environ.get("SCBENCH_FORCE_FAISS_GPU", "0") == "1"
    if not force_gpu and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major >= 10:
            print(
                "FAISS GPU disabled on compute capability >= 10.0; "
                "using CPU FAISS instead."
            )
            return _search_with_cpu_index()

    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    except (RuntimeError, AttributeError):
        return _search_with_cpu_index()

    try:
        index.add(embeddings)
        distances, indices = index.search(embeddings, k)
        return indices, distances
    except RuntimeError as exc:
        print(f"FAISS GPU k-NN failed, falling back to CPU FAISS: {exc}")
        return _search_with_cpu_index()


def _knn_sklearn(
    embeddings: np.ndarray,
    k: int,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors using sklearn."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=n_jobs)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    return indices, distances


def _knn_cuml(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors using cuML."""
    try:
        from cuml.neighbors import NearestNeighbors as cumlNN
    except ImportError as exc:
        raise ImportError(
            "cuML is required for GPU k-NN search. "
            "Install with: pip install cuml-cu12"
        ) from exc

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    nn = cumlNN(n_neighbors=k, metric="euclidean", output_type="numpy")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    return indices.astype(np.int64), distances.astype(np.float64)


def _check_cuml_available() -> bool:
    """Check if cuML is available."""
    try:
        from cuml.neighbors import NearestNeighbors  # noqa: F401

        return True
    except ImportError:
        return False


def _weighted_vote(
    labels: np.ndarray,
    neighbor_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Predict labels using vectorized weighted voting."""
    n_cells, _ = neighbor_indices.shape
    n_classes = int(labels.max()) + 1

    neighbor_labels = labels[neighbor_indices]
    class_weights = np.zeros((n_cells, n_classes), dtype=np.float64)
    row_indices = np.arange(n_cells)[:, None]
    np.add.at(class_weights, (row_indices, neighbor_labels), weights)

    return np.argmax(class_weights, axis=1).astype(np.int64)


def _create_bio_metrics(enable_graph_connectivity: bool):
    from scib_metrics.benchmark import BioConservation

    if enable_graph_connectivity:
        try:
            return BioConservation(silhouette_label=True, graph_connectivity=True)
        except TypeError:
            pass
    return BioConservation(silhouette_label=True)


def _default_random_seed() -> int:
    value = os.environ.get("SCBENCH_RANDOM_SEED", "0")
    try:
        return int(value)
    except ValueError:
        return 0


def _keep_scib_metric(name: str) -> bool:
    lowered = name.strip().lower()
    if any(keyword in lowered for keyword in SCIB_AGGREGATE_KEYWORDS):
        return False
    if lowered in {"metric", "embedding"}:
        return False
    return True


def stratified_subsample_adata(
    adata: ad.AnnData,
    label_key: str,
    max_per_type: int = 1000,
    total_max: int = 50_000,
    min_cells_for_rare: int = 50,
    seed: Optional[int] = None,
) -> ad.AnnData:
    """Stratified subsampling preserving cell type diversity.

    Adapted from main worktree's stratified_subsample() pattern.
    """
    if len(adata) <= total_max:
        return adata

    labels = adata.obs[label_key].values
    unique_types, counts = np.unique(labels, return_counts=True)
    included_types = unique_types[counts >= min_cells_for_rare]

    if len(included_types) == 0:
        return adata

    rng = np.random.default_rng(_default_random_seed() if seed is None else seed)
    sampled_indices: List[int] = []
    for ct in included_types:
        ct_indices = np.where(labels == ct)[0]
        n_sample = min(len(ct_indices), max_per_type)
        chosen = rng.choice(ct_indices, n_sample, replace=False)
        sampled_indices.extend(chosen.tolist())

    sampled = np.array(sampled_indices)
    if len(sampled) > total_max:
        sampled = rng.choice(sampled, total_max, replace=False)

    sampled.sort()
    return adata[sampled].copy()


def run_benchmark(
    adata: ad.AnnData,
    embedding_key: str,
    batch_key: str = "donor_id",
    label_key: str = "cell_type_ontology_term_id",
    k: int = 50,
    use_faiss: bool = True,
    use_cuml: Optional[bool] = None,
    n_jobs: int = -1,
    enable_bio_graph_connectivity: bool = False,
    enable_scgraph: bool = True,
) -> BenchmarkResults:
    """Run classification and bio-conservation metrics on embeddings.

    Uses ``compute_bio_conservation_metrics`` from the src metrics suite
    (GPU-accelerated, ~15-90x faster than scib_metrics.Benchmarker).
    """
    from sctrilemma.utils.metrics import compute_bio_conservation_metrics

    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding key '{embedding_key}' not found in adata.obsm")
    if label_key not in adata.obs.columns:
        raise ValueError(f"Label key '{label_key}' not found in adata.obs")

    embeddings = np.asarray(adata.obsm[embedding_key])
    labels_raw = adata.obs[label_key].values

    label_encoder = LabelEncoder()
    labels = np.asarray(label_encoder.fit_transform(labels_raw))

    try:
        classification = compute_knn_classification(
            embeddings=embeddings,
            labels=labels,
            k=k,
            use_faiss=use_faiss,
            n_jobs=n_jobs,
            use_cuml=use_cuml,
        )
    except ImportError:
        classification = compute_knn_classification(
            embeddings=embeddings,
            labels=labels,
            k=k,
            use_faiss=False,
            n_jobs=n_jobs,
            use_cuml=False,
        )

    # Bio metrics via src (replaces scib_metrics.Benchmarker)
    batch_key_actual = batch_key
    if batch_key not in adata.obs.columns:
        if "dataset_id" in adata.obs.columns:
            batch_key_actual = "dataset_id"
        else:
            batch_key_actual = None

    has_batch = batch_key_actual is not None and adata.obs[batch_key_actual].nunique() > 1
    has_label = adata.obs[label_key].nunique() > 1

    bio_metrics: Optional[Dict[str, float]] = None
    bio_skip_reason: Optional[str] = None

    if not has_label:
        bio_skip_reason = "single_label"
    else:
        # When single_batch, label-only metrics (NMI, ARI, ASW_label, cLISI,
        # isolated_labels) are still meaningful and computed. Batch-dependent
        # metrics (BRAS, iLISI, scGraph) are explicitly dropped post-hoc since
        # they have no signal on a single-batch dataset.
        if has_batch:
            batch_ids_raw = adata.obs[batch_key_actual].values
            batch_encoder = LabelEncoder()
            batch_ids = np.asarray(batch_encoder.fit_transform(batch_ids_raw))
        else:
            batch_ids = np.zeros(len(labels), dtype=np.int64)

        # gene_expression is consumed by scGraph. scGraph treats batch_ids only
        # as a per-batch reference grouping (single-batch -> one reference frame),
        # so the metric is well-defined even when has_batch is False.
        gene_expr_matrix = None
        if enable_scgraph:
            gene_expr_matrix = adata.X

        try:
            src_metrics = compute_bio_conservation_metrics(
                embeddings=torch.from_numpy(embeddings).float(),
                cell_type_labels=torch.from_numpy(labels).long(),
                batch_ids=torch.from_numpy(batch_ids).long(),
                subsample=True,
                k=k,
                compute_graph_connectivity=enable_bio_graph_connectivity,
                gene_expression=gene_expr_matrix,
            )
            # Filter None values and convert to flat dict
            bio_metrics = {
                f"bio_{key}": float(val)
                for key, val in src_metrics.items()
                if val is not None
            }
            # Drop batch-dependent metrics on single-batch datasets — their
            # values would be degenerate (iLISI ≡ 1.0, BRAS ill-defined). scGraph
            # is kept: its batch_ids only group references, single-batch yields
            # one valid reference frame.
            if not has_batch:
                for m in ("bio_bras", "bio_ilisi"):
                    bio_metrics.pop(m, None)
            if not bio_metrics:
                bio_skip_reason = "bio_failed" if has_batch else "single_batch"
                bio_metrics = None
        except Exception as exc:
            print(f"Warning: bio metrics failed: {exc}")
            bio_skip_reason = "bio_failed"

    return BenchmarkResults(
        classification=classification,
        scib_metrics=bio_metrics,
        scib_skip_reason=bio_skip_reason,
    )


# ---------------------------------------------------------------------------
# Reconstruction metric primitives are hosted in
# ``src/sctrilemma/utils/metrics.py``.  The aliases below re-bind the
# canonical src names to the legacy underscore-prefixed names expected by the
# ``compute_reconstruction_metrics`` body so the orchestrator can stay unchanged.
# See the plan at ``~/.claude/plans/floofy-spinning-sparkle.md`` for the dedup
# strategy and parity evidence.
# ---------------------------------------------------------------------------
from sctrilemma.utils.metrics import (  # noqa: E402, I001
    # public names (dense GPU helpers) — average-tie Spearman by default
    normalize_to_log1p_cp10k as _normalize_to_log1p_cp10k,
    cell_pearson_dense as _cell_pearson_gpu,
    cell_spearman_dense as _cell_spearman_gpu,
    gene_pearson_dense as _gene_pearson_gpu,
    gene_spearman_dense as _gene_spearman_gpu,
    hvg_cell_correlations_dense as _hvg_cell_correlations_gpu,
    gene_mean_correlation_dense as _gene_mean_correlation_gpu,
    mmd_pca_dense as _mmd_pca_gpu,
    wasserstein2_dense as _wasserstein2_gpu,
    frechet_distance_dense as _frechet_distance_gpu,
    gene_corr_matrix_pearson_dense as _gene_corr_matrix_pearson_gpu,
    cell_corr_ks_dense as _cell_corr_ks_gpu,
    compute_mae_and_zero_stats as _compute_mae_and_zero_stats,
    compute_marginal_ks as _compute_marginal_ks,
    compute_cell_distance_ks as _compute_cell_distance_ks,
    compute_cell_detect_freq_ks as _compute_cell_detect_freq_ks,
    # CPU fallback path (used when on_gpu=False)
    _weighted_chunk_correlation,
    _weighted_chunk_spearman_correlation,
    _compute_gene_and_hvg_metrics,
    _compute_gene_mean_correlation,
    _compute_mmd_pca,
    _compute_wasserstein2,
    _compute_frechet_distance,
    _compute_joint_distribution_metrics,
)




def _compute_recon_chunked_gpu(
    raw_norm: np.ndarray,
    recon_norm: np.ndarray,
    scalar_metrics: dict,
    cell_chunk_size: int = 10_000,
    gene_chunk_size: int = 200,
) -> Tuple["ReconstructionMetrics", Optional[str]]:
    """Chunked GPU path for large datasets that would OOM with full-array.

    Mathematically identical to the full-GPU path:
      - Cell-level metrics: chunk over cells, weighted-average per-cell values.
      - Gene-level Pearson: accumulate per-gene scatter stats across cell
        chunks (sum_x, sum_y, sum_xx, sum_yy, sum_xy), then compute Pearson
        from aggregates.
      - Gene-level Spearman: chunk over GENES (load all N values per gene
        chunk to GPU, rank via rankdata_average_gpu, correlate).
      - HVG: select HVG from gene_var, then chunk cell-level on HVG subset.
      - mmd_pca, cell_corr_ks: already subsample internally (≤2000 cells).
      - gene_corr_matrix: top-100 genes → N×100 fits in GPU.
      - marginal_ks: CPU numpy (no GPU needed).
      - gene_mean_correlation: accumulate per-gene sums, then correlate means.
    """
    dev = torch.device("cuda")
    n_cells, n_genes = raw_norm.shape

    # ── 1. Cell-level correlations: chunk over cells ──
    # Accumulate: total_sum += chunk_mean * n_valid, total_count += n_valid
    accum = {}
    for key in (
        "recon_pearson", "recon_spearman",
        "recon_pearson_nonzero", "recon_spearman_nonzero",
        "recon_pearson_highexpr", "recon_spearman_highexpr",
    ):
        accum[key] = {"wsum": 0.0, "count": 0}

    # Also accumulate gene-level scatter stats (float64 for precision)
    sum_r = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    sum_p = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    sum_rr = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    sum_pp = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    sum_rp = torch.zeros(n_genes, dtype=torch.float64, device=dev)

    for start in range(0, n_cells, cell_chunk_size):
        end = min(start + cell_chunk_size, n_cells)
        raw_c = torch.from_numpy(raw_norm[start:end]).to(dev)
        rec_c = torch.from_numpy(recon_norm[start:end]).to(dev)

        # Gene-level scatter accumulation
        sum_r += raw_c.double().sum(dim=0)
        sum_p += rec_c.double().sum(dim=0)
        sum_rr += (raw_c.double() ** 2).sum(dim=0)
        sum_pp += (rec_c.double() ** 2).sum(dim=0)
        sum_rp += (raw_c.double() * rec_c.double()).sum(dim=0)

        # Cell-level: full mask
        mask_full = torch.ones_like(raw_c, dtype=torch.bool)
        n_full = int((mask_full.sum(dim=-1) >= 2).sum().item())
        if n_full > 0:
            accum["recon_pearson"]["wsum"] += _cell_pearson_gpu(raw_c, rec_c, mask_full) * n_full
            accum["recon_pearson"]["count"] += n_full
            accum["recon_spearman"]["wsum"] += _cell_spearman_gpu(raw_c, rec_c, mask_full) * n_full
            accum["recon_spearman"]["count"] += n_full

        # Cell-level: nonzero mask
        mask_nz = raw_c > 0
        n_nz = int((mask_nz.sum(dim=-1) >= 2).sum().item())
        if n_nz > 0:
            accum["recon_pearson_nonzero"]["wsum"] += _cell_pearson_gpu(raw_c, rec_c, mask_nz) * n_nz
            accum["recon_pearson_nonzero"]["count"] += n_nz
            accum["recon_spearman_nonzero"]["wsum"] += _cell_spearman_gpu(raw_c, rec_c, mask_nz) * n_nz
            accum["recon_spearman_nonzero"]["count"] += n_nz

        # Cell-level: highexpr mask
        mask_he = raw_c >= 2.0
        mask_he = mask_he & (mask_he.sum(dim=-1, keepdim=True) >= 10)
        n_he = int((mask_he.sum(dim=-1) >= 10).sum().item())
        if n_he > 0:
            accum["recon_pearson_highexpr"]["wsum"] += _cell_pearson_gpu(raw_c, rec_c, mask_he, min_valid_count=10) * n_he
            accum["recon_pearson_highexpr"]["count"] += n_he
            accum["recon_spearman_highexpr"]["wsum"] += _cell_spearman_gpu(raw_c, rec_c, mask_he, min_valid_count=10) * n_he
            accum["recon_spearman_highexpr"]["count"] += n_he

        del raw_c, rec_c, mask_full, mask_nz, mask_he
        torch.cuda.empty_cache()

    for key, a in accum.items():
        scalar_metrics[key] = a["wsum"] / max(a["count"], 1)

    # ── 2. Gene-level Pearson from accumulated scatter stats ──
    eps = 1e-8
    r_mean = sum_r / n_cells
    p_mean = sum_p / n_cells
    cov = sum_rp / n_cells - r_mean * p_mean
    r_var = (sum_rr / n_cells - r_mean ** 2).clamp(min=0)
    p_var = (sum_pp / n_cells - p_mean ** 2).clamp(min=0)
    r_std = r_var.sqrt()
    p_std = p_var.sqrt()
    denom = r_std * p_std
    valid = denom > eps
    gene_pearson = float((cov[valid] / denom[valid]).float().mean().item()) if bool(valid.any()) else float("nan")
    scalar_metrics["gene_pearson"] = gene_pearson
    gene_var = r_var.float()  # for HVG
    del sum_r, sum_p, sum_rr, sum_pp, sum_rp, r_mean, p_mean, cov, r_var, p_var

    # ── 3. Gene-level Spearman: chunk over genes ──
    from sctrilemma.utils.metrics import rankdata_average_gpu

    gene_spearman_corrs: list[float] = []
    for g_start in range(0, n_genes, gene_chunk_size):
        g_end = min(g_start + gene_chunk_size, n_genes)
        # Load all cells for this gene chunk: (N, chunk) → transpose → (chunk, N)
        raw_gc = torch.from_numpy(raw_norm[:, g_start:g_end].copy()).to(dev).T  # (chunk, N)
        rec_gc = torch.from_numpy(recon_norm[:, g_start:g_end].copy()).to(dev).T
        r_ranks = rankdata_average_gpu(raw_gc)
        p_ranks = rankdata_average_gpu(rec_gc)
        r_m = r_ranks.mean(dim=-1, keepdim=True)
        p_m = p_ranks.mean(dim=-1, keepdim=True)
        r_c = r_ranks - r_m
        p_c = p_ranks - p_m
        c = (r_c * p_c).sum(dim=-1)
        rn = (r_c * r_c).sum(dim=-1).sqrt()
        pn = (p_c * p_c).sum(dim=-1).sqrt()
        d = rn * pn
        v = d > eps
        if bool(v.any()):
            corrs = c[v] / d[v]
            finite = torch.isfinite(corrs)
            if bool(finite.any()):
                gene_spearman_corrs.extend(corrs[finite].cpu().tolist())
        del raw_gc, rec_gc, r_ranks, p_ranks
        torch.cuda.empty_cache()

    scalar_metrics["gene_spearman"] = float(np.mean(gene_spearman_corrs)) if gene_spearman_corrs else 0.0

    # ── 4. HVG cell correlations: select HVG, then chunk over cells ──
    n_top = 2000
    valid_var = gene_var > 0
    k_hvg = min(n_top, int(valid_var.sum().item()))
    if k_hvg >= 2:
        top_idx = torch.argsort(gene_var, descending=True)[:k_hvg]
        top_idx_sorted, _ = torch.sort(top_idx)
        top_np = top_idx_sorted.cpu().numpy()

        hvg_p_wsum, hvg_p_count = 0.0, 0
        hvg_s_wsum, hvg_s_count = 0.0, 0
        for start in range(0, n_cells, cell_chunk_size):
            end = min(start + cell_chunk_size, n_cells)
            raw_h = torch.from_numpy(raw_norm[start:end][:, top_np].copy()).to(dev)
            rec_h = torch.from_numpy(recon_norm[start:end][:, top_np].copy()).to(dev)
            mask_h = torch.ones_like(raw_h, dtype=torch.bool)
            n_v = int((mask_h.sum(dim=-1) >= 2).sum().item())
            if n_v > 0:
                hvg_p_wsum += _cell_pearson_gpu(raw_h, rec_h, mask_h) * n_v
                hvg_p_count += n_v
                hvg_s_wsum += _cell_spearman_gpu(raw_h, rec_h, mask_h) * n_v
                hvg_s_count += n_v
            del raw_h, rec_h, mask_h
            torch.cuda.empty_cache()
        scalar_metrics["recon_pearson_hvg"] = hvg_p_wsum / max(hvg_p_count, 1)
        scalar_metrics["recon_spearman_hvg"] = hvg_s_wsum / max(hvg_s_count, 1)
    else:
        scalar_metrics["recon_pearson_hvg"] = 0.0
        scalar_metrics["recon_spearman_hvg"] = 0.0

    # ── 5. Gene mean correlation: accumulate means then correlate ──
    raw_gene_mean = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    rec_gene_mean = torch.zeros(n_genes, dtype=torch.float64, device=dev)
    for start in range(0, n_cells, cell_chunk_size):
        end = min(start + cell_chunk_size, n_cells)
        raw_gene_mean += torch.from_numpy(raw_norm[start:end]).to(dev).double().sum(dim=0)
        rec_gene_mean += torch.from_numpy(recon_norm[start:end]).to(dev).double().sum(dim=0)
    raw_gene_mean /= n_cells
    rec_gene_mean /= n_cells
    # Stack as (2, G) for gene_mean_correlation_dense
    stacked_raw = raw_gene_mean.float().unsqueeze(0)  # (1, G)
    stacked_rec = rec_gene_mean.float().unsqueeze(0)
    gm_p, gm_s = _gene_mean_correlation_gpu(
        stacked_raw.expand(2, -1), stacked_rec.expand(2, -1)
    )
    # Actually gene_mean_correlation_dense computes mean(dim=0) internally,
    # so we need to pass the raw/recon means directly. Let me inline it.
    rm = raw_gene_mean.float()
    pm = rec_gene_mean.float()
    a = rm - rm.mean()
    b = pm - pm.mean()
    d = a.norm() * b.norm() + 1e-8
    gm_p = float(((a * b).sum() / d).item())
    raw_r = rankdata_average_gpu(rm.unsqueeze(0)).squeeze(0)
    rec_r = rankdata_average_gpu(pm.unsqueeze(0)).squeeze(0)
    a = raw_r - raw_r.mean()
    b = rec_r - rec_r.mean()
    d = a.norm() * b.norm() + 1e-8
    gm_s = float(((a * b).sum() / d).item())
    import math
    scalar_metrics["gene_mean_pearson"] = gm_p if math.isfinite(gm_p) else 0.0
    scalar_metrics["gene_mean_spearman"] = gm_s if math.isfinite(gm_s) else 0.0
    del raw_gene_mean, rec_gene_mean

    # ── 6. mmd_pca + gene_corr_matrix + cell_corr_ks ──
    # These use small subsets; load only what's needed.

    # mmd_pca: subsample to 2000 cells → tiny on GPU
    try:
        rng = np.random.default_rng(seed=0)
        n_sub = min(2000, n_cells)
        sub_idx = np.sort(rng.choice(n_cells, n_sub, replace=False))
        raw_sub = torch.from_numpy(raw_norm[sub_idx].copy()).to(dev)
        rec_sub = torch.from_numpy(recon_norm[sub_idx].copy()).to(dev)
        scalar_metrics["mmd_pca"] = _mmd_pca_gpu(raw_sub, rec_sub)
        scalar_metrics["wasserstein2"] = _wasserstein2_gpu(raw_sub, rec_sub)
        scalar_metrics["frechet_distance"] = _frechet_distance_gpu(raw_sub, rec_sub)
        del raw_sub, rec_sub
    except Exception as exc:
        print(f"Warning: MMD PCA / W2 / FD (chunked GPU) failed: {exc}")
        scalar_metrics.setdefault("mmd_pca", float("nan"))
        scalar_metrics.setdefault("wasserstein2", float("nan"))
        scalar_metrics.setdefault("frechet_distance", float("nan"))
    torch.cuda.empty_cache()

    # marginal_ks: CPU-only
    try:
        scalar_metrics.update(_compute_marginal_ks(raw_norm, recon_norm))
    except Exception as exc:
        print(f"Warning: marginal KS failed: {exc}")
        for key in ("marginal_gene_mean_ks", "marginal_gene_var_ks", "marginal_gene_detect_ks"):
            scalar_metrics[key] = float("nan")

    # gene_corr_matrix: top-100 genes → N×100 on GPU (fits easily)
    try:
        gene_mean_t = torch.from_numpy(raw_norm.mean(axis=0).astype(np.float32)).to(dev)
        top100 = torch.argsort(gene_mean_t, descending=True)[:100].cpu().numpy()
        raw_g100 = torch.from_numpy(raw_norm[:, top100].copy()).to(dev)
        rec_g100 = torch.from_numpy(recon_norm[:, top100].copy()).to(dev)
        scalar_metrics["gene_corr_matrix_pearson"] = _gene_corr_matrix_pearson_gpu(raw_g100, rec_g100)
        del raw_g100, rec_g100
    except Exception as exc:
        print(f"Warning: gene_corr_matrix (chunked GPU) failed: {exc}")
        scalar_metrics["gene_corr_matrix_pearson"] = float("nan")
    torch.cuda.empty_cache()

    # cell_corr_ks: subsample 2000 cells
    try:
        rng = np.random.default_rng(seed=1)
        n_sub = min(2000, n_cells)
        sub_idx = np.sort(rng.choice(n_cells, n_sub, replace=False))
        raw_sub = torch.from_numpy(raw_norm[sub_idx].copy()).to(dev)
        rec_sub = torch.from_numpy(recon_norm[sub_idx].copy()).to(dev)
        scalar_metrics["cell_corr_ks"] = _cell_corr_ks_gpu(raw_sub, rec_sub)
        del raw_sub, rec_sub
    except Exception as exc:
        print(f"Warning: cell_corr_ks (chunked GPU) failed: {exc}")
        scalar_metrics["cell_corr_ks"] = float("nan")
    torch.cuda.empty_cache()

    # scDesign3 cell-distance KS and cell-detection-frequency KS (CPU)
    try:
        scalar_metrics["cell_distance_ks"] = _compute_cell_distance_ks(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: cell_distance_ks failed: {exc}")
        scalar_metrics["cell_distance_ks"] = float("nan")
    try:
        scalar_metrics["cell_detect_freq_ks"] = _compute_cell_detect_freq_ks(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: cell_detect_freq_ks failed: {exc}")
        scalar_metrics["cell_detect_freq_ks"] = float("nan")

    # Legacy fields
    scalar_metrics["recon_pearson_log1p"] = scalar_metrics["recon_pearson"]
    scalar_metrics["recon_pearson_sqrt"] = scalar_metrics["recon_pearson"]

    if "recon_mse" not in scalar_metrics:
        try:
            import torch as _torch
            diff_sq = (_torch.as_tensor(raw_norm) - _torch.as_tensor(recon_norm)) ** 2
            scalar_metrics["recon_mse"] = float(diff_sq.mean().item())
        except Exception:
            scalar_metrics["recon_mse"] = float("nan")

    return ReconstructionMetrics(**scalar_metrics), None


def compute_reconstruction_metrics(
    raw_matrix,
    reconstructed: np.ndarray,
    *,
    threshold: float = 0.01,
    cell_chunk_size: int = 1024,
    on_gpu: Optional[bool] = None,
) -> Tuple[ReconstructionMetrics, Optional[str]]:
    """Compute reconstruction metrics in **log1p(CP10K)** space.

    Both *raw_matrix* and *reconstructed* are independently normalized via
    ``log1p(x / sum(x) * 10000)`` before correlation/MAE metrics are computed.
    Zero-recovery metrics are computed in raw count space with a per-cell
    rank-based predicted-zero classifier — this handles softmax/dense outputs
    (scVI, scTrilemma) correctly, since ``pred_zero`` is calibrated to match the
    actual zero rate per cell.

    The *threshold* parameter is retained for backward compatibility but
    unused in the new per-cell rank-based zero classification.
    """
    del threshold  # deprecated — kept in signature for backward compat
    reconstructed = np.ascontiguousarray(reconstructed, dtype=np.float32)
    if reconstructed.ndim != 2:
        raise ValueError("reconstructed must be a 2D array")

    # Materialise raw counts into a dense float32 buffer that we fully own.
    # NOTE: ``reconstructed`` and the dense-raw buffer are normalized
    # **in place** below; callers must not rely on these arrays retaining
    # their original (pre-call) contents.  sctrilemma.benchmark.run does not reuse
    # them after this call, so this is safe.
    if sp.issparse(raw_matrix):
        raw_norm = raw_matrix.toarray().astype(np.float32, copy=False)
    else:
        raw_norm = np.array(raw_matrix, dtype=np.float32, copy=True)

    # --- Normalize to log1p(CP10K) in place ---
    _normalize_to_log1p_cp10k(raw_norm, in_place=True)
    recon_norm = reconstructed  # alias; normalized in place next
    _normalize_to_log1p_cp10k(recon_norm, in_place=True)

    # GPU dispatch: default ON when CUDA is available.  Parity vs CPU is
    # bit-exact (≲ 1e-7) for every correlation-based metric and within
    # ~2e-3 for mmd_pca (torch SVD vs sklearn PCA numerics).  See
    # experiments/_gpu_metrics_proto.py for the parity check.
    if on_gpu is None:
        on_gpu = torch.cuda.is_available()
    if on_gpu and not torch.cuda.is_available():
        on_gpu = False

    # --- MAE (log1p CP10K) and zero stats ---
    scalar_metrics = _compute_mae_and_zero_stats(
        raw_norm,
        recon_norm,
        cell_chunk_size=cell_chunk_size,
    )

    if on_gpu:
        # Estimate GPU memory: N × G × 4 bytes × ~12 concurrent buffers.
        # If that exceeds 70% of free VRAM, switch to chunked GPU path
        # which processes cells/genes in slices (mathematically identical).
        n_cells, n_genes = raw_norm.shape
        est_bytes = n_cells * n_genes * 4 * 12
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except Exception:
            free_bytes = float("inf")
        if est_bytes > free_bytes * 0.7:
            print(
                f"  [recon] Chunked GPU path: {n_cells:,}×{n_genes} "
                f"needs ~{est_bytes / 1e9:.1f} GB, free={free_bytes / 1e9:.1f} GB"
            )
            chunked_metrics, gene_skip = _compute_recon_chunked_gpu(
                raw_norm, recon_norm, scalar_metrics, cell_chunk_size
            )
            return chunked_metrics, gene_skip

        # Full-GPU path: ship both buffers at once (small-enough datasets).
        dev = torch.device("cuda")
        raw_g = torch.from_numpy(raw_norm).to(dev)
        recon_g = torch.from_numpy(recon_norm).to(dev)
        mask_full = torch.ones_like(raw_g, dtype=torch.bool)
        mask_nonzero = raw_g > 0.0
        highexpr = raw_g >= 2.0
        highexpr = highexpr & (highexpr.sum(dim=-1, keepdim=True) >= 10)

        scalar_metrics["recon_pearson"] = _cell_pearson_gpu(
            raw_g, recon_g, mask_full
        )
        scalar_metrics["recon_spearman"] = _cell_spearman_gpu(
            raw_g, recon_g, mask_full
        )
        scalar_metrics["recon_pearson_nonzero"] = _cell_pearson_gpu(
            raw_g, recon_g, mask_nonzero
        )
        scalar_metrics["recon_spearman_nonzero"] = _cell_spearman_gpu(
            raw_g, recon_g, mask_nonzero
        )
        scalar_metrics["recon_pearson_highexpr"] = _cell_pearson_gpu(
            raw_g, recon_g, highexpr, min_valid_count=10
        )
        scalar_metrics["recon_spearman_highexpr"] = _cell_spearman_gpu(
            raw_g, recon_g, highexpr, min_valid_count=10
        )

        gene_p, gene_var_g = _gene_pearson_gpu(raw_g, recon_g)
        scalar_metrics["gene_pearson"] = gene_p
        scalar_metrics["gene_spearman"] = _gene_spearman_gpu(raw_g, recon_g)
        hvg_p, hvg_s = _hvg_cell_correlations_gpu(raw_g, recon_g, gene_var_g)
        scalar_metrics["recon_pearson_hvg"] = hvg_p
        scalar_metrics["recon_spearman_hvg"] = hvg_s

        gm_p, gm_s = _gene_mean_correlation_gpu(raw_g, recon_g)
        scalar_metrics["gene_mean_pearson"] = gm_p
        scalar_metrics["gene_mean_spearman"] = gm_s

        try:
            scalar_metrics["mmd_pca"] = _mmd_pca_gpu(raw_g, recon_g)
        except Exception as exc:
            print(f"Warning: MMD PCA (GPU) failed: {exc}")
            scalar_metrics["mmd_pca"] = float("nan")

        try:
            scalar_metrics["wasserstein2"] = _wasserstein2_gpu(raw_g, recon_g)
        except Exception as exc:
            print(f"Warning: W2 (GPU) failed: {exc}")
            scalar_metrics["wasserstein2"] = float("nan")

        try:
            scalar_metrics["frechet_distance"] = _frechet_distance_gpu(raw_g, recon_g)
        except Exception as exc:
            print(f"Warning: FD (GPU) failed: {exc}")
            scalar_metrics["frechet_distance"] = float("nan")

        try:
            scalar_metrics.update(_compute_marginal_ks(raw_norm, recon_norm))
        except Exception as exc:
            print(f"Warning: marginal KS (CPU) failed: {exc}")
            for key in (
                "marginal_gene_mean_ks",
                "marginal_gene_var_ks",
                "marginal_gene_detect_ks",
            ):
                scalar_metrics[key] = float("nan")

        try:
            scalar_metrics["gene_corr_matrix_pearson"] = (
                _gene_corr_matrix_pearson_gpu(raw_g, recon_g)
            )
        except Exception as exc:
            print(f"Warning: gene_corr_matrix (GPU) failed: {exc}")
            scalar_metrics["gene_corr_matrix_pearson"] = float("nan")

        try:
            scalar_metrics["cell_corr_ks"] = _cell_corr_ks_gpu(raw_g, recon_g)
        except Exception as exc:
            print(f"Warning: cell_corr_ks (hybrid GPU+CPU) failed: {exc}")
            scalar_metrics["cell_corr_ks"] = float("nan")

        try:
            scalar_metrics["cell_distance_ks"] = _compute_cell_distance_ks(raw_norm, recon_norm)
        except Exception as exc:
            print(f"Warning: cell_distance_ks failed: {exc}")
            scalar_metrics["cell_distance_ks"] = float("nan")
        try:
            scalar_metrics["cell_detect_freq_ks"] = _compute_cell_detect_freq_ks(raw_norm, recon_norm)
        except Exception as exc:
            print(f"Warning: cell_detect_freq_ks failed: {exc}")
            scalar_metrics["cell_detect_freq_ks"] = float("nan")

        # Legacy fields: same as recon_pearson in log1p(CP10K) space
        scalar_metrics["recon_pearson_log1p"] = scalar_metrics["recon_pearson"]
        scalar_metrics["recon_pearson_sqrt"] = scalar_metrics["recon_pearson"]

        # Recon MSE: required by dataclass. Compute as mean squared error on the
        # full aligned (raw vs reconstructed) log1p(CP10K) space. Previously
        # absent from the dict and caused TypeError when constructing the
        # dataclass.
        if "recon_mse" not in scalar_metrics:
            try:
                import torch as _torch
                diff_sq = (_torch.as_tensor(raw_norm) - _torch.as_tensor(recon_norm)) ** 2
                scalar_metrics["recon_mse"] = float(diff_sq.mean().item())
            except Exception:
                scalar_metrics["recon_mse"] = float("nan")

        del raw_g, recon_g, mask_full, mask_nonzero, highexpr, gene_var_g
        torch.cuda.empty_cache()

        return ReconstructionMetrics(**scalar_metrics), None

    # --- Correlation variants ---
    def full_mask(raw: np.ndarray) -> np.ndarray:
        return np.ones(raw.shape, dtype=bool)

    # "nonzero" = genes with nonzero expression in the ground truth.
    # In log1p(CP10K) space, raw_count==0 maps to exactly 0.0.
    def nonzero_mask(raw: np.ndarray) -> np.ndarray:
        return raw > 0.0

    def highexpr_mask(raw: np.ndarray) -> np.ndarray:
        # High-expression threshold in log1p(CP10K): log1p(4 / sum * 10000)
        # is dataset-dependent, so we use a fixed threshold in this space.
        # log1p(CP10K) >= 2.0 roughly corresponds to CP10K >= 6.4, i.e.
        # genes contributing >=0.064% of library (moderately expressed).
        mask = raw >= 2.0
        valid_rows = mask.sum(axis=1) >= 10
        if valid_rows.any():
            mask = mask & valid_rows[:, None]
        return mask

    scalar_metrics.update(
        {
            "recon_pearson": _weighted_chunk_correlation(
                raw_norm,
                recon_norm,
                full_mask,
                compute_pearson_correlation,
                min_valid_count=2,
                cell_chunk_size=cell_chunk_size,
            ),
            "recon_spearman": _weighted_chunk_spearman_correlation(
                raw_norm,
                recon_norm,
                full_mask,
                min_valid_count=2,
                cell_chunk_size=cell_chunk_size,
            ),
            "recon_pearson_nonzero": _weighted_chunk_correlation(
                raw_norm,
                recon_norm,
                nonzero_mask,
                compute_pearson_correlation,
                min_valid_count=2,
                cell_chunk_size=cell_chunk_size,
            ),
            "recon_spearman_nonzero": _weighted_chunk_spearman_correlation(
                raw_norm,
                recon_norm,
                nonzero_mask,
                min_valid_count=2,
                cell_chunk_size=cell_chunk_size,
            ),
            "recon_pearson_highexpr": _weighted_chunk_correlation(
                raw_norm,
                recon_norm,
                highexpr_mask,
                compute_pearson_correlation,
                min_valid_count=10,
                cell_chunk_size=cell_chunk_size,
            ),
            "recon_spearman_highexpr": _weighted_chunk_spearman_correlation(
                raw_norm,
                recon_norm,
                highexpr_mask,
                min_valid_count=10,
                cell_chunk_size=cell_chunk_size,
            ),
        }
    )

    gene_metrics, gene_skip_reason = _compute_gene_and_hvg_metrics(
        raw_norm,
        recon_norm,
    )
    scalar_metrics.update(gene_metrics)

    # Distribution/generation metrics (scGen/scLDM style)
    gene_mean_p, gene_mean_s = _compute_gene_mean_correlation(raw_norm, recon_norm)
    scalar_metrics["gene_mean_pearson"] = gene_mean_p
    scalar_metrics["gene_mean_spearman"] = gene_mean_s

    try:
        scalar_metrics["mmd_pca"] = _compute_mmd_pca(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: MMD PCA computation failed: {exc}")
        scalar_metrics["mmd_pca"] = float("nan")

    try:
        scalar_metrics["wasserstein2"] = _compute_wasserstein2(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: Wasserstein-2 computation failed: {exc}")
        scalar_metrics["wasserstein2"] = float("nan")

    try:
        scalar_metrics["frechet_distance"] = _compute_frechet_distance(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: Fréchet Distance computation failed: {exc}")
        scalar_metrics["frechet_distance"] = float("nan")

    # scDesign2/3 marginal distribution metrics
    # (per-metric try/except is inside _compute_marginal_ks)
    try:
        scalar_metrics.update(
            _compute_marginal_ks(raw_norm, recon_norm)
        )
    except Exception as exc:
        print(f"Warning: marginal KS metrics wrapper failed: {exc}")
        for key in (
            "marginal_gene_mean_ks",
            "marginal_gene_var_ks",
            "marginal_gene_detect_ks",
        ):
            scalar_metrics[key] = float("nan")

    # scDesign3 joint distribution metrics
    # (per-metric try/except is inside _compute_joint_distribution_metrics)
    try:
        scalar_metrics.update(_compute_joint_distribution_metrics(raw_norm, recon_norm))
    except Exception as exc:
        print(f"Warning: joint distribution metrics wrapper failed: {exc}")
        scalar_metrics["gene_corr_matrix_pearson"] = float("nan")
        scalar_metrics["cell_corr_ks"] = float("nan")

    try:
        scalar_metrics["cell_distance_ks"] = _compute_cell_distance_ks(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: cell_distance_ks failed: {exc}")
        scalar_metrics["cell_distance_ks"] = float("nan")
    try:
        scalar_metrics["cell_detect_freq_ks"] = _compute_cell_detect_freq_ks(raw_norm, recon_norm)
    except Exception as exc:
        print(f"Warning: cell_detect_freq_ks failed: {exc}")
        scalar_metrics["cell_detect_freq_ks"] = float("nan")

    # Legacy fields: in log1p(CP10K) space these equal recon_pearson.
    scalar_metrics["recon_pearson_log1p"] = scalar_metrics["recon_pearson"]
    scalar_metrics["recon_pearson_sqrt"] = scalar_metrics["recon_pearson"]

    if "recon_mse" not in scalar_metrics:
        try:
            import torch as _torch
            diff_sq = (_torch.as_tensor(raw_norm) - _torch.as_tensor(recon_norm)) ** 2
            scalar_metrics["recon_mse"] = float(diff_sq.mean().item())
        except Exception:
            scalar_metrics["recon_mse"] = float("nan")

    return ReconstructionMetrics(**scalar_metrics), gene_skip_reason


def _metric_type(
    metric_name: str,
    scib_metric_names: set[str],
) -> str:
    if metric_name in {"accuracy", "f1_weighted", "f1_macro"}:
        return "classification"
    if metric_name in scib_metric_names or metric_name.startswith("bio_"):
        return "bio"
    if metric_name.startswith("gene_"):
        return "gene"
    return "reconstruction"


def _should_emit_metric(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def flatten_benchmark_results(
    results: BenchmarkResults,
    meta: Dict[str, str],
) -> List[Dict[str, str]]:
    """Flatten benchmark results to one CSV row per emitted metric."""
    rows: List[Dict[str, str]] = []
    metrics_dict = results.to_dict()
    scib_metric_names = set(results.scib_metrics.keys()) if results.scib_metrics else set()

    for metric_name, value in metrics_dict.items():
        if not _should_emit_metric(value):
            continue
        entry = {
            "dataset": meta["dataset"],
            "model": meta["model"],
            "input_type": "embedding",
            "target_type": "cell_type",
            "loss_type": "zero_shot",
            "embedding_key": meta.get("embedding_key", f"X_{meta['model']}"),
            "metric": metric_name,
            "metric_type": _metric_type(metric_name, scib_metric_names),
            "value": f"{float(value):.6f}",
        }
        rows.append(entry)

    return rows


def summarize_results_csv(results_path: Path, summary_path: Path) -> Optional[pd.DataFrame]:
    """Generate a summary CSV from the full results CSV."""
    if not results_path.exists():
        return None

    results_df = pd.read_csv(results_path)
    if results_df.empty:
        return None

    summary = (
        results_df.groupby(["model", "metric", "metric_type"])
        .agg(
            value_mean=("value", lambda x: x.astype(float).mean()),
            value_std=("value", lambda x: x.astype(float).std()),
            n_datasets=("dataset", "nunique"),
        )
        .reset_index()
        .sort_values(["model", "metric", "metric_type"])
    )
    summary["value_mean"] = summary["value_mean"].apply(lambda x: f"{x:.6f}")
    summary["value_std"] = summary["value_std"].apply(lambda x: f"{x:.6f}")
    summary.to_csv(summary_path, index=False)
    return summary
