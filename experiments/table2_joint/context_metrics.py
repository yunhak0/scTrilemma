"""Donor-context invariance metrics within one biological group.

Both metrics are computed within one cell type (or cell type x disease state) so that donor
effects are not confounded with biology:

* ``residual_donor_variance``: adjusted multivariate omega-squared attributable to donor
  (0 = no detectable donor displacement, 1 = complete donor separation); the paper reports
  ``1 - value`` as *residual donor invariance*;
* ``local_donor_mixing_from_similarity``: chance-corrected same-donor enrichment among the
  cosine k nearest neighbours, clipped to [0, 1] (1 = random-or-better mixing).
"""

from __future__ import annotations

import numpy as np


def normalized_embeddings(values: np.ndarray) -> np.ndarray:
    """Return finite row-wise L2-normalized embeddings."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Embeddings contain zero-norm rows")
    return array / norms


def local_donor_mixing_from_similarity(
    similarity: np.ndarray,
    donors: np.ndarray,
    *,
    k: int,
) -> float:
    """Chance-correct same-donor enrichment among the top-k neighbors.

    The result is clipped to [0, 1], where 1 denotes random-or-better donor
    mixing and 0 denotes complete same-donor segregation.
    """
    matrix = np.asarray(similarity, dtype=np.float32)
    donor_values = np.asarray(donors).astype(str)
    if matrix.shape != (len(donor_values), len(donor_values)):
        raise ValueError("Similarity and donor dimensions do not match")
    if not 1 <= k < len(donor_values):
        raise ValueError("k must be between 1 and n_cells - 1")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, -np.inf)
    neighbors = np.argpartition(-matrix, kth=k - 1, axis=1)[:, :k]
    observed = float(np.mean(donor_values[neighbors] == donor_values[:, None]))
    _, counts = np.unique(donor_values, return_counts=True)
    n_cells = len(donor_values)
    chance = float(np.sum(counts * (counts - 1)) / (n_cells * (n_cells - 1)))
    enrichment = (observed - chance) / max(1.0 - chance, 1e-12)
    return float(np.clip(1.0 - enrichment, 0.0, 1.0))


def residual_donor_variance(values: np.ndarray, donors: np.ndarray) -> float:
    """Adjusted multivariate omega-squared attributable to donor.

    The input should contain one cell type only.  Zero means no detectable
    donor-associated displacement and one means complete donor separation.
    """
    array = np.asarray(values, dtype=np.float64)
    donor_values = np.asarray(donors).astype(str)
    if array.ndim != 2 or len(array) != len(donor_values):
        raise ValueError("Embedding and donor dimensions do not match")
    unique_donors = np.unique(donor_values)
    if len(unique_donors) < 2:
        raise ValueError("At least two donors are required")
    grand_mean = array.mean(axis=0)
    ss_between = 0.0
    ss_within = 0.0
    for donor in unique_donors:
        group = array[donor_values == donor]
        group_mean = group.mean(axis=0)
        ss_between += float(len(group) * np.sum((group_mean - grand_mean) ** 2))
        ss_within += float(np.sum((group - group_mean) ** 2))
    ss_total = ss_between + ss_within
    if ss_total <= 1e-15:
        return 0.0
    df_between = len(unique_donors) - 1
    df_within = len(array) - len(unique_donors)
    if df_within <= 0:
        raise ValueError("Insufficient residual degrees of freedom")
    ms_within = ss_within / df_within
    omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)
    return float(np.clip(omega_squared, 0.0, 1.0))
