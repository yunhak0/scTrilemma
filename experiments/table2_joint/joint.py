"""Matched evaluation of the three representation demands on the common cohorts (Table 2).

Five held-out disease cohorts were fixed from metadata and cache coverage only (Table 3 and
``cohort_audit.csv``). Every repeat balances two disease states, two donors per state and
ten cells per donor within each eligible cell type; the same selected cells are scored for
biological identity and state (label-balanced cross-donor retrieval on the embeddings),
donor-conditional context invariance (embeddings) and expression fidelity (disease logFC
concordance of the reconstructions with the raw counts on the paper's common gene universe).

Caches are given as repeated ``--embedding-cache NAME=DIR`` and ``--recon-cache NAME=DIR``
options (release defaults: ``sctrilemma``); any cache with the release layout can be scored.

    pixi run python -m experiments.table2_joint.joint
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scib_metrics import bras
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from experiments.common import (
    cache_payload,
    dense_columns,
    feature_lookup,
    normalize_gene,
    read_ids,
    stable_seed,
)
from experiments.table2_joint._common import (
    ASSAY_KEY,
    CELL_TYPE_KEY,
    DEFAULT_CANDIDATE_PAIRS,
    DEFAULT_EMBEDDING_DIR,
    DEFAULT_IDS_FILE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RECON_DIR,
    DEFAULT_SAMPLES_DIR,
    DONOR_KEY,
    MODEL_NAME,
    SOMA_KEY,
    STATE_KEY,
    TISSUE_KEY,
    GeneUniverse,
    add_gene_universe_arguments,
    cache_path,
    cache_soma_ids,
    describe_path,
    normalized_obs,
    parse_model_dirs,
)
from experiments.table2_joint.context_metrics import (
    local_donor_mixing_from_similarity,
    normalized_embeddings,
    residual_donor_variance,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "joint"
OBS_COLUMNS = [SOMA_KEY, ASSAY_KEY, TISSUE_KEY, STATE_KEY, CELL_TYPE_KEY, DONOR_KEY]
CONTEXT_METRICS = ("donor_bras_conditional", "residual_donor_invariance", "local_donor_mixing")
COMPARISON_METRICS = (
    "biological_identity",
    "biological_state",
    "context_invariance",
    "donor_bras",
    "local_donor_mixing",
    "expression_fidelity",
    "deg_jaccard",
    "deg_sign",
)
PARETO_AXES = ["biological_identity", "context_invariance", "expression_fidelity"]


@dataclass(frozen=True)
class Cohort:
    name: str
    dataset_id: str
    assay: str
    tissue: str
    disease: str
    reference: str = "normal"


COHORTS = (
    Cohort(
        "Kidney",
        "867757c1-3b1a-49d9-a0cd-17767eb160cc",
        "10x multiome",
        "kidney",
        "obstructive nephropathy",
    ),
    Cohort(
        "Liver",
        "e3ed2ba4-edf5-40ac-8750-8a417ad1eefe",
        "BD Rhapsody Whole Transcriptome Analysis",
        "liver",
        "colorectal carcinoma || metastatic malignant neoplasm",
    ),
    Cohort(
        "Brain",
        "203025fe-fa99-4d57-81da-458ed8f0c334",
        "10x multiome",
        "dorsolateral prefrontal cortex",
        "cognitive disorder",
    ),
    Cohort(
        "Tendon",
        "acd544d0-4d8b-46c7-98b3-fc48f7e6fdb7",
        "10x 3' v3",
        "tendon of quadriceps femoris",
        "injury",
    ),
    Cohort(
        "Colon",
        "829a3cd1-a466-49f1-b2e9-d3f6b7f392e2",
        "10x 3' v3",
        "sigmoid colon",
        "colorectal cancer",
    ),
)


@dataclass
class Inputs:
    """Resolved input locations of one run."""

    samples_dir: Path
    manifest_dir: Path | None
    embedding_dirs: dict[str, Path]
    recon_dirs: dict[str, Path]
    universe: GeneUniverse

    def manifest_ids(self, dataset_id: str) -> np.ndarray:
        """Cell IDs (as strings) of the embedded cell universe of one dataset.

        Taken from ``--manifest-dir`` when given, otherwise from the ``soma_joinid`` array
        of the first embedding cache that carries valid IDs.
        """
        if self.manifest_dir is not None:
            with np.load(cache_path(self.manifest_dir, dataset_id), allow_pickle=False) as manifest:
                return np.asarray(manifest[SOMA_KEY]).astype(str)
        for model, directory in self.embedding_dirs.items():
            path = cache_path(directory, dataset_id)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as cached:
                ids = cache_soma_ids(cached)
            if ids is not None:
                if len(np.unique(ids)) != len(ids):
                    raise RuntimeError(f"{model}/{dataset_id}: duplicate cache soma_joinid")
                return ids
        raise RuntimeError(
            f"{dataset_id}: no embedding cache carries soma_joinid; pass --manifest-dir"
        )


def choose_dataset_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Select one metadata-supported normal contrast per held-out dataset."""
    normal = candidates.loc[candidates["normal_contrast"]].copy()
    normal = normal.sort_values(
        ["dataset_id", "n_cells_pair", "n_eligible_cell_types", "state_a", "state_b"],
        ascending=[True, False, False, True, True],
    )
    return normal.groupby("dataset_id", as_index=False, sort=True).first()


def common_observations(dataset_id: str, inputs: Inputs) -> pd.DataFrame:
    """Read metadata for cells shared by the staged sample and the embedded cell universe."""
    adata = ad.read_h5ad(inputs.samples_dir / f"{dataset_id}.h5ad", backed="r")
    try:
        obs = normalized_obs(adata.obs, OBS_COLUMNS)
    finally:
        adata.file.close()
    manifest_ids = set(inputs.manifest_ids(dataset_id))
    return obs.loc[obs[SOMA_KEY].isin(manifest_ids)].copy()


def strict_cell_types(
    obs: pd.DataFrame,
    *,
    assay: str,
    tissue: str,
    states: tuple[str, str],
    min_cells_per_state: int,
    min_cells_per_donor: int,
    min_donors_per_state: int,
) -> list[str]:
    """Return cell types supporting an exactly balanced three-axis comparison."""
    candidate = obs.loc[
        (obs[ASSAY_KEY] == assay) & (obs[TISSUE_KEY] == tissue) & obs[STATE_KEY].isin(states)
    ]
    eligible: list[str] = []
    for cell_type, group in candidate.groupby(CELL_TYPE_KEY, observed=True, sort=True):
        valid = True
        for state in states:
            state_group = group.loc[group[STATE_KEY] == state]
            donor_counts = state_group[DONOR_KEY].value_counts()
            valid &= len(state_group) >= min_cells_per_state
            valid &= int((donor_counts >= min_cells_per_donor).sum()) >= min_donors_per_state
        if valid:
            eligible.append(str(cell_type))
    return eligible


def audit_all_datasets(
    args: argparse.Namespace,
    inputs: Inputs,
    dataset_ids: list[str],
) -> pd.DataFrame:
    """Apply the outcome-blind selection rule to all held-out datasets."""
    candidates = choose_dataset_candidates(pd.read_csv(args.candidate_pairs))
    candidate_lookup = candidates.set_index("dataset_id")
    selected_lookup = {cohort.dataset_id: cohort for cohort in COHORTS}
    rows: list[dict[str, object]] = []
    for dataset_id in sorted(dataset_ids):
        base: dict[str, object] = {
            "dataset_id": dataset_id,
            "has_fixed_context_normal_contrast": dataset_id in candidate_lookup.index,
            "selected": dataset_id in selected_lookup,
        }
        if dataset_id not in candidate_lookup.index:
            rows.append(
                {
                    **base,
                    "assay": "",
                    "tissue": "",
                    "state_a": "",
                    "state_b": "",
                    "audit_n_cells_pair": 0,
                    "strict_n_eligible_cell_types": 0,
                    "strict_eligible_cell_types": "",
                    "decision": "No fixed-context normal contrast",
                }
            )
            continue
        candidate = candidate_lookup.loc[dataset_id]
        obs = common_observations(dataset_id, inputs)
        cell_types = strict_cell_types(
            obs,
            assay=str(candidate["assay"]),
            tissue=str(candidate["tissue"]),
            states=(str(candidate["state_a"]), str(candidate["state_b"])),
            min_cells_per_state=args.min_cells_per_state,
            min_cells_per_donor=args.min_cells_per_donor,
            min_donors_per_state=args.min_donors_per_state,
        )
        passed = len(cell_types) >= args.min_cell_types
        rows.append(
            {
                **base,
                "assay": candidate["assay"],
                "tissue": candidate["tissue"],
                "state_a": candidate["state_a"],
                "state_b": candidate["state_b"],
                "audit_n_cells_pair": int(candidate["n_cells_pair"]),
                "strict_n_eligible_cell_types": len(cell_types),
                "strict_eligible_cell_types": "; ".join(cell_types),
                "decision": "Selected" if passed else "Insufficient balanced support",
            }
        )
        if passed != (dataset_id in selected_lookup):
            raise RuntimeError(
                f"Selection audit disagrees with the frozen cohort panel for {dataset_id}"
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(dataset_ids) or int(frame["selected"].sum()) != len(COHORTS):
        raise RuntimeError(
            f"Invalid selection audit: rows={len(frame)}, selected={frame['selected'].sum()}"
        )
    return frame


def _embedding_key(files: list[str]) -> str:
    if "embeddings" in files:
        return "embeddings"
    if "embedding" in files:
        return "embedding"
    raise KeyError("Embedding cache has neither 'embedding' nor 'embeddings'")


def aligned_embeddings(
    model: str,
    path: Path,
    manifest_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    """Load only target rows and verify the exact cache-to-manifest contract.

    Caches with a valid ``soma_joinid`` array are matched by ID; caches without one must be
    row-aligned with the manifest.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as cached:
        key = _embedding_key(cached.files)
        n_rows = int(cached[key].shape[0])
        cache_ids = cache_soma_ids(cached)
        if cache_ids is None:
            cache_ids = manifest_ids
            if n_rows != len(manifest_ids):
                raise RuntimeError(f"{model}/{path.stem}: cache is not manifest-aligned")
        if n_rows != len(cache_ids):
            raise RuntimeError(f"{model}/{path.stem}: cache ID length mismatch")
        positions = pd.Series(np.arange(len(cache_ids)), index=cache_ids).reindex(target_ids)
        if positions.isna().any():
            missing = int(positions.isna().sum())
            raise RuntimeError(f"{model}/{path.stem}: missing {missing} target cells")
        rows = positions.to_numpy(dtype=np.int64)
        values = np.asarray(cached[key][rows], dtype=np.float32)
    if len(values) != len(target_ids):
        raise RuntimeError(f"{model}/{path.stem}: selected row mismatch")
    return values


def load_cohort_pool(
    cohort: Cohort,
    inputs: Inputs,
    *,
    min_cells_per_state: int,
    min_cells_per_donor: int,
    min_donors_per_state: int,
    min_cell_types: int,
) -> tuple[ad.AnnData, pd.DataFrame, list[str], np.ndarray]:
    """Load the outcome-blind common pool and identify supported cell types."""
    adata = ad.read_h5ad(inputs.samples_dir / f"{cohort.dataset_id}.h5ad")
    manifest_ids = inputs.manifest_ids(cohort.dataset_id)
    obs = adata.obs.copy()
    for column in OBS_COLUMNS:
        if column not in obs:
            raise KeyError(f"{cohort.dataset_id}: missing obs[{column!r}]")
        obs[column] = obs[column].astype("string").fillna("unknown").astype(str)
    mask = (
        obs[SOMA_KEY].isin(set(manifest_ids))
        & (obs[ASSAY_KEY] == cohort.assay)
        & (obs[TISSUE_KEY] == cohort.tissue)
        & obs[STATE_KEY].isin((cohort.reference, cohort.disease))
    )
    eligible = strict_cell_types(
        obs.loc[mask].copy(),
        assay=cohort.assay,
        tissue=cohort.tissue,
        states=(cohort.reference, cohort.disease),
        min_cells_per_state=min_cells_per_state,
        min_cells_per_donor=min_cells_per_donor,
        min_donors_per_state=min_donors_per_state,
    )
    if len(eligible) < min_cell_types:
        raise RuntimeError(
            f"{cohort.name}: {len(eligible)} eligible types, requires {min_cell_types}"
        )
    selected_positions = np.flatnonzero(mask.to_numpy() & obs[CELL_TYPE_KEY].isin(eligible))
    pool = adata[selected_positions].copy()
    pool_obs = obs.iloc[selected_positions].reset_index(drop=True)
    pool.obs = pool_obs.copy()
    target_ids = pool_obs[SOMA_KEY].to_numpy(dtype=str)
    if len(np.unique(target_ids)) != len(target_ids):
        raise RuntimeError(f"{cohort.name}: duplicate target IDs")
    return pool, pool_obs, eligible, manifest_ids


def balanced_repeat_indices(
    obs: pd.DataFrame,
    cohort: Cohort,
    cell_types: list[str],
    *,
    seed: int,
    donors_per_state: int,
    cells_per_donor: int,
) -> np.ndarray:
    """Select equal donors and cells for every cell-type-by-state stratum."""
    rng = np.random.default_rng(stable_seed(seed, cohort.dataset_id))
    selected: list[np.ndarray] = []
    for cell_type in cell_types:
        for state in (cohort.reference, cohort.disease):
            mask = (obs[CELL_TYPE_KEY] == cell_type) & (obs[STATE_KEY] == state)
            group = obs.loc[mask]
            donor_groups = {
                str(donor): indices.to_numpy(dtype=np.int64)
                for donor, indices in group.groupby(DONOR_KEY, observed=True).groups.items()
                if len(indices) >= cells_per_donor
            }
            donors = np.asarray(sorted(donor_groups), dtype=str)
            if len(donors) < donors_per_state:
                raise RuntimeError(f"{cohort.name}/{cell_type}/{state}: insufficient donors")
            chosen_donors = rng.choice(donors, size=donors_per_state, replace=False)
            for donor in chosen_donors:
                selected.append(
                    rng.choice(
                        donor_groups[str(donor)],
                        size=cells_per_donor,
                        replace=False,
                    ).astype(np.int64)
                )
    indices = np.sort(np.concatenate(selected))
    expected = len(cell_types) * 2 * donors_per_state * cells_per_donor
    if len(indices) != expected or len(np.unique(indices)) != expected:
        raise RuntimeError(f"{cohort.name}: repeat selection size mismatch")
    return indices


def _balanced_reference_indices(
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample an equal number of cross-donor references per target label."""
    candidate_labels = labels[candidates]
    values, counts = np.unique(candidate_labels, return_counts=True)
    if len(values) < 2:
        raise ValueError("Balanced retrieval requires at least two target labels")
    target_count = int(counts.min())
    if target_count < 1:
        raise ValueError("Balanced retrieval found an empty target label")
    selected = [
        rng.choice(candidates[candidate_labels == value], size=target_count, replace=False)
        for value in values
    ]
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _nearest_labels(
    normalized: np.ndarray,
    query: int,
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    k: int,
) -> tuple[str, float]:
    n_neighbors = min(k, len(candidates))
    if n_neighbors < 1:
        raise ValueError("No candidate neighbors")
    similarity = normalized[candidates] @ normalized[query]
    local = np.argpartition(-similarity, n_neighbors - 1)[:n_neighbors]
    neighbor_labels = labels[candidates[local]]
    values, counts = np.unique(neighbor_labels, return_counts=True)
    truth = str(labels[query])
    return str(values[np.argmax(counts)]), float(np.mean(neighbor_labels == truth))


def biological_scores(
    embeddings: np.ndarray,
    obs: pd.DataFrame,
    *,
    k: int,
    seed: int,
) -> dict[str, float]:
    """Score label-balanced cross-donor identity and state retrieval."""
    normalized = normalized_embeddings(embeddings)
    rng = np.random.default_rng(seed)
    cell_types = obs[CELL_TYPE_KEY].to_numpy(dtype=str)
    states = obs[STATE_KEY].to_numpy(dtype=str)
    donors = obs[DONOR_KEY].to_numpy(dtype=str)

    identity_ba: list[float] = []
    identity_purity: list[float] = []
    n_types = len(np.unique(cell_types))
    for state in sorted(np.unique(states)):
        indices = np.flatnonzero(states == state)
        truth: list[str] = []
        predictions: list[str] = []
        purities: list[float] = []
        for query in indices:
            candidates = indices[donors[indices] != donors[query]]
            candidates = _balanced_reference_indices(candidates, cell_types, rng=rng)
            prediction, purity = _nearest_labels(
                normalized, int(query), candidates, cell_types, k=k
            )
            truth.append(str(cell_types[query]))
            predictions.append(prediction)
            purities.append(purity)
        identity_ba.append(float(balanced_accuracy_score(truth, predictions)))
        type_values = np.asarray(truth)
        purity_values = np.asarray(purities)
        identity_purity.append(
            float(
                np.mean(
                    [
                        purity_values[type_values == value].mean()
                        for value in np.unique(type_values)
                    ]
                )
            )
        )

    state_ba: list[float] = []
    state_accuracy: list[float] = []
    for cell_type in sorted(np.unique(cell_types)):
        indices = np.flatnonzero(cell_types == cell_type)
        truth = []
        predictions = []
        for query in indices:
            candidates = indices[donors[indices] != donors[query]]
            candidates = _balanced_reference_indices(candidates, states, rng=rng)
            prediction, _ = _nearest_labels(normalized, int(query), candidates, states, k=k)
            truth.append(str(states[query]))
            predictions.append(prediction)
        state_ba.append(float(balanced_accuracy_score(truth, predictions)))
        state_accuracy.append(float(accuracy_score(truth, predictions)))

    return {
        "identity_ba_minus_chance": float(np.mean(identity_ba) - 1.0 / n_types),
        "identity_purity_macro": float(np.mean(identity_purity)),
        "state_balanced_accuracy": float(np.mean(state_ba)),
        "state_accuracy": float(np.mean(state_accuracy)),
    }


def context_scores(embeddings: np.ndarray, obs: pd.DataFrame) -> dict[str, float]:
    """Score donor invariance after conditioning on cell type and disease state."""
    normalized = normalized_embeddings(embeddings)
    cell_types = obs[CELL_TYPE_KEY].to_numpy(dtype=str)
    states = obs[STATE_KEY].to_numpy(dtype=str)
    donors = obs[DONOR_KEY].to_numpy(dtype=str)
    biology = np.asarray(
        [f"{cell_type}||{state}" for cell_type, state in zip(cell_types, states)],
        dtype=str,
    )
    residual_values: list[float] = []
    local_values: list[float] = []
    for label in sorted(np.unique(biology)):
        indices = np.flatnonzero(biology == label)
        residual_values.append(residual_donor_variance(normalized[indices], donors[indices]))
        similarity = normalized[indices] @ normalized[indices].T
        local_values.append(
            local_donor_mixing_from_similarity(
                similarity,
                donors[indices],
                k=min(10, len(indices) - 1),
            )
        )
    return {
        "donor_bras_conditional": float(
            bras(
                normalized,
                biology,
                donors,
                chunk_size=512,
                metric="cosine",
                between_cluster_distances="mean_other",
            )
        ),
        "residual_donor_invariance": float(1.0 - np.mean(residual_values)),
        "local_donor_mixing": float(np.mean(local_values)),
    }


def log_cp10k_from_library(values: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Normalize selected genes with a full-matrix library size."""
    return np.log1p(
        values.astype(np.float32) * (10_000.0 / np.maximum(library, 1.0))[:, None]
    ).astype(np.float32)


def _matrix_library(matrix: object) -> np.ndarray:
    sums = matrix.sum(axis=1)  # type: ignore[attr-defined]
    if sp.issparse(sums):
        sums = sums.A1
    return np.asarray(sums, dtype=np.float32).reshape(-1)


def prepare_expression(
    pool: ad.AnnData,
    cohort: Cohort,
    inputs: Inputs,
    models: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """Align raw and reconstructed expression on cells and the common genes.

    The common genes are the intersection of the raw features and every reconstruction
    cache, restricted to the paper's gene universe (``GeneUniverse``) and then to genes
    detected in at least ``min_sample_detection_rate`` of the pooled cells. Raw counts are
    normalised with the full-cell library size; each reconstruction with the sum over all
    genes of its cache.
    """
    target_ids = pool.obs[SOMA_KEY].to_numpy().astype(np.int64)
    recon_payloads: dict[str, tuple[np.ndarray, list[str], np.ndarray]] = {}
    for model in models:
        recon_payloads[model] = cache_payload(
            cache_path(inputs.recon_dirs[model], cohort.dataset_id)
        )
    gene_sets = [set(map(normalize_gene, genes)) for _, genes, _ in recon_payloads.values()]
    raw_lookup = feature_lookup(pool)
    common = inputs.universe.restrict(
        cohort.dataset_id, set(raw_lookup).intersection(*gene_sets)
    )
    common_genes = sorted(common)
    if len(common_genes) < 100:
        raise RuntimeError(f"{cohort.name}: only {len(common_genes)} common genes")
    raw_columns = [raw_lookup[gene] for gene in common_genes]
    raw_counts = dense_columns(pool.X, raw_columns)
    if inputs.universe.min_sample_detection_rate > 0:
        detected = (
            np.asarray(raw_counts) > 0
        ).mean(axis=0) >= inputs.universe.min_sample_detection_rate
        common_genes = [g for g, keep in zip(common_genes, detected) if keep]
        raw_counts = np.asarray(raw_counts)[:, detected]
    raw_log = log_cp10k_from_library(raw_counts, _matrix_library(pool.X))

    recon_log: dict[str, np.ndarray] = {}
    for model, (recon, genes, soma_ids) in recon_payloads.items():
        row_lookup = pd.Series(np.arange(len(soma_ids)), index=soma_ids.astype(np.int64))
        rows = row_lookup.reindex(target_ids)
        if rows.isna().any():
            raise RuntimeError(f"{cohort.name}/{model}: reconstruction cells missing")
        gene_lookup = {normalize_gene(gene): index for index, gene in enumerate(genes)}
        columns = [gene_lookup[gene] for gene in common_genes]
        aligned = recon[rows.to_numpy(dtype=np.int64)]
        selected = aligned[:, columns]
        recon_log[model] = log_cp10k_from_library(selected, aligned.sum(axis=1))
    return raw_counts, raw_log, recon_log, common_genes


def _topk(values: np.ndarray, k: int) -> np.ndarray:
    if len(values) <= k:
        return np.arange(len(values))
    return np.argpartition(-np.abs(values), k - 1)[:k]


def expression_scores(
    raw_counts: np.ndarray,
    raw_log: np.ndarray,
    recon_log: dict[str, np.ndarray],
    obs: pd.DataFrame,
    *,
    disease: str,
    reference: str,
    top_k: int,
    min_detection_rate: float,
) -> dict[str, dict[str, float]]:
    """Compute within-cell-type disease logFC concordance for each model."""
    rows: dict[str, dict[str, list[float]]] = {
        model: {"logfc_spearman": [], "deg_jaccard": [], "deg_sign": []} for model in recon_log
    }
    for cell_type in sorted(obs[CELL_TYPE_KEY].unique()):
        pos = (obs[CELL_TYPE_KEY].to_numpy(dtype=str) == cell_type) & (
            obs[STATE_KEY].to_numpy(dtype=str) == disease
        )
        neg = (obs[CELL_TYPE_KEY].to_numpy(dtype=str) == cell_type) & (
            obs[STATE_KEY].to_numpy(dtype=str) == reference
        )
        detected = (
            (raw_counts[pos] > 0).mean(axis=0) + (raw_counts[neg] > 0).mean(axis=0)
        ) > min_detection_rate
        raw_logfc = raw_log[pos].mean(axis=0) - raw_log[neg].mean(axis=0)
        keep = detected & np.isfinite(raw_logfc)
        if int(keep.sum()) < max(20, top_k):
            raise RuntimeError(f"{cell_type}: insufficient detected genes")
        raw_eval = raw_logfc[keep]
        for model, values in recon_log.items():
            recon_logfc = values[pos].mean(axis=0) - values[neg].mean(axis=0)
            recon_eval = recon_logfc[keep]
            valid = np.isfinite(raw_eval) & np.isfinite(recon_eval)
            rho = float(spearmanr(raw_eval[valid], recon_eval[valid]).statistic)
            recon_top = _topk(recon_eval[valid], top_k)
            raw_top_valid = _topk(raw_eval[valid], top_k)
            intersection = len(set(raw_top_valid) & set(recon_top))
            union = len(set(raw_top_valid) | set(recon_top))
            rows[model]["logfc_spearman"].append(rho)
            rows[model]["deg_jaccard"].append(intersection / union)
            rows[model]["deg_sign"].append(
                float(
                    np.mean(
                        np.sign(raw_eval[valid][raw_top_valid])
                        == np.sign(recon_eval[valid][raw_top_valid])
                    )
                )
            )
    return {
        model: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for model, metrics in rows.items()
    }


def pareto_frontier(values: np.ndarray) -> np.ndarray:
    """Return a mask for non-dominated rows when every objective is maximized."""
    array = np.asarray(values, dtype=np.float64)
    frontier = np.ones(len(array), dtype=bool)
    for index, point in enumerate(array):
        dominates = np.all(array >= point, axis=1) & np.any(array > point, axis=1)
        dominates[index] = False
        frontier[index] = not np.any(dominates)
    return frontier


def write_outputs(
    repeat_frame: pd.DataFrame,
    embedding_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    output_dir: Path,
    *,
    three_axis_models: tuple[str, ...],
    reference_model: str,
) -> None:
    """Write repeat tables, cohort and equal-cohort summaries, paired comparisons, Pareto."""
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_frame.to_csv(output_dir / "repeat_metrics.csv", index=False)
    embedding_frame.to_csv(output_dir / "embedding_repeat_metrics.csv", index=False)
    (
        embedding_frame.groupby("model", observed=True)
        .mean(numeric_only=True)
        .reset_index()
        .to_csv(output_dir / "embedding_equal_cohort_summary.csv", index=False)
    )
    audit_frame.to_csv(output_dir / "cohort_audit.csv", index=False)

    if embedding_frame["model"].nunique() >= 3:
        rank_rows: list[dict[str, object]] = []
        for (cohort, repeat), group in embedding_frame.groupby(
            ["cohort", "repeat"], observed=True
        ):
            for metric in CONTEXT_METRICS:
                rank_rows.append(
                    {
                        "cohort": cohort,
                        "repeat": int(repeat),
                        "context_metric": metric,
                        "identity_rank_rho": float(
                            spearmanr(group["identity_ba_minus_chance"], group[metric]).statistic
                        ),
                        "state_rank_rho": float(
                            spearmanr(group["state_balanced_accuracy"], group[metric]).statistic
                        ),
                    }
                )
        rank_frame = pd.DataFrame(rank_rows)
        rank_frame.to_csv(output_dir / "embedding_rank_relationships.csv", index=False)
        (
            rank_frame.groupby("context_metric", observed=True)
            .agg(
                identity_mean_rho=("identity_rank_rho", "mean"),
                identity_median_rho=("identity_rank_rho", "median"),
                identity_fraction_negative=(
                    "identity_rank_rho",
                    lambda x: float((x < 0).mean()),
                ),
                state_mean_rho=("state_rank_rho", "mean"),
                state_median_rho=("state_rank_rho", "median"),
                state_fraction_negative=("state_rank_rho", lambda x: float((x < 0).mean())),
            )
            .reset_index()
            .to_csv(output_dir / "embedding_rank_relationship_summary.csv", index=False)
        )

    cohort_means = (
        repeat_frame.groupby(["cohort", "dataset_id", "model"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    cohort_means.to_csv(output_dir / "cohort_means.csv", index=False)
    equal_cohort = (
        cohort_means.groupby("model", observed=True)
        .agg(
            biological_identity=("biological_identity", "mean"),
            biological_state=("biological_state", "mean"),
            context_invariance=("context_invariance", "mean"),
            donor_bras=("donor_bras", "mean"),
            local_donor_mixing=("local_donor_mixing", "mean"),
            expression_fidelity=("expression_fidelity", "mean"),
            deg_jaccard=("deg_jaccard", "mean"),
            deg_sign=("deg_sign", "mean"),
            n_cohorts=("cohort", "nunique"),
        )
        .reset_index()
    )
    equal_cohort["pareto_frontier"] = pareto_frontier(equal_cohort[PARETO_AXES].to_numpy())
    equal_cohort.to_csv(output_dir / "equal_cohort_summary.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    baselines = [model for model in three_axis_models if model != reference_model]
    if reference_model in three_axis_models:
        for baseline in baselines:
            paired = repeat_frame.loc[
                repeat_frame["model"].isin((reference_model, baseline))
            ].pivot(index=["cohort", "repeat"], columns="model", values=list(COMPARISON_METRICS))
            for metric in COMPARISON_METRICS:
                target = paired[(metric, reference_model)].to_numpy()
                comparator = paired[(metric, baseline)].to_numpy()
                close = np.isclose(target, comparator, rtol=1e-7, atol=1e-9)
                comparison_rows.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "n_pairs": len(target),
                        "wins": int(np.sum((target > comparator) & ~close)),
                        "ties": int(np.sum(close)),
                        "losses": int(np.sum((target < comparator) & ~close)),
                        "mean_paired_difference": float(np.mean(target - comparator)),
                    }
                )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "paired_model_comparisons.csv", index=False)

    repeat_means = (
        repeat_frame.groupby(["repeat", "model"], observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    pareto_rows: list[dict[str, object]] = []
    for repeat, group in repeat_means.groupby("repeat", observed=True):
        repeat_frontier = pareto_frontier(group[PARETO_AXES].to_numpy())
        for model, is_frontier in zip(group["model"], repeat_frontier):
            pareto_rows.append(
                {"repeat": int(repeat), "model": model, "pareto_frontier": bool(is_frontier)}
            )
    pareto_frame = pd.DataFrame(pareto_rows)
    pareto_frame.to_csv(output_dir / "repeat_pareto.csv", index=False)
    (
        pareto_frame.groupby("model", observed=True)["pareto_frontier"]
        .mean()
        .rename("pareto_frequency")
        .reset_index()
        .to_csv(output_dir / "pareto_frequency.csv", index=False)
    )


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI options shared with ``pathway.py``: samples, caches, gene universe, protocol."""
    group = parser.add_argument_group("inputs")
    group.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Sampled datasets written by experiments/prepare_samples.py",
    )
    group.add_argument(
        "--embedding-cache",
        action="append",
        metavar="NAME=DIR",
        help="Embedding cache directory of one model (<dataset>.npz with 'embeddings' or "
        f"'embedding' and, ideally, 'soma_joinid'); default {MODEL_NAME}=<release cache>",
    )
    group.add_argument(
        "--recon-cache",
        action="append",
        metavar="NAME=DIR",
        help="Reconstruction cache directory of one model (<dataset>.npz with 'recon', "
        f"'gene_names', 'soma_joinid'); default {MODEL_NAME}=<release cache>",
    )
    group.add_argument(
        "--manifest-dir",
        type=Path,
        default=None,
        help="Per-dataset <dataset>.npz with the 'soma_joinid' of the embedded cells, for "
        "embedding caches that carry no soma_joinid themselves (release caches do)",
    )
    group.add_argument(
        "--three-axis-models",
        nargs="+",
        default=None,
        help="Models scored on all three demands (default: every --recon-cache model)",
    )
    add_gene_universe_arguments(parser)
    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--seed-start", type=int, default=42)
    protocol.add_argument("--repeats", type=int, default=20)
    protocol.add_argument("--min-cell-types", type=int, default=3)
    protocol.add_argument("--min-cells-per-state", type=int, default=40)
    protocol.add_argument("--min-cells-per-donor", type=int, default=10)
    protocol.add_argument("--min-donors-per-state", type=int, default=2)
    protocol.add_argument("--donors-per-state", type=int, default=2)
    protocol.add_argument("--cells-per-donor", type=int, default=10)
    protocol.add_argument("--min-detection-rate", type=float, default=0.02)
    protocol.add_argument(
        "--cohorts",
        nargs="+",
        choices=[cohort.name for cohort in COHORTS],
        default=None,
        help="Subset of the five cohorts (default: all)",
    )


def resolve_inputs(args: argparse.Namespace) -> tuple[Inputs, tuple[str, ...], tuple[str, ...]]:
    """Resolve caches and model lists from the parsed arguments."""
    embedding_dirs = parse_model_dirs(args.embedding_cache, {MODEL_NAME: DEFAULT_EMBEDDING_DIR})
    recon_dirs = parse_model_dirs(args.recon_cache, {MODEL_NAME: DEFAULT_RECON_DIR})
    three_axis = tuple(args.three_axis_models or recon_dirs)
    for model in three_axis:
        if model not in recon_dirs or model not in embedding_dirs:
            raise SystemExit(f"three-axis model {model!r} needs both an embedding and a recon cache")
    inputs = Inputs(
        samples_dir=args.samples_dir,
        manifest_dir=args.manifest_dir,
        embedding_dirs=embedding_dirs,
        recon_dirs=recon_dirs,
        universe=GeneUniverse(args),
    )
    return inputs, tuple(embedding_dirs), three_axis


def selected_cohorts(args: argparse.Namespace) -> tuple[Cohort, ...]:
    """Cohorts to evaluate, in the fixed panel order."""
    if not args.cohorts:
        return COHORTS
    return tuple(cohort for cohort in COHORTS if cohort.name in set(args.cohorts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    add_input_arguments(parser)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--reference-model",
        default=MODEL_NAME,
        help="Model of the paired win/tie/loss comparisons against every other model",
    )
    parser.add_argument(
        "--candidate-pairs",
        type=Path,
        default=DEFAULT_CANDIDATE_PAIRS,
        help="candidate_pairs.csv of the metadata audit (experiments.table2_joint.cohort_audit)",
    )
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument(
        "--no-selection-audit",
        action="store_true",
        help="Skip re-deriving the cohort panel from all held-out datasets "
        "(needs every sample and the first embedding cache of the candidate datasets)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs, embedding_models, three_axis_models = resolve_inputs(args)
    cohorts = selected_cohorts(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "cohorts": [asdict(cohort) for cohort in cohorts],
        "embedding_models": list(embedding_models),
        "three_axis_models": list(three_axis_models),
        "embedding_caches": {k: describe_path(v) for k, v in inputs.embedding_dirs.items()},
        "recon_caches": {k: describe_path(v) for k, v in inputs.recon_dirs.items()},
        "samples_dir": describe_path(inputs.samples_dir),
        "manifest_dir": describe_path(inputs.manifest_dir),
        **inputs.universe.config(),
        "retrieval_reference_pool": "equal cells per target label after excluding the query donor",
        **{
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "output_dir",
                "embedding_cache",
                "recon_cache",
                "samples_dir",
                "manifest_dir",
                "genelist_dir",
                "no_genelist",
                "excluded_genes",
                "ignore_excluded_genes",
                "min_sample_detection_rate",
                "three_axis_models",
                "cohorts",
            }
        },
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not args.no_selection_audit:
        dataset_ids = read_ids(args.dataset_ids_file)
        selection_audit = audit_all_datasets(args, inputs, dataset_ids)
        selection_audit.to_csv(args.output_dir / "candidate_audit_89.csv", index=False)

    rows: list[dict[str, object]] = []
    embedding_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for cohort in cohorts:
        print(f"\n=== {cohort.name}: {cohort.dataset_id} ===", flush=True)
        pool, obs, cell_types, manifest_ids = load_cohort_pool(
            cohort,
            inputs,
            min_cells_per_state=args.min_cells_per_state,
            min_cells_per_donor=args.min_cells_per_donor,
            min_donors_per_state=args.min_donors_per_state,
            min_cell_types=args.min_cell_types,
        )
        target_ids = obs[SOMA_KEY].to_numpy(dtype=str)
        embedding_by_model = {
            model: aligned_embeddings(
                model,
                cache_path(inputs.embedding_dirs[model], cohort.dataset_id),
                manifest_ids,
                target_ids,
            )
            for model in embedding_models
        }
        raw_counts, raw_log, recon_log, common_genes = prepare_expression(
            pool, cohort, inputs, three_axis_models
        )
        audit_rows.append(
            {
                "cohort": cohort.name,
                "dataset_id": cohort.dataset_id,
                "assay": cohort.assay,
                "tissue": cohort.tissue,
                "disease": cohort.disease,
                "reference": cohort.reference,
                "n_common_pool_cells": len(obs),
                "n_eligible_cell_types": len(cell_types),
                "eligible_cell_types": "; ".join(cell_types),
                "n_common_genes": len(common_genes),
            }
        )
        for repeat in range(args.repeats):
            seed = args.seed_start + repeat
            indices = balanced_repeat_indices(
                obs,
                cohort,
                cell_types,
                seed=seed,
                donors_per_state=args.donors_per_state,
                cells_per_donor=args.cells_per_donor,
            )
            selected_obs = obs.iloc[indices].reset_index(drop=True)
            expression = expression_scores(
                raw_counts[indices],
                raw_log[indices],
                {model: values[indices] for model, values in recon_log.items()},
                selected_obs,
                disease=cohort.disease,
                reference=cohort.reference,
                top_k=args.top_k,
                min_detection_rate=args.min_detection_rate,
            )
            for model in embedding_models:
                biological = biological_scores(
                    embedding_by_model[model][indices],
                    selected_obs,
                    k=args.neighbor_k,
                    seed=seed,
                )
                context = context_scores(embedding_by_model[model][indices], selected_obs)
                embedding_rows.append(
                    {
                        "cohort": cohort.name,
                        "dataset_id": cohort.dataset_id,
                        "repeat": repeat,
                        "seed": seed,
                        "model": model,
                        "n_cells": len(indices),
                        "n_cell_types": len(cell_types),
                        **biological,
                        **context,
                    }
                )
                if model not in three_axis_models:
                    continue
                rows.append(
                    {
                        "cohort": cohort.name,
                        "dataset_id": cohort.dataset_id,
                        "repeat": repeat,
                        "seed": seed,
                        "model": model,
                        "n_cells": len(indices),
                        "n_cell_types": len(cell_types),
                        "biological_identity": biological["identity_ba_minus_chance"],
                        "identity_purity": biological["identity_purity_macro"],
                        "biological_state": biological["state_balanced_accuracy"],
                        "state_accuracy": biological["state_accuracy"],
                        "context_invariance": context["residual_donor_invariance"],
                        "donor_bras": context["donor_bras_conditional"],
                        "local_donor_mixing": context["local_donor_mixing"],
                        "expression_fidelity": expression[model]["logfc_spearman"],
                        "deg_jaccard": expression[model]["deg_jaccard"],
                        "deg_sign": expression[model]["deg_sign"],
                    }
                )
            print(f"  repeat {repeat + 1:02d}/{args.repeats}: {len(indices)} cells", flush=True)
    repeat_frame = pd.DataFrame(rows)
    embedding_frame = pd.DataFrame(embedding_rows)
    required = len(cohorts) * args.repeats * len(three_axis_models)
    embedding_required = len(cohorts) * args.repeats * len(embedding_models)
    if len(repeat_frame) != required or len(embedding_frame) != embedding_required:
        raise RuntimeError(
            f"Invalid row counts: three-axis={len(repeat_frame)}/{required}, "
            f"embedding={len(embedding_frame)}/{embedding_required}"
        )
    if not np.isfinite(repeat_frame.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError("Three-axis result table contains non-finite values")
    if not np.isfinite(embedding_frame.select_dtypes(include=[np.number])).all().all():
        raise RuntimeError("Embedding result table contains non-finite values")
    write_outputs(
        repeat_frame,
        embedding_frame,
        pd.DataFrame(audit_rows),
        args.output_dir,
        three_axis_models=three_axis_models,
        reference_model=args.reference_model,
    )
    print(f"Done: {len(repeat_frame)} rows across {len(cohorts)} cohorts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
