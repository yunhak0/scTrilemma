"""Donor-supported disease panel for the supplementary expression-fidelity analysis (Tables 7, 18).

The screen is outcome blind. It operates on the staged seed-42 samples of the 89 held-out
datasets (``experiments/prepare_samples.py``) and retains one normal-versus-disease comparison
per dataset. A cell type is supported when both states contain at least 40 cells and at least
two donors contribute at least 10 cells each; datasets need at least three supported cell
types; the winning condition per dataset must be a named disease (``injury`` is excluded).
Up to six of the best-supported cell-type contrasts are evaluated per selected cohort with
the DEG (``deg_concordance``) and pathway (``pathway_concordance``, Enrichr) scorers on the
paper's common gene universe.

    pixi run python -m experiments.table2_joint.rq4_panel all
    pixi run python -m experiments.table2_joint.rq4_panel deg      # no network needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from experiments.common import normalize_gene, read_ids
from experiments.table2_joint._common import (
    CELL_TYPE_KEY,
    DEFAULT_IDS_FILE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RECON_DIR,
    DEFAULT_SAMPLES_DIR,
    DONOR_KEY,
    MODEL_NAME,
    SOMA_KEY,
    STATE_KEY,
    GeneUniverse,
    add_gene_universe_arguments,
    describe_path,
    display_name,
    normalized_obs,
    parse_model_dirs,
)
from experiments.table2_joint.deg_concordance import (
    Contrast,
    ReconResult,
    align_common_matrices,
    evaluate_contrast,
    load_recon_cache,
)
from experiments.table2_joint.pathway_concordance import (
    DEFAULT_LIBRARIES,
    EnrichrClient,
    add_enrichr_arguments,
    evaluate_pathway_contrast,
    gene_symbol_lookup,
    resolve_libraries,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "rq4_panel"

NON_DISEASE_LABELS = {"injury"}
EXPECTED_SUPPORT_PREFIXES = {
    "203025fe",
    "30c2a6fd",
    "829a3cd1",
    "84cfa5aa",
    "867757c1",
    "acd544d0",
    "e3ed2ba4",
    "e51bae9a",
}
EXPECTED_DISEASE_PREFIXES = EXPECTED_SUPPORT_PREFIXES - {"acd544d0"}

MIN_CELLS_PER_STATE = 40
MIN_CELLS_PER_DONOR = 10
MIN_DONORS_PER_STATE = 2
MIN_CELL_TYPES = 3
MAX_CONTRASTS = 6
DEG_MIN_DETECTION_RATE = 0.02
DEG_TOP_K = 100
PATHWAY_TOP_GENES = 100
PATHWAY_TOP_TERMS = 10
OBS_COLUMNS = [SOMA_KEY, CELL_TYPE_KEY, STATE_KEY, DONOR_KEY]


def load_sample(samples_dir: Path, dataset_id: str, *, backed: bool = False) -> ad.AnnData:
    """Read one staged sample and check the metadata columns the panel needs."""
    path = samples_dir / f"{dataset_id}.h5ad"
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r" if backed else None)
    missing = [column for column in OBS_COLUMNS if column not in adata.obs]
    if missing:
        if backed:
            adata.file.close()
        raise KeyError(f"{dataset_id}: missing obs columns {missing}")
    if "dataset_id" not in adata.uns:
        adata.uns["dataset_id"] = dataset_id
    return adata


def cell_type_support(
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    normal: str,
    disease: str,
) -> pd.DataFrame:
    """Per-cell-type cell and donor support of one normal-versus-condition pair."""
    pair = obs.loc[obs[STATE_KEY].isin((normal, disease))]
    rows: list[dict[str, object]] = []
    for cell_type, group in pair.groupby(CELL_TYPE_KEY, observed=True, sort=True):
        state_values: dict[str, dict[str, int]] = {}
        for state in (normal, disease):
            state_group = group.loc[group[STATE_KEY] == state]
            donor_counts = state_group[DONOR_KEY].value_counts()
            state_values[state] = {
                "cells": int(len(state_group)),
                "supported_donors": int((donor_counts >= MIN_CELLS_PER_DONOR).sum()),
                "all_donors": int(donor_counts.size),
            }
        normal_values = state_values[normal]
        disease_values = state_values[disease]
        eligible = (
            normal_values["cells"] >= MIN_CELLS_PER_STATE
            and disease_values["cells"] >= MIN_CELLS_PER_STATE
            and normal_values["supported_donors"] >= MIN_DONORS_PER_STATE
            and disease_values["supported_donors"] >= MIN_DONORS_PER_STATE
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": dataset_id[:8],
                "normal_label": normal,
                "disease_label": disease,
                "cell_type": str(cell_type),
                "normal_cells": normal_values["cells"],
                "disease_cells": disease_values["cells"],
                "minimum_state_cells": min(normal_values["cells"], disease_values["cells"]),
                "total_cells": normal_values["cells"] + disease_values["cells"],
                "normal_supported_donors": normal_values["supported_donors"],
                "disease_supported_donors": disease_values["supported_donors"],
                "minimum_supported_donors": min(
                    normal_values["supported_donors"],
                    disease_values["supported_donors"],
                ),
                "total_supported_donors": (
                    normal_values["supported_donors"] + disease_values["supported_donors"]
                ),
                "eligible": eligible,
            }
        )
    return pd.DataFrame(rows)


def screen_panel(
    samples_dir: Path,
    dataset_ids: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen every held-out sample and freeze the named-disease cohort panel."""
    candidate_rows: list[dict[str, object]] = []
    support_frames: list[pd.DataFrame] = []
    if len(dataset_ids) != 89:
        raise RuntimeError(f"Expected 89 held-out datasets, found {len(dataset_ids)}")

    for dataset_id in dataset_ids:
        adata = load_sample(samples_dir, dataset_id, backed=True)
        obs = normalized_obs(adata.obs, OBS_COLUMNS)
        adata.file.close()
        normal_labels = sorted(
            label for label in obs[STATE_KEY].unique() if label.strip().lower() == "normal"
        )
        if not normal_labels:
            continue
        normal = normal_labels[0]
        disease_labels = sorted(
            label
            for label in obs[STATE_KEY].unique()
            if label != normal and label.strip().lower() not in {"", "unknown", "nan", "none"}
        )
        for disease in disease_labels:
            support = cell_type_support(obs, dataset_id=dataset_id, normal=normal, disease=disease)
            support_frames.append(support)
            eligible = support.loc[support["eligible"]].copy()
            pair = obs.loc[obs[STATE_KEY].isin((normal, disease))]
            candidate_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_prefix": dataset_id[:8],
                    "normal_label": normal,
                    "disease_label": disease,
                    "n_eligible_cell_types": int(len(eligible)),
                    "minimum_state_support_sum": int(eligible["minimum_state_cells"].sum()),
                    "total_cell_type_support": int(eligible["total_cells"].sum()),
                    "n_donors_pair": int(pair[DONOR_KEY].nunique()),
                    "n_cells_pair": int(len(pair)),
                    "passes_support": len(eligible) >= MIN_CELL_TYPES,
                    "named_disease": disease.strip().lower() not in NON_DISEASE_LABELS,
                    "eligible_cell_types": "; ".join(eligible["cell_type"].astype(str)),
                }
            )

    candidates = pd.DataFrame(candidate_rows)
    supports = pd.concat(support_frames, ignore_index=True)
    candidates["dataset_winner"] = False
    candidates["selected"] = False
    candidates["decision"] = "Insufficient donor-supported cell types"

    passing = candidates.loc[candidates["passes_support"]].copy()
    passing = passing.sort_values(
        [
            "dataset_id",
            "n_eligible_cell_types",
            "minimum_state_support_sum",
            "total_cell_type_support",
            "n_donors_pair",
            "n_cells_pair",
            "disease_label",
        ],
        ascending=[True, False, False, False, False, False, True],
    )
    winners = passing.groupby("dataset_id", sort=True, as_index=False).first()
    winner_keys = set(zip(winners["dataset_id"], winners["disease_label"], strict=True))
    winner_mask = pd.Series(
        [
            (dataset_id, disease) in winner_keys
            for dataset_id, disease in zip(
                candidates["dataset_id"], candidates["disease_label"], strict=True
            )
        ],
        index=candidates.index,
    )
    candidates.loc[winner_mask, "dataset_winner"] = True
    named_mask = winner_mask & candidates["named_disease"]
    candidates.loc[named_mask, "selected"] = True
    candidates.loc[winner_mask & ~candidates["named_disease"], "decision"] = (
        "Excluded because the selected condition is not a named disease"
    )
    candidates.loc[named_mask, "decision"] = "Selected"
    nonwinning_pass = candidates["passes_support"] & ~winner_mask
    candidates.loc[nonwinning_pass, "decision"] = "Not selected by within-dataset tie-break"

    selected = candidates.loc[candidates["selected"]].copy()
    selected = selected.sort_values("dataset_prefix").reset_index(drop=True)
    support_prefixes = set(winners["dataset_prefix"])
    selected_prefixes = set(selected["dataset_prefix"])
    if support_prefixes != EXPECTED_SUPPORT_PREFIXES:
        raise RuntimeError(f"Unexpected support-screen datasets: {sorted(support_prefixes)}")
    if selected_prefixes != EXPECTED_DISEASE_PREFIXES:
        raise RuntimeError(f"Unexpected named-disease panel: {sorted(selected_prefixes)}")

    selected_contrast_frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        subset = supports.loc[
            (supports["dataset_id"] == row.dataset_id)
            & (supports["disease_label"] == row.disease_label)
            & supports["eligible"]
        ].copy()
        subset = subset.sort_values(
            [
                "minimum_state_cells",
                "total_cells",
                "minimum_supported_donors",
                "total_supported_donors",
                "cell_type",
            ],
            ascending=[False, False, False, False, True],
        ).head(MAX_CONTRASTS)
        subset.insert(5, "contrast_rank", np.arange(1, len(subset) + 1))
        selected_contrast_frames.append(subset)
    contrasts = pd.concat(selected_contrast_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates.sort_values(
        ["dataset_prefix", "dataset_winner", "disease_label"],
        ascending=[True, False, True],
    ).to_csv(output_dir / "screen_all_normal_condition_contrasts.csv", index=False)
    supports.to_csv(output_dir / "screen_cell_type_support.csv", index=False)
    selected.to_csv(output_dir / "selected_cohorts.csv", index=False)
    contrasts.to_csv(output_dir / "selected_cell_type_contrasts.csv", index=False)
    return candidates, selected, contrasts


def selected_contrasts_for_dataset(
    selected: pd.DataFrame,
    contrast_support: pd.DataFrame,
    dataset_id: str,
) -> list[Contrast]:
    """Contrast objects of the frozen cell-type contrasts of one cohort."""
    cohort = selected.loc[selected["dataset_id"] == dataset_id].iloc[0]
    support = contrast_support.loc[contrast_support["dataset_id"] == dataset_id]
    return [
        Contrast(
            contrast_id=(
                f"{dataset_id}::{row.cell_type}::{cohort.disease_label}_vs_{cohort.normal_label}"
            ),
            dataset_id=dataset_id,
            cell_type=str(row.cell_type),
            positive_label=str(cohort.disease_label),
            negative_label=str(cohort.normal_label),
            positive_n=int(row.disease_cells),
            negative_n=int(row.normal_cells),
        )
        for row in support.itertuples(index=False)
    ]


def load_reconstructions(
    cache_dirs: dict[str, Path],
    dataset_id: str,
    adata: ad.AnnData,
) -> list[ReconResult]:
    """Strictly aligned reconstruction caches of every model, labelled with table names."""
    return [
        load_recon_cache(display_name(model), location, dataset_id, adata)
        for model, location in cache_dirs.items()
    ]


def cohort_summaries(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    stem: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cohort means over contrasts, then equal-cohort means."""
    cohort = (
        frame.groupby(
            ["dataset_id", "dataset_prefix", "disease_label", "model", "metric"],
            observed=True,
        )
        .agg(value=("value", "mean"), n_contrasts=("contrast_id", "nunique"))
        .reset_index()
    )
    equal = (
        cohort.groupby(["model", "metric"], observed=True)
        .agg(value=("value", "mean"), n_cohorts=("dataset_id", "nunique"))
        .reset_index()
    )
    cohort.to_csv(output_dir / f"{stem}_cohort_summary.csv", index=False)
    equal.to_csv(output_dir / f"{stem}_equal_cohort_summary.csv", index=False)
    return cohort, equal


def run_deg(
    output_dir: Path,
    selected: pd.DataFrame,
    contrast_support: pd.DataFrame,
    *,
    samples_dir: Path,
    cache_dirs: dict[str, Path],
    universe: GeneUniverse,
) -> pd.DataFrame:
    """DEG concordance of every selected contrast (``deg_metrics_long.csv`` and summaries)."""
    rows: list[pd.DataFrame] = []
    universe_rows: list[dict[str, object]] = []
    for cohort in selected.itertuples(index=False):
        dataset_id = str(cohort.dataset_id)
        print(f"\n=== DEG {dataset_id[:8]}: {cohort.disease_label} ===", flush=True)
        adata = load_sample(samples_dir, dataset_id)
        reconstructions = load_reconstructions(cache_dirs, dataset_id, adata)
        genes, raw_counts, raw_log, recon_log = align_common_matrices(
            adata, reconstructions, universe=universe
        )
        contrasts = selected_contrasts_for_dataset(selected, contrast_support, dataset_id)
        universe_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_prefix": dataset_id[:8],
                "disease_label": cohort.disease_label,
                "sampled_cells": adata.n_obs,
                "selected_contrasts": len(contrasts),
                "common_genes": len(genes),
            }
        )
        for contrast in contrasts:
            result = evaluate_contrast(
                adata=adata,
                genes=genes,
                raw_counts=raw_counts,
                raw_log=raw_log,
                recon_log_by_model=recon_log,
                contrast=contrast,
                contrast_key=STATE_KEY,
                cell_type_key=CELL_TYPE_KEY,
                min_detection_rate=DEG_MIN_DETECTION_RATE,
                top_k=DEG_TOP_K,
            )
            result_frame = pd.DataFrame(result)
            result_frame.insert(1, "dataset_prefix", dataset_id[:8])
            result_frame.insert(2, "disease_label", cohort.disease_label)
            rows.append(result_frame)
        print(
            f"  cells={adata.n_obs}, contrasts={len(contrasts)}, genes={len(genes)}",
            flush=True,
        )
    frame = pd.concat(rows, ignore_index=True)
    if not np.isfinite(frame["value"]).all():
        raise RuntimeError("DEG output contains non-finite metric values")
    frame.to_csv(output_dir / "deg_metrics_long.csv", index=False)
    pd.DataFrame(universe_rows).to_csv(output_dir / "evaluation_support.csv", index=False)
    cohort_summaries(frame, output_dir=output_dir, stem="deg")
    return frame


def run_pathway(
    output_dir: Path,
    selected: pd.DataFrame,
    contrast_support: pd.DataFrame,
    *,
    samples_dir: Path,
    cache_dirs: dict[str, Path],
    universe: GeneUniverse,
    libraries: list[str],
    client: EnrichrClient,
) -> pd.DataFrame:
    """Pathway concordance of every selected contrast (Enrichr; ``pathway/`` outputs)."""
    pathway_dir = output_dir / "pathway"
    pathway_dir.mkdir(parents=True, exist_ok=True)
    metric_frames: list[pd.DataFrame] = []
    enrichment_frames: list[pd.DataFrame] = []
    for cohort in selected.itertuples(index=False):
        dataset_id = str(cohort.dataset_id)
        print(f"\n=== Pathway {dataset_id[:8]}: {cohort.disease_label} ===", flush=True)
        adata = load_sample(samples_dir, dataset_id)
        reconstructions = load_reconstructions(cache_dirs, dataset_id, adata)
        genes, raw_counts, raw_log, recon_log = align_common_matrices(
            adata, reconstructions, universe=universe
        )
        lookup = gene_symbol_lookup(adata)
        symbols = [lookup.get(normalize_gene(gene), gene) for gene in genes]
        contrasts = selected_contrasts_for_dataset(selected, contrast_support, dataset_id)
        for index, contrast in enumerate(contrasts, start=1):
            print(f"  [{index}/{len(contrasts)}] {contrast.cell_type}", flush=True)
            enrichment, metrics = evaluate_pathway_contrast(
                adata=adata,
                genes=genes,
                symbols=symbols,
                raw_counts=raw_counts,
                raw_log=raw_log,
                recon_log_by_model=recon_log,
                contrast=contrast,
                contrast_key=STATE_KEY,
                cell_type_key=CELL_TYPE_KEY,
                min_detection_rate=DEG_MIN_DETECTION_RATE,
                top_genes=PATHWAY_TOP_GENES,
                top_terms=PATHWAY_TOP_TERMS,
                libraries=libraries,
                client=client,
            )
            metric_frame = pd.DataFrame(metrics)
            metric_frame.insert(1, "dataset_prefix", dataset_id[:8])
            metric_frame.insert(2, "disease_label", cohort.disease_label)
            metric_frames.append(metric_frame)
            enrichment_frames.append(pd.DataFrame(enrichment))
            pd.concat(metric_frames, ignore_index=True).to_csv(
                pathway_dir / "pathway_metrics_long.partial.csv", index=False
            )
    metrics = pd.concat(metric_frames, ignore_index=True)
    enrichment = pd.concat(enrichment_frames, ignore_index=True)
    if not np.isfinite(metrics["value"]).all():
        raise RuntimeError("Pathway output contains non-finite metric values")
    metrics.to_csv(pathway_dir / "pathway_metrics_long.csv", index=False)
    enrichment.to_csv(pathway_dir / "enrichment_long.csv", index=False)
    cohort_summaries(metrics, output_dir=pathway_dir, stem="pathway")
    return metrics


def _rank_counts(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["rank"] = frame.groupby(keys, observed=True)["value"].rank(
        method="min", ascending=False
    )
    counts = (
        frame.groupby("model", observed=True)
        .agg(
            first_place=("rank", lambda values: int((values == 1).sum())),
            second_place=("rank", lambda values: int((values == 2).sum())),
            comparisons=("rank", "size"),
        )
        .reset_index()
    )
    return frame, counts


def combined_summary(output_dir: Path, *, reference_label: str) -> pd.DataFrame:
    """Merge DEG and pathway summaries; rank counts and pairwise wins of the reference model."""
    contrast_columns = [
        "dataset_id",
        "dataset_prefix",
        "disease_label",
        "contrast_id",
        "cell_type",
        "model",
        "metric",
        "value",
    ]
    deg = pd.read_csv(output_dir / "deg_equal_cohort_summary.csv")
    cohort_frames = [pd.read_csv(output_dir / "deg_cohort_summary.csv")]
    deg_long = pd.read_csv(output_dir / "deg_metrics_long.csv")
    contrast_frames = [deg_long[contrast_columns]]
    pathway_summary = output_dir / "pathway/pathway_equal_cohort_summary.csv"
    if pathway_summary.exists():
        combined = pd.concat([deg, pd.read_csv(pathway_summary)], ignore_index=True)
        cohort_frames.append(pd.read_csv(output_dir / "pathway/pathway_cohort_summary.csv"))
        pathway_long = pd.read_csv(output_dir / "pathway/pathway_metrics_long.csv")
        contrast_frames.append(
            pathway_long.groupby(
                contrast_columns[:-1],
                observed=True,
                as_index=False,
            )["value"].mean()
        )
    else:
        print("Pathway outputs not found; summarising the DEG metrics only", flush=True)
        combined = deg
    combined.to_csv(output_dir / "equal_cohort_summary.csv", index=False)

    cohort = pd.concat(cohort_frames, ignore_index=True)
    cohort.to_csv(output_dir / "cohort_metric_summary.csv", index=False)
    _, counts = _rank_counts(cohort, ["dataset_id", "metric"])
    counts.to_csv(output_dir / "cohort_metric_rank_counts.csv", index=False)

    contrast = pd.concat(contrast_frames, ignore_index=True)
    contrast, contrast_counts = _rank_counts(contrast, ["dataset_id", "contrast_id", "metric"])
    contrast.to_csv(output_dir / "contrast_metric_summary.csv", index=False)
    contrast_counts.to_csv(output_dir / "contrast_metric_rank_counts.csv", index=False)

    wide = contrast.pivot(
        index=["dataset_id", "contrast_id", "metric"],
        columns="model",
        values="value",
    ).reset_index()
    pairwise_rows: list[dict[str, object]] = []
    competitors = [str(model) for model in contrast["model"].unique() if model != reference_label]
    if reference_label in wide.columns:
        for competitor in competitors:
            for metric, group in wide.groupby("metric", observed=True, sort=True):
                pairwise_rows.append(
                    {
                        "model": reference_label,
                        "competitor": competitor,
                        "metric": metric,
                        "wins": int((group[reference_label] > group[competitor]).sum()),
                        "comparisons": int(len(group)),
                    }
                )
    pd.DataFrame(pairwise_rows).to_csv(
        output_dir / "contrast_metric_pairwise_wins.csv", index=False
    )
    return combined


def write_config(
    output_dir: Path,
    *,
    command: str,
    cache_dirs: dict[str, Path],
    samples_dir: Path,
    universe: GeneUniverse,
    libraries: list[str],
) -> None:
    """Record the frozen selection rules and inputs of the run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "command": command,
        "population": "89 held-out seed-42 samples of at most 2,500 cells",
        "samples_dir": describe_path(samples_dir),
        "selection": {
            "normal_disease_contrast": True,
            "min_cells_per_state": MIN_CELLS_PER_STATE,
            "min_cells_per_donor": MIN_CELLS_PER_DONOR,
            "min_donors_per_state": MIN_DONORS_PER_STATE,
            "min_supported_cell_types": MIN_CELL_TYPES,
            "max_contrasts_per_cohort": MAX_CONTRASTS,
            "non_disease_labels": sorted(NON_DISEASE_LABELS),
            "disease_tie_break": [
                "n_eligible_cell_types",
                "minimum_state_support_sum",
                "total_cell_type_support",
                "n_donors_pair",
                "n_cells_pair",
                "disease_label_lexical",
            ],
            "cell_type_tie_break": [
                "minimum_state_cells",
                "total_cells",
                "minimum_supported_donors",
                "total_supported_donors",
                "cell_type_lexical",
            ],
        },
        "models": [display_name(model) for model in cache_dirs],
        "model_caches": {display_name(k): describe_path(v) for k, v in cache_dirs.items()},
        **universe.config(),
        "normalization": "independent log1p(CP10K) on the common gene universe",
        "deg": {"min_detection_rate": DEG_MIN_DETECTION_RATE, "top_genes": DEG_TOP_K},
        "pathway": {
            "libraries": libraries,
            "top_genes_per_direction": PATHWAY_TOP_GENES,
            "top_terms": PATHWAY_TOP_TERMS,
        },
        "aggregation": "cell-type contrasts within cohort, then equal cohort weight",
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("screen", "deg", "pathway", "summarize", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument(
        "--recon-cache",
        action="append",
        metavar="NAME=DIR",
        help="Reconstruction cache directory of one model (order preserved); "
        f"default {MODEL_NAME}=<release cache>",
    )
    parser.add_argument(
        "--reference-model",
        default=MODEL_NAME,
        help="Model of the pairwise win counts against every other model",
    )
    add_gene_universe_arguments(parser)
    add_enrichr_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dirs = parse_model_dirs(args.recon_cache, {MODEL_NAME: DEFAULT_RECON_DIR})
    universe = GeneUniverse(args)
    dataset_ids = read_ids(args.dataset_ids_file)
    write_config(
        args.output_dir,
        command=args.command,
        cache_dirs=cache_dirs,
        samples_dir=args.samples_dir,
        universe=universe,
        libraries=list(args.libraries),
    )
    _, selected, contrasts = screen_panel(args.samples_dir, dataset_ids, args.output_dir)
    if args.command == "screen":
        print(selected.to_string(index=False))
        print(f"Selected {len(selected)} named-disease cohorts")
        return 0
    if args.command in {"deg", "all"}:
        run_deg(
            args.output_dir,
            selected,
            contrasts,
            samples_dir=args.samples_dir,
            cache_dirs=cache_dirs,
            universe=universe,
        )
    if args.command in {"pathway", "all"}:
        libraries = resolve_libraries(list(args.libraries or DEFAULT_LIBRARIES))
        client = EnrichrClient(
            cache_path=args.enrichr_cache or args.output_dir / "pathway/enrichr_cache.jsonl",
            sleep=args.sleep,
        )
        print(f"Enrichr libraries: {libraries}; cached requests: {len(client.cache)}", flush=True)
        run_pathway(
            args.output_dir,
            selected,
            contrasts,
            samples_dir=args.samples_dir,
            cache_dirs=cache_dirs,
            universe=universe,
            libraries=libraries,
            client=client,
        )
    if args.command in {"summarize", "all"}:
        summary = combined_summary(
            args.output_dir, reference_label=display_name(args.reference_model)
        )
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
