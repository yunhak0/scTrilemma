"""Pathway-level expression fidelity on the common cohorts (Table 2, last two rows).

For every cohort and eligible cell type the disease logFC of the raw counts and of each
reconstruction is averaged over the same 20 balanced cell selections as ``joint.py`` (genes
kept when detected in at least ``--min-repeat-fraction`` of the repeats). The stabilised
rankings are scored exactly as in ``pathway_concordance``: top-100 up and down gene symbols to
Enrichr, top-10 term Jaccard and combined-score Spearman per library and direction, averaged
over cell types within a cohort and then over cohorts. Pathway metrics are finally added to the
metric-family Pareto sensitivity audit of the joint results (``--joint-dir``).

Enrichr is queried over the network; responses are cached in ``enrichr_cache.jsonl``.

    pixi run python -m experiments.table2_joint.pathway
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import normalize_gene
from experiments.table2_joint._common import (
    CELL_TYPE_KEY,
    DEFAULT_OUTPUT_ROOT,
    STATE_KEY,
    describe_path,
)
from experiments.table2_joint.deg_concordance import Contrast
from experiments.table2_joint.joint import (
    DEFAULT_OUTPUT as DEFAULT_JOINT_DIR,
)
from experiments.table2_joint.joint import (
    Cohort,
    add_input_arguments,
    balanced_repeat_indices,
    load_cohort_pool,
    pareto_frontier,
    prepare_expression,
    resolve_inputs,
    selected_cohorts,
)
from experiments.table2_joint.pathway_concordance import (
    EnrichrClient,
    add_enrichr_arguments,
    gene_symbol_lookup,
    metric_rows_for_terms,
    ranked_gene_set,
    resolve_libraries,
)

DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "pathway"


def averaged_logfc_profiles(
    *,
    raw_counts: np.ndarray,
    raw_log: np.ndarray,
    recon_log: dict[str, np.ndarray],
    obs: pd.DataFrame,
    cohort: Cohort,
    cell_types: list[str],
    models: tuple[str, ...],
    seed_start: int,
    repeats: int,
    donors_per_state: int,
    cells_per_donor: int,
    min_detection_rate: float,
    min_repeat_fraction: float,
) -> dict[tuple[str, str], np.ndarray]:
    """Average detected disease logFC profiles over balanced repeat selections."""
    sources = ("raw", *models)
    n_genes = raw_log.shape[1]
    sums = {
        (cell_type, source): np.zeros(n_genes, dtype=np.float64)
        for cell_type in cell_types
        for source in sources
    }
    counts = {
        (cell_type, source): np.zeros(n_genes, dtype=np.int16)
        for cell_type in cell_types
        for source in sources
    }

    for repeat in range(repeats):
        indices = balanced_repeat_indices(
            obs,
            cohort,
            cell_types,
            seed=seed_start + repeat,
            donors_per_state=donors_per_state,
            cells_per_donor=cells_per_donor,
        )
        selected_obs = obs.iloc[indices].reset_index(drop=True)
        selected_types = selected_obs[CELL_TYPE_KEY].to_numpy(dtype=str)
        selected_states = selected_obs[STATE_KEY].to_numpy(dtype=str)
        selected_raw_counts = raw_counts[indices]
        source_values = {
            "raw": raw_log[indices],
            **{model: values[indices] for model, values in recon_log.items()},
        }
        for cell_type in cell_types:
            positive = (selected_types == cell_type) & (selected_states == cohort.disease)
            negative = (selected_types == cell_type) & (selected_states == cohort.reference)
            detected = (
                (selected_raw_counts[positive] > 0).mean(axis=0)
                + (selected_raw_counts[negative] > 0).mean(axis=0)
            ) > min_detection_rate
            for source, values in source_values.items():
                logfc = values[positive].mean(axis=0) - values[negative].mean(axis=0)
                valid = detected & np.isfinite(logfc)
                sums[(cell_type, source)][valid] += logfc[valid]
                counts[(cell_type, source)][valid] += 1

    minimum_repeats = max(1, int(np.ceil(repeats * min_repeat_fraction)))
    profiles: dict[tuple[str, str], np.ndarray] = {}
    for key in sums:
        profile = np.full(n_genes, np.nan, dtype=np.float64)
        valid = counts[key] >= minimum_repeats
        profile[valid] = sums[key][valid] / counts[key][valid]
        profiles[key] = profile
    return profiles


def enrich_profiles(
    *,
    cohort: Cohort,
    cell_types: list[str],
    symbols: list[str],
    profiles: dict[tuple[str, str], np.ndarray],
    models: tuple[str, ...],
    libraries: list[str],
    client: EnrichrClient,
    top_genes: int,
    top_terms: int,
    cells_per_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Pathway concordance of the averaged logFC profiles of every cell type."""
    metric_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    sources = ("raw", *models)
    for cell_type in cell_types:
        contrast = Contrast(
            contrast_id=(
                f"{cohort.dataset_id}::{cell_type}::{cohort.disease}_vs_{cohort.reference}"
            ),
            dataset_id=cohort.dataset_id,
            cell_type=cell_type,
            positive_label=cohort.disease,
            negative_label=cohort.reference,
            positive_n=cells_per_state,
            negative_n=cells_per_state,
        )
        contrast_rows.append({"cohort": cohort.name, **asdict(contrast)})
        for direction in ("up", "down"):
            source_enrichment: dict[str, pd.DataFrame] = {}
            for source in sources:
                genes = ranked_gene_set(
                    profiles[(cell_type, source)],
                    symbols,
                    direction=direction,
                    top_n=top_genes,
                )
                if len(genes) < 10:
                    continue
                rows = client.enrich(
                    genes,
                    libraries=libraries,
                    description=f"table2:{contrast.contrast_id}:{source}:{direction}",
                )
                source_enrichment[source] = pd.DataFrame(rows)
            raw_terms = source_enrichment.get("raw")
            if raw_terms is None or raw_terms.empty:
                continue
            for model in models:
                model_terms = source_enrichment.get(model)
                if model_terms is None or model_terms.empty:
                    continue
                for library in libraries:
                    raw_library = raw_terms.loc[raw_terms["library"] == library]
                    model_library = model_terms.loc[model_terms["library"] == library]
                    if raw_library.empty or model_library.empty:
                        continue
                    rows = metric_rows_for_terms(
                        model=model,
                        contrast=contrast,
                        direction=direction,
                        library=library,
                        raw_terms=raw_library,
                        model_terms=model_library,
                        top_terms=top_terms,
                    )
                    metric_rows.extend({"cohort": cohort.name, **row} for row in rows)
    return contrast_rows, metric_rows


def pathway_summaries(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return cohort-level and equal-cohort pathway summaries."""
    cohort = (
        metrics.groupby(["cohort", "dataset_id", "model", "metric"], observed=True)
        .agg(
            value=("value", "mean"),
            n_values=("value", "count"),
            n_cell_types=(CELL_TYPE_KEY, "nunique"),
        )
        .reset_index()
    )
    equal = (
        cohort.groupby(["model", "metric"], observed=True)
        .agg(value=("value", "mean"), n_cohorts=("cohort", "nunique"))
        .reset_index()
    )
    return cohort, equal


def metric_combination_sensitivity(
    repeat_metrics: pd.DataFrame,
    pathway_cohort: pd.DataFrame,
) -> pd.DataFrame:
    """Audit Pareto membership across all metric-family combinations."""
    pathway_wide = pathway_cohort.pivot(
        index=["cohort", "dataset_id", "model"],
        columns="metric",
        values="value",
    ).reset_index()
    joint = repeat_metrics.merge(
        pathway_wide,
        on=["cohort", "dataset_id", "model"],
        how="inner",
        validate="many_to_one",
    )
    identity = ("biological_identity", "identity_purity", "biological_state")
    context = ("context_invariance", "donor_bras", "local_donor_mixing")
    expression = (
        "expression_fidelity",
        "deg_jaccard",
        "deg_sign",
        "top10_pathway_score_spearman",
        "top10_pathway_jaccard",
    )
    cohort_means = (
        joint.groupby(["cohort", "model"], observed=True).mean(numeric_only=True).reset_index()
    )
    equal_cohort = (
        cohort_means.groupby("model", observed=True).mean(numeric_only=True).reset_index()
    )
    repeat_means = (
        joint.groupby(["repeat", "model"], observed=True).mean(numeric_only=True).reset_index()
    )
    models = sorted(equal_cohort["model"].astype(str).unique())
    rows: list[dict[str, object]] = []
    for identity_metric, context_metric, expression_metric in product(
        identity, context, expression
    ):
        metrics = [identity_metric, context_metric, expression_metric]
        aggregate_mask = pareto_frontier(equal_cohort[metrics].to_numpy())
        aggregate_frontier = set(equal_cohort.loc[aggregate_mask, "model"].astype(str))
        repeat_counts = {model: 0 for model in models}
        for _, group in repeat_means.groupby("repeat", observed=True):
            repeat_mask = pareto_frontier(group[metrics].to_numpy())
            for model in group.loc[repeat_mask, "model"].astype(str):
                repeat_counts[model] += 1
        row: dict[str, object] = {
            "identity_metric": identity_metric,
            "context_metric": context_metric,
            "expression_metric": expression_metric,
            "aggregate_frontier": "; ".join(sorted(aggregate_frontier)),
        }
        n_repeats = repeat_means["repeat"].nunique()
        for model in models:
            row[f"{model}_aggregate_pareto"] = model in aggregate_frontier
            row[f"{model}_repeat_pareto_frequency"] = repeat_counts[model] / n_repeats
        rows.append(row)
    return pd.DataFrame(rows)


def common_gene_symbols(pool, common_genes: list[str]) -> list[str]:
    """Gene symbols of the common genes, in the order used by ``prepare_expression``."""
    symbol_lookup = gene_symbol_lookup(pool)
    return [symbol_lookup.get(normalize_gene(gene), gene) for gene in common_genes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--joint-dir",
        type=Path,
        default=DEFAULT_JOINT_DIR,
        help="Output directory of experiments.table2_joint.joint (repeat_metrics.csv, "
        "cohort_audit.csv) for the Pareto sensitivity audit",
    )
    add_input_arguments(parser)
    parser.add_argument("--min-repeat-fraction", type=float, default=0.8)
    add_enrichr_arguments(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs, _, three_axis_models = resolve_inputs(args)
    cohorts = selected_cohorts(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    libraries = resolve_libraries(list(args.libraries))
    client = EnrichrClient(
        cache_path=args.enrichr_cache or args.output_dir / "enrichr_cache.jsonl",
        sleep=args.sleep,
    )
    print(f"Enrichr libraries: {libraries}; cached requests: {len(client.cache)}", flush=True)
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"embedding_cache", "recon_cache", "cohorts"}
        },
        "cohorts": [asdict(cohort) for cohort in cohorts],
        "three_axis_models": list(three_axis_models),
        "recon_caches": {k: describe_path(v) for k, v in inputs.recon_dirs.items()},
        **inputs.universe.config(),
        "resolved_libraries": libraries,
        "pathway_profile_aggregation": "mean logFC over balanced repeats",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    contrast_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for cohort in cohorts:
        print(f"\n=== {cohort.name}: {cohort.dataset_id} ===", flush=True)
        pool, obs, cell_types, _ = load_cohort_pool(
            cohort,
            inputs,
            min_cells_per_state=args.min_cells_per_state,
            min_cells_per_donor=args.min_cells_per_donor,
            min_donors_per_state=args.min_donors_per_state,
            min_cell_types=args.min_cell_types,
        )
        raw_counts, raw_log, recon_log, common_genes = prepare_expression(
            pool, cohort, inputs, three_axis_models
        )
        symbols = common_gene_symbols(pool, common_genes)
        profiles = averaged_logfc_profiles(
            raw_counts=raw_counts,
            raw_log=raw_log,
            recon_log=recon_log,
            obs=obs,
            cohort=cohort,
            cell_types=cell_types,
            models=three_axis_models,
            seed_start=args.seed_start,
            repeats=args.repeats,
            donors_per_state=args.donors_per_state,
            cells_per_donor=args.cells_per_donor,
            min_detection_rate=args.min_detection_rate,
            min_repeat_fraction=args.min_repeat_fraction,
        )
        print(
            f"  {len(cell_types)} cell types, {len(common_genes)} common genes; enrichment",
            flush=True,
        )
        cohort_contrasts, cohort_metrics = enrich_profiles(
            cohort=cohort,
            cell_types=cell_types,
            symbols=symbols,
            profiles=profiles,
            models=three_axis_models,
            libraries=libraries,
            client=client,
            top_genes=args.top_genes,
            top_terms=args.top_terms,
            cells_per_state=args.donors_per_state * args.cells_per_donor,
        )
        contrast_rows.extend(cohort_contrasts)
        metric_rows.extend(cohort_metrics)
        pd.DataFrame(metric_rows).to_csv(
            args.output_dir / "pathway_metrics_long.partial.csv", index=False
        )
        print(f"  completed {len(cohort_metrics)} pathway metric rows", flush=True)

    contrasts = pd.DataFrame(contrast_rows)
    metrics = pd.DataFrame(metric_rows)
    contrasts.to_csv(args.output_dir / "selected_contrasts.csv", index=False)
    metrics.to_csv(args.output_dir / "pathway_metrics_long.csv", index=False)
    cohort_summary, equal_summary = pathway_summaries(metrics)
    cohort_summary.to_csv(args.output_dir / "pathway_cohort_summary.csv", index=False)
    equal_summary.to_csv(args.output_dir / "pathway_equal_cohort_summary.csv", index=False)

    audit = pd.read_csv(args.joint_dir / "cohort_audit.csv")
    audit = audit.loc[audit["cohort"].isin({cohort.name for cohort in cohorts})]
    expected_contrasts = int(audit["n_eligible_cell_types"].sum())
    if len(contrasts) != expected_contrasts:
        raise RuntimeError(f"Expected {expected_contrasts} contrasts, found {len(contrasts)}")
    expected_metrics = expected_contrasts * len(three_axis_models) * 2 * len(libraries) * 2
    if len(metrics) != expected_metrics:
        raise RuntimeError(f"Expected {expected_metrics} metrics, found {len(metrics)}")
    if not np.isfinite(metrics["value"]).all():
        raise RuntimeError("Pathway metrics contain non-finite values")

    repeat_metrics = pd.read_csv(args.joint_dir / "repeat_metrics.csv")
    sensitivity = metric_combination_sensitivity(repeat_metrics, cohort_summary)
    sensitivity.to_csv(args.output_dir / "metric_combination_sensitivity.csv", index=False)
    if len(sensitivity) != 3 * 3 * 5:
        raise RuntimeError(f"Expected 45 metric combinations, found {len(sensitivity)}")
    print(
        f"Done: {len(contrasts)} contrasts, {len(metrics)} pathway metrics, "
        f"{len(sensitivity)} metric combinations",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
