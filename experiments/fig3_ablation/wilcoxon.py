"""Paired Wilcoxon tests for the component ablation (Figure 3 / Table 24).

The unit of inference is a held-out dataset, not an individual cell.  For each
metric, the script pairs a comparison arm with the reference arm by dataset,
then applies a two-sided Wilcoxon signed-rank test to the per-dataset deltas.
The output also retains the means and delta SEM needed to report effect sizes.

Example:
    pixi run python -m experiments.fig3_ablation.wilcoxon --input outputs/experiments/fig3_ablation/ablation_per_dataset_long.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/experiments/fig3_ablation/ablation_per_dataset_long.csv"
DEFAULT_OUTPUT = ROOT / "outputs/experiments/fig3_ablation/ablation_wilcoxon_tests.csv"
DEFAULT_COMPARISONS = ["w/o_E-Gate", "w/o_C-Route", "w/o_PB-Cond"]


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    """Return Benjamini--Hochberg FDR-adjusted q-values."""
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted

    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n_tests = len(ranked)
    ranked_adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    adjusted[finite_indices[order]] = np.minimum(ranked_adjusted, 1.0)
    return adjusted


def load_scope_results(path: Path, scope: str) -> pd.DataFrame:
    """Load unique finite (condition, dataset, metric) results for one scope."""
    frame = pd.read_csv(path)
    required = {"scope", "condition", "dataset", "metric", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame = frame.loc[frame["scope"] == scope, ["condition", "dataset", "metric", "value"]].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        raise ValueError(f"No finite rows found for scope={scope!r}")
    duplicates = frame.duplicated(subset=["condition", "dataset", "metric"], keep=False)
    if duplicates.any():
        examples = frame.loc[duplicates, ["condition", "dataset", "metric"]].head(5)
        raise ValueError(
            "Duplicate (condition, dataset, metric) rows found; refusing to choose one:\n"
            f"{examples.to_string(index=False)}"
        )
    return frame


def paired_wilcoxon(
    results: pd.DataFrame,
    reference_name: str,
    comparison_name: str,
    scope: str,
) -> list[dict[str, object]]:
    """Return one Wilcoxon summary row per metric for a paired comparison."""
    reference = results.loc[results["condition"] == reference_name].rename(
        columns={"value": "reference_value"}
    )
    comparison = results.loc[results["condition"] == comparison_name].rename(
        columns={"value": "comparison_value"}
    )
    if reference.empty or comparison.empty:
        missing = reference_name if reference.empty else comparison_name
        raise ValueError(f"No rows found for condition={missing!r} in scope={scope!r}")

    paired = reference.merge(
        comparison,
        on=["dataset", "metric"],
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_comparison"),
    )
    if paired.empty:
        raise ValueError(f"{reference_name!r} and {comparison_name!r} share no dataset-metric pairs")

    rows: list[dict[str, object]] = []
    for metric, group in paired.groupby("metric", sort=True):
        deltas = group["comparison_value"].to_numpy(dtype=float) - group["reference_value"].to_numpy(
            dtype=float
        )
        nonzero = deltas[deltas != 0.0]
        if nonzero.size == 0:
            statistic = float("nan")
            p_value = float("nan")
        else:
            # ``method='auto'`` uses an exact calculation when SciPy permits it
            # and its standard tie/zero-aware approximation otherwise.
            result: Any = wilcoxon(
                deltas,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)

        n_datasets = int(len(group))
        rows.append(
            {
                "scope": scope,
                "reference": reference_name,
                "comparison": comparison_name,
                "metric": metric,
                "n_datasets": n_datasets,
                "n_nonzero_deltas": int(nonzero.size),
                "reference_mean": float(group["reference_value"].mean()),
                "comparison_mean": float(group["comparison_value"].mean()),
                "mean_delta_comparison_minus_reference": float(deltas.mean()),
                "delta_sem": float(deltas.std(ddof=1) / np.sqrt(n_datasets)) if n_datasets > 1 else float("nan"),
                "wilcoxon_statistic": statistic,
                "p_two_sided": p_value,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Long-form per-dataset result CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--scope", default="leave_one_out", help="Value in the input CSV's scope column")
    parser.add_argument("--reference", default="Full", help="Reference condition")
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=DEFAULT_COMPARISONS,
        help="One or more comparison conditions",
    )
    args = parser.parse_args()

    results = load_scope_results(args.input, args.scope)
    rows = [
        row
        for comparison in args.comparisons
        for row in paired_wilcoxon(results, args.reference, comparison, args.scope)
    ]
    table = pd.DataFrame(rows)
    table["p_bh_fdr"] = benjamini_hochberg(table["p_two_sided"])
    table.to_csv(args.output, index=False, float_format="%.10g")
    print(table.to_string(index=False))
    print(f"wrote {len(table)} tests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
