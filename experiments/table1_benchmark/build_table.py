"""Table 1: zero-shot benchmark summary from the 20-repeat metric values.

Input is a long CSV with one row per (dataset, model, metric, repeat): NMI/ARI are repeated
over 20 k-means seeds on the seed-0 metric sample, the other metrics over 20 metric samples.
Per dataset the repeats are averaged first; the table reports the mean and SD of these
per-dataset values across datasets. Stars mark scTrilemma gains over the strongest baseline
of each metric by a dataset-paired two-sided Wilcoxon test.

    pixi run python -m experiments.table1_benchmark.build_table
    pixi run python -m experiments.table1_benchmark.build_table --long <other_long.csv> --output-dir <dir>
"""

from __future__ import annotations

import argparse
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MODEL = "sctrilemma"
METRICS = ["nmi", "ari", "asw", "clisi", "iso_label", "bras", "ilisi", "pcr"]
LABEL = {"nmi": "NMI", "ari": "ARI", "asw": "ASW", "clisi": "cLISI", "iso_label": "Iso. Label", "bras": "BRAS", "ilisi": "iLISI", "pcr": "PCR"}
DISPLAY = {"scvi": "scVI", "geneformer": "Geneformer", "scgpt": "scGPT", "cellplm": "CellPLM", "scprint": "scPRINT", MODEL: "scTrilemma"}


def per_dataset_means(long: pd.DataFrame) -> pd.DataFrame:
    return long.groupby(["model", "metric", "dataset"], as_index=False)["value"].mean()


def summarize(per: pd.DataFrame) -> pd.DataFrame:
    out = per.groupby(["model", "metric"]).value.agg(mean="mean", sd="std", n_datasets="count").reset_index()
    return out


def paired_tests(per: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """scTrilemma against the strongest baseline of each metric, paired by dataset."""
    piv = per.pivot_table(index="dataset", columns=["model", "metric"], values="value")
    means = summary.pivot(index="model", columns="metric", values="mean")
    rows = []
    for metric in METRICS:
        baselines = [m for m in means.index if m != MODEL and not np.isnan(means.loc[m, metric])]
        if not baselines or (MODEL, metric) not in piv:
            continue
        strongest = max(baselines, key=lambda m: means.loc[m, metric])
        pair = piv[[(MODEL, metric), (strongest, metric)]].dropna()
        delta = pair.iloc[:, 0] - pair.iloc[:, 1]
        p = float(wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
        rows.append({
            "metric": metric, "strongest_baseline": strongest, "n_datasets": int(len(delta)),
            "sctrilemma_mean": float(pair.iloc[:, 0].mean()), "baseline_mean": float(pair.iloc[:, 1].mean()),
            "mean_difference": float(delta.mean()), "n_wins": int((delta > 0).sum()), "n_losses": int((delta < 0).sum()),
            "p_two_sided": p, "stars": stars(p) if delta.mean() > 0 else "",
        })
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    """Three decimals with half-up rounding (as in the paper table)."""
    return str(Decimal(f"{value:.6f}").quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def stars(p: float) -> str:
    return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else ""))


def latex_rows(summary: pd.DataFrame, tests: pd.DataFrame, order: list[str]) -> str:
    means = summary.pivot(index="model", columns="metric", values="mean")
    sds = summary.pivot(index="model", columns="metric", values="sd")
    star = tests.set_index("metric")["stars"].to_dict() if len(tests) else {}
    lines = []
    for model in order:
        cells = []
        for metric in METRICS:
            if model not in means.index or np.isnan(means.loc[model, metric]):
                cells.append("--")
                continue
            ranked = means[metric].dropna().sort_values(ascending=False).index.tolist()
            value = fmt(float(means.loc[model, metric]))
            if ranked and ranked[0] == model:
                value = f"\\mathbf{{{value}}}"
            elif len(ranked) > 1 and ranked[1] == model:
                value = f"\\mathit{{{value}}}"
            if model == MODEL and star.get(metric):
                value += f"^{{{star[metric]}}}"
            cells.append(f"\\tabmeanstd{{{value}}}{{{fmt(float(sds.loc[model, metric]))}}}")
        lines.append(f"\\tabmethod{{{DISPLAY.get(model, model)}}} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", type=Path, default=HERE / "results/table1_long.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/table1_benchmark")
    args = parser.parse_args()
    long = pd.read_csv(args.long)
    per = per_dataset_means(long)
    summary = summarize(per)
    tests = paired_tests(per, summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "table1_summary.csv", index=False, float_format="%.6g")
    tests.to_csv(args.output_dir / "table1_wilcoxon.csv", index=False, float_format="%.6g")
    order = [m for m in ("scvi", "geneformer", "scgpt", "cellplm", "scprint", MODEL) if m in set(long.model)]
    table = summary.pivot(index="model", columns="metric", values="mean").reindex(order)[[m for m in METRICS if m in summary.metric.unique()]]
    print(table.round(3).rename(columns=LABEL).to_string())
    print()
    print(tests.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print(latex_rows(summary, tests, order))
    (args.output_dir / "table1_rows.tex").write_text(latex_rows(summary, tests, order) + "\n")


if __name__ == "__main__":
    main()
