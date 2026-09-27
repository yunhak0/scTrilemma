"""Assemble the Table 1 long file from the scorer outputs of one model.

Takes the three scorer outputs of ``experiments.scoring`` (k-means repeats, scib-metrics
repeats, PCR repeats) for one model, converts them to the ``table1_long.csv`` layout
(dataset, model, metric, repeat_kind, repeat, value) and merges them with the rows of the
other models from the shipped ``results/table1_long.csv``.

    pixi run python -m experiments.table1_benchmark.assemble_long \
        --kmeans-long outputs/experiments/scoring/kmeans_repeat/kmeans_repeat_long__sctrilemma.csv \
        --scib-long outputs/experiments/scoring/scib_repeat/scib_repeat20_long__sctrilemma.csv \
        --pcr-long outputs/experiments/scoring/pcr_repeat/pcr_repeat20_long.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCORING = ROOT / "outputs/experiments/scoring"

SCIB_METRICS = {"silhouette_label": "asw", "clisi_knn": "clisi", "isolated_labels": "iso_label", "bras": "bras", "ilisi_knn": "ilisi"}


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["dataset"] != "dataset"].copy()  # tolerate a repeated header from resumed runs
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def kmeans_rows(path: Path, model: str) -> pd.DataFrame:
    frame = _numeric(pd.read_csv(path))
    frame = frame[frame["metric"].isin(["nmi", "ari"])]
    return pd.DataFrame({"dataset": frame["dataset"], "model": model, "metric": frame["metric"],
                         "repeat_kind": "kmeans_seed", "repeat": frame["kmeans_seed"].astype(int), "value": frame["value"]})


def scib_rows(path: Path, model: str) -> pd.DataFrame:
    frame = _numeric(pd.read_csv(path))
    frame = frame[frame["metric"].isin(SCIB_METRICS)]
    return pd.DataFrame({"dataset": frame["dataset"], "model": model, "metric": frame["metric"].map(SCIB_METRICS),
                         "repeat_kind": "sample_seed", "repeat": frame["sample_seed"].astype(int), "value": frame["value"]})


def pcr_rows(path: Path, model: str, source_model: str) -> pd.DataFrame:
    frame = _numeric(pd.read_csv(path))
    if "model" in frame:
        frame = frame[frame["model"].astype(str) == source_model]
    if "status" in frame:
        frame = frame[frame["status"] == "complete"]
    return pd.DataFrame({"dataset": frame["dataset"], "model": model, "metric": "pcr",
                         "repeat_kind": "sample_seed", "repeat": frame["sample_seed"].astype(int), "value": frame["value"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="sctrilemma")
    parser.add_argument("--kmeans-long", type=Path, default=SCORING / "kmeans_repeat/kmeans_repeat_long__sctrilemma.csv")
    parser.add_argument("--scib-long", type=Path, default=SCORING / "scib_repeat/scib_repeat20_long__sctrilemma.csv")
    parser.add_argument("--pcr-long", type=Path, default=SCORING / "pcr_repeat/pcr_repeat20_long.csv")
    parser.add_argument("--pcr-source-model", default=None,
                        help="Value of the `model` column in the PCR long file (default: --model-name)")
    parser.add_argument("--baselines", type=Path, default=HERE / "results/table1_long.csv",
                        help="Shipped long file whose rows of the other models are kept")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/table1_benchmark/table1_long.csv")
    args = parser.parse_args()

    ours = pd.concat([kmeans_rows(args.kmeans_long, args.model_name), scib_rows(args.scib_long, args.model_name),
                      pcr_rows(args.pcr_long, args.model_name, args.pcr_source_model or args.model_name)], ignore_index=True)
    others = pd.read_csv(args.baselines)
    others = others[others["model"] != args.model_name]
    long = pd.concat([others, ours], ignore_index=True).sort_values(["model", "metric", "dataset", "repeat"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output, index=False, float_format="%.10g")
    counts = ours.groupby("metric")["dataset"].nunique()
    print(f"{args.model_name}: {len(ours)} rows; datasets per metric: {counts.to_dict()}")
    print(f"wrote {args.output} ({len(long)} rows)")


if __name__ == "__main__":
    main()
