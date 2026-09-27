"""Assemble the per-dataset ablation table (input of ``wilcoxon.py`` and ``plot.py``).

For the released checkpoint (``Full``) and each retrained arm the per-dataset value of every
metric is the mean over its repeats:

* ``nmi``, ``ari``                      FAISS k-means, 20 seeds        (experiments.scoring.kmeans_repeat)
* ``asw``, ``bras``                     scib-metrics, 20 metric samples (experiments.scoring.scib_repeat)
* ``pcr``                               20 metric samples, multi-donor datasets (experiments.scoring.pcr_repeat)
* ``training_style_hvg``, ``gene_mean_pearson``  (experiments.reconstruction.hvg_fidelity)
* ``mse``                               cell-wise MSE on the shared gene list, detection >= 2%
                                        (experiments.reconstruction.score_agreement)

Default layout: the ``Full`` sources are the Table 1 / Table 2 outputs of the released
checkpoint; each arm's sources live under ``outputs/experiments/fig3_ablation/{scoring,hvg,recon}/<arm>``
as written by ``score_arms.sh``. ``--paths-json`` overrides the source files per condition
(``{"<condition>": {"kmeans": ..., "scib": ..., "pcr": ..., "hvg": ..., "mse": ..., "model": ...}}``).

    pixi run python -m experiments.fig3_ablation.build_long
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/fig3_ablation"

# condition label, E-Gate, C-Route, PB-Cond, arm name (experiment_name / model column), note
CONDITIONS = [
    ("Full", 1, 1, 1, "sctrilemma", "Released checkpoint (Table 1/2 source)."),
    ("Full_(retrained)", 1, 1, 1, "ablation_full_retrained", "Full configuration retrained alongside the ablation arms."),
    ("w/o_E-Gate", 0, 1, 1, "ablation_no_egate", "expr_proj_mode=linear."),
    ("w/o_C-Route", 1, 0, 1, "ablation_no_croute", "decoder_query_mode=off."),
    ("w/o_PB-Cond", 1, 1, 0, "ablation_no_pbcond", "tissue_prior_mode=''."),
    ("PB-Cond_(global_shuffle)", 1, 1, 1, "ablation_pbcond_shuffled_codes", "PB-Cond trained with globally shuffled pseudo-bulk codes."),
    ("PB-Cond_(constant_code)", 1, 1, 1, "ablation_pbcond_constant_code", "PB-Cond trained with a constant (mean) pseudo-bulk code."),
]
SCIB_METRICS = {"silhouette_label": "asw", "bras": "bras"}
HVG_METRICS = {"training_style_recon_pearson_hvg": "training_style_hvg", "gene_mean_pearson": "gene_mean_pearson"}
COLUMNS = ["scope", "condition", "e_gate", "c_route", "pb_cond", "provenance", "dataset", "metric", "value", "note"]


def default_sources(arm: str, args: argparse.Namespace) -> dict[str, Path]:
    if arm == "sctrilemma":
        scoring = args.full_scoring_root
        return {
            "kmeans": scoring / "kmeans_repeat/kmeans_repeat_long__sctrilemma.csv",
            "scib": scoring / "scib_repeat/scib_repeat20_long__sctrilemma.csv",
            "pcr": scoring / "pcr_repeat/pcr_repeat20_long.csv",
            "hvg": args.full_hvg,
            "mse": args.full_mse,
        }
    return {
        "kmeans": args.arm_root / "scoring" / arm / f"kmeans_repeat/kmeans_repeat_long__{arm}.csv",
        "scib": args.arm_root / "scoring" / arm / f"scib_repeat/scib_repeat20_long__{arm}.csv",
        "pcr": args.arm_root / "scoring" / arm / "pcr_repeat/pcr_repeat20_long.csv",
        "hvg": args.arm_root / "hvg" / f"{arm}.csv",
        "mse": args.arm_root / "recon" / arm / "per_dataset.csv",
    }


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["dataset"] != "dataset"].copy() if "dataset" in frame else frame.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"])


def _filter_model(frame: pd.DataFrame, model: str, source: Path) -> pd.DataFrame:
    if "model" not in frame:
        return frame
    models = frame["model"].astype(str).unique()
    if len(models) == 1:
        return frame
    if model not in models:
        raise ValueError(f"{source}: model {model!r} not among {sorted(models)}")
    return frame[frame["model"].astype(str) == model]


def load_kmeans(path: Path, model: str) -> pd.DataFrame:
    frame = _filter_model(_numeric(pd.read_csv(path)), model, path)
    frame = frame[frame["metric"].isin(["nmi", "ari"])]
    return frame.groupby(["dataset", "metric"], as_index=False)["value"].mean()


def load_scib(path: Path, model: str) -> pd.DataFrame:
    frame = _filter_model(_numeric(pd.read_csv(path)), model, path)
    frame = frame[frame["metric"].isin(SCIB_METRICS)]
    out = frame.groupby(["dataset", "metric"], as_index=False)["value"].mean()
    out["metric"] = out["metric"].map(SCIB_METRICS)
    return out


def load_pcr(path: Path, model: str) -> pd.DataFrame:
    frame = _filter_model(_numeric(pd.read_csv(path)), model, path)
    if "status" in frame:
        frame = frame[frame["status"] == "complete"]
    out = frame.groupby("dataset", as_index=False)["value"].mean()
    out["metric"] = "pcr"
    return out[["dataset", "metric", "value"]]


def load_hvg(path: Path) -> pd.DataFrame:
    frame = _numeric(pd.read_csv(path))
    frame = frame[frame["metric"].isin(HVG_METRICS)]
    out = frame.groupby(["dataset", "metric"], as_index=False)["value"].mean()
    out["metric"] = out["metric"].map(HVG_METRICS)
    return out


def load_mse(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = _filter_model(frame, model, path)
    out = frame[["dataset_id", "recon_mse"]].rename(columns={"dataset_id": "dataset", "recon_mse": "value"}).copy()
    out["metric"] = "mse"
    return out[["dataset", "metric", "value"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-root", type=Path, default=OUT, help="Root with scoring/<arm>, hvg/<arm>.csv, recon/<arm>/")
    parser.add_argument("--full-scoring-root", type=Path, default=ROOT / "outputs/experiments/scoring")
    parser.add_argument("--full-hvg", type=Path, default=ROOT / "outputs/experiments/reconstruction/hvg_fidelity/sctrilemma.csv")
    parser.add_argument("--full-mse", type=Path, default=ROOT / "outputs/experiments/reconstruction/results_det02/per_dataset.csv")
    parser.add_argument("--paths-json", type=Path, default=None, help="Per-condition source overrides")
    parser.add_argument("--conditions", nargs="+", default=None, help="Subset of condition labels")
    parser.add_argument("--provenance", default="ablation_arms")
    parser.add_argument("--allow-missing", action="store_true", help="Skip sources that do not exist")
    parser.add_argument("--output", type=Path, default=OUT / "ablation_per_dataset_long.csv")
    args = parser.parse_args()

    overrides = json.loads(args.paths_json.read_text()) if args.paths_json else {}
    frames = []
    for condition, eg, cr, pb, arm, note in CONDITIONS:
        if args.conditions and condition not in args.conditions:
            continue
        sources = default_sources(arm, args)
        custom = overrides.get(condition, {})
        sources.update({k: Path(v) for k, v in custom.items() if k != "model"})
        model = custom.get("model", arm)
        # a source may label the model differently (``"<source>_model"`` in --paths-json)
        name = {key: custom.get(f"{key}_model", model) for key in ("kmeans", "scib", "pcr", "mse")}
        loaders = {"kmeans": lambda p: load_kmeans(p, name["kmeans"]), "scib": lambda p: load_scib(p, name["scib"]),
                   "pcr": lambda p: load_pcr(p, name["pcr"]), "hvg": load_hvg, "mse": lambda p: load_mse(p, name["mse"])}
        parts = []
        for key, loader in loaders.items():
            path = sources[key]
            if not path.exists():
                if args.allow_missing:
                    print(f"[missing] {condition}: {key} -> {path}")
                    continue
                raise FileNotFoundError(f"{condition}: {key} source not found: {path}")
            parts.append(loader(path))
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame.insert(0, "scope", "leave_one_out")
        frame.insert(1, "condition", condition)
        frame.insert(2, "e_gate", eg)
        frame.insert(3, "c_route", cr)
        frame.insert(4, "pb_cond", pb)
        frame.insert(5, "provenance", args.provenance)
        frame["note"] = note
        frames.append(frame[COLUMNS])
    long = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output, index=False)
    print(long.groupby(["condition", "metric"])["value"].agg(["count", "mean"]).round(4).to_string())
    print(f"wrote {args.output} ({len(long)} rows)")


if __name__ == "__main__":
    main()
