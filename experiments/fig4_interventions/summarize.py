"""Summarise demand-targeted latent interventions: paired deltas vs alpha=0 and response curves.

    pixi run python -m experiments.fig4_interventions.summarize
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/experiments/fig4_interventions"
DEMAND = {
    "nmi": "identity", "ari": "identity", "silhouette_label": "identity", "clisi_knn": "identity", "isolated_labels": "identity",
    "bras": "invariance", "ilisi_knn": "invariance",
    "recon_pearson": "fidelity", "recon_spearman": "fidelity",
}
PUSHED = {"donor_centering": "bras", "centroid_shrinkage": "silhouette_label", "label_shrinkage": "nmi", "latent_refinement": "recon_pearson"}
LABEL = {"nmi": "NMI", "ari": "ARI", "silhouette_label": "Label ASW", "bras": "BRAS", "ilisi_knn": "iLISI",
         "recon_pearson": "Recon. Pearson", "recon_spearman": "Recon. Spearman", "clisi_knn": "cLISI", "isolated_labels": "Iso. label"}
COLOR = {"identity": "#4C72B0", "invariance": "#55A868", "fidelity": "#C44E52"}


def load(dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for d in dirs:
        p = d / "interventions_long.csv"
        if p.exists():
            f = pd.read_csv(p)
            f["run"] = d.name.split("_part")[0]  # dataset-split parts form one run
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Delta vs the alpha=0 row of the same run and dataset; intervention rows carry the alpha."""
    base = df[df.intervention == "none"].set_index(["run", "dataset", "metric"])["value"]
    rows = []
    for (run, name, a, m), g in df[df.intervention != "none"].groupby(["run", "intervention", "alpha", "metric"]):
        g = g.set_index("dataset")["value"]
        b = base.loc[run].xs(m, level="metric").reindex(g.index)
        d = (g - b).dropna()
        if len(d) < 3:
            continue
        try:
            p = wilcoxon(d.values).pvalue if (d != 0).any() else 1.0
        except ValueError:
            p = np.nan
        rows.append({"intervention": name, "alpha": a, "metric": m, "demand": DEMAND.get(m, "other"), "n": len(d),
                     "base_mean": b.reindex(d.index).mean(), "mean_delta": d.mean(), "sem_delta": d.std(ddof=1) / np.sqrt(len(d)),
                     "n_down": int((d < 0).sum()), "n_up": int((d > 0).sum()), "p_wilcoxon": p})
    return pd.DataFrame(rows).sort_values(["intervention", "metric", "alpha"])


def response_curves(deltas: pd.DataFrame, out: Path) -> None:
    """2 x 3 grid: rows = intervention, columns = demand; x = intervention strength alpha."""
    cols = [("identity", ["nmi", "ari", "silhouette_label"]), ("invariance", ["bras", "ilisi_knn"]), ("fidelity", ["recon_pearson", "recon_spearman"])]
    pushed_demand = {"donor_centering": "invariance", "centroid_shrinkage": "identity", "label_shrinkage": "identity", "latent_refinement": "fidelity"}
    titles = {"donor_centering": "Donor centering", "centroid_shrinkage": "Centroid shrinkage", "label_shrinkage": "Label-centroid shrinkage", "latent_refinement": "Latent refinement"}
    deltas = deltas[~(deltas.intervention.isin(["centroid_shrinkage", "label_shrinkage"]) & (deltas.alpha >= 1.0))]  # alpha=1 collapses cells onto K points
    names = [n for n in ["donor_centering", "centroid_shrinkage", "label_shrinkage", "latent_refinement"] if n in set(deltas.intervention)]
    fig, axes = plt.subplots(len(names), 3, figsize=(7.4, 2.1 * len(names)), constrained_layout=True, squeeze=False)
    for r, name in enumerate(names):
        d = deltas[deltas.intervention == name]
        for c, (demand, ms) in enumerate(cols):
            ax = axes[r, c]
            for m in ms:
                s = d[d.metric == m].sort_values("alpha")
                if s.empty:
                    continue
                ax.errorbar(s["alpha"], s["mean_delta"], yerr=s["sem_delta"], marker="o", ms=3, lw=1.1, color=COLOR[demand],
                            alpha=1.0 if m in ("nmi", "bras", "recon_pearson") else 0.45, label=LABEL[m])
            ax.axhline(0, color="#888", lw=0.7)
            if name == "donor_centering":
                ax.axvline(1.0, color="#bbb", lw=0.7, ls="--")
            if demand == pushed_demand[name]:
                ax.set_facecolor("#f3f3f3")
                ax.set_title(f"{demand} (pushed)", fontsize=8.5, fontweight="bold")
            else:
                ax.set_title(demand, fontsize=8.5)
            ax.legend(fontsize=6.2, frameon=False, loc="best")
            ax.tick_params(labelsize=7)
            if r == len(names) - 1:
                ax.set_xlabel("intervention strength α", fontsize=8)
        axes[r, 0].set_ylabel(f"{titles[name]}\nΔ vs. unmodified", fontsize=7.5)
    fig.savefig(out / "response_curves.pdf", bbox_inches="tight")
    fig.savefig(out / "response_curves.png", dpi=200, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="+",
                    default=[BASE, BASE / "alpha_fine", BASE / "alpha_over", BASE / "alpha_mid"] + sorted(BASE.glob("refine_label*")))
    ap.add_argument("--output-dir", type=Path, default=BASE / "summary")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.dirs)
    deltas = paired_deltas(df)
    deltas.to_csv(args.output_dir / "paired_deltas.csv", index=False)
    response_curves(deltas, args.output_dir)
    show = deltas[deltas.metric.isin(["nmi", "silhouette_label", "bras", "ilisi_knn", "recon_pearson"]) & deltas.alpha.isin([0.25, 0.5, 0.75, 1.0])]
    for name, g in show.groupby("intervention"):
        print(f"\n== {name}")
        piv = g.pivot_table(index="alpha", columns="metric", values="mean_delta")
        cnt = g.pivot_table(index="alpha", columns="metric", values="n_up")
        n = g.pivot_table(index="alpha", columns="metric", values="n")
        for a in piv.index:
            print(f"  a={a:<5} " + "  ".join(f"{m}={piv.loc[a, m]:+.4f}({int(cnt.loc[a, m])}/{int(n.loc[a, m])}↑)" for m in piv.columns))


if __name__ == "__main__":
    main()
