"""Figure 3: component ablation as standardized effects.

For each removed component, the per-dataset change relative to the full model is
summarized as mean(delta) / SD(delta) across datasets (MSE is sign-flipped so that
negative values always mean degradation). Stars come from the BH-FDR-corrected paired
Wilcoxon tests written by ``experiments.fig3_ablation.wilcoxon``.

    pixi run python -m experiments.fig3_ablation.plot \
        --long outputs/experiments/fig3_ablation/ablation_per_dataset_long.csv \
        --tests outputs/experiments/fig3_ablation/ablation_wilcoxon_tests.csv \
        --output-dir outputs/experiments/fig3_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.figure_style import apply_style, demote_ticks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "outputs/experiments/fig3_ablation"

REFERENCE = "Full"
PANELS = [("w/o_E-Gate", "w/o E-Gate"), ("w/o_C-Route", "w/o C-Route"), ("w/o_PB-Cond", "w/o PB-Cond")]
# (metric key, axis label, demand, sign) -- top to bottom
METRICS = [
    ("nmi", "NMI", "identity", 1.0),
    ("asw", "ASW", "identity", 1.0),
    ("bras", "BRAS", "context", 1.0),
    ("pcr", "PCR", "context", 1.0),
    ("training_style_hvg", "HVG $r$", "fidelity", 1.0),
    ("mse", "$-$MSE", "fidelity", -1.0),
]
COLOR = {"identity": "#2D6FD6", "context": "#26A862", "fidelity": "#8E5BD0"}
BAND = {"identity": "#EEF3FC", "context": "#EEF8F2", "fidelity": "#F6F2FB"}


def standardized_effects(long: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """One row per (condition, metric): standardized effect, n and BH-FDR q."""
    full = long[long.condition == REFERENCE].pivot(index="dataset", columns="metric", values="value")
    rows = []
    for condition, _ in PANELS:
        for metric, _, demand, sign in METRICS:
            arm = long[(long.condition == condition) & (long.metric == metric)].set_index("dataset")["value"]
            ref = full[metric].reindex(arm.index).dropna()
            delta = (arm.reindex(ref.index) - ref) * sign
            q = tests[(tests.comparison == condition) & (tests.metric == metric)]["p_bh_fdr"]
            rows.append({
                "condition": condition, "metric": metric, "demand": demand, "n": int(len(delta)),
                "mean_delta": float(delta.mean()), "sd_delta": float(delta.std(ddof=1)),
                "effect": float(delta.mean() / delta.std(ddof=1)),
                "q": float(q.iloc[0]) if len(q) else np.nan,
            })
    return pd.DataFrame(rows)


def stars(q: float) -> str:
    if np.isnan(q):
        return ""
    return "**" if q < 0.01 else ("*" if q < 0.05 else "")


def draw(effects: pd.DataFrame, output_dir: Path, stem: str = "fig3_ablation") -> None:
    apply_style({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5})
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 1.55), sharex=True, sharey=True)
    y_positions = np.arange(len(METRICS))[::-1]
    x_min = min(-4.5, float(effects.effect.min()) - 1.6)
    x_max = max(3.0, float(effects.effect.max()) + 1.6)
    for ax, (condition, title) in zip(axes, PANELS):
        for y, (_, _, demand, _) in zip(y_positions, METRICS):
            ax.axhspan(y - 0.5, y + 0.5, color=BAND[demand], zorder=0, lw=0)
        ax.axvline(0, color="black", lw=1.0, zorder=1)
        ax.grid(axis="x", color="#DADADA", lw=0.6, zorder=0)
        sub = effects[effects.condition == condition].set_index("metric")
        for y, (metric, _, demand, _) in zip(y_positions, METRICS):
            row = sub.loc[metric]
            e = float(row["effect"])
            ax.scatter([e], [y], marker="D", s=26, color=COLOR[demand], edgecolor="black", linewidth=0.6, zorder=3)
            label = f"{e:+.2f}{stars(float(row['q']))}"
            if e < 0:
                ax.text(e - 0.18, y, label, ha="right", va="center", fontsize=6.6, fontweight="bold", zorder=4)
            else:
                ax.text(e + 0.18, y, label, ha="left", va="center", fontsize=6.6, fontweight="bold", zorder=4)
        ax.set_title(title, pad=4)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, len(METRICS) - 0.5)
        ax.set_xticks([-4, -2, 0, 2])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([label for _, label, _, _ in METRICS])
    for tick, (_, _, demand, _) in zip(axes[0].get_yticklabels(), METRICS):
        tick.set_color(COLOR[demand])
    demote_ticks(axes, which="x")
    fig.supxlabel("standardized effect ($\\Delta$ / SD across datasets); MSE sign flipped", fontsize=8, y=-0.04)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.85, bottom=0.2, wspace=0.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long", type=Path, default=DEFAULT_DIR / "ablation_per_dataset_long.csv")
    parser.add_argument("--tests", type=Path, default=DEFAULT_DIR / "ablation_wilcoxon_tests.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    long = pd.read_csv(args.long)
    tests = pd.read_csv(args.tests)
    effects = standardized_effects(long, tests)
    effects.to_csv(args.output_dir / "ablation_standardized_effects.csv", index=False) if args.output_dir.exists() else None
    draw(effects, args.output_dir)
    print(effects.assign(label=lambda d: d.apply(lambda r: f"{r.effect:+.2f}{stars(r.q)}", axis=1))[["condition", "metric", "n", "effect", "q", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
