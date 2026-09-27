"""Figure 4: trilemma response curves and coupling matrix from the demand-targeted interventions.

    pixi run python -m experiments.fig4_interventions.make_figure

Panel set 1 (response curves): one panel per pushed demand; x = change of the pushed metric,
y = change of the other two demands. Panel set 2 (coupling matrix): slope of each responding
demand per unit change of the pushed demand (least squares through the origin over the
non-degenerate alpha range), with the fraction of datasets moving in the mean direction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.figure_style import PRIMARY_TEXT, apply_style, demote_legend, demote_ticks

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "outputs/experiments/fig4_interventions/summary"
COLOR = {"identity": "#2D6FD6", "invariance": "#26A862", "fidelity": "#8E5BD0"}
ALPHAS = (0.25, 0.5, 0.75)  # unified grid for the main-text curves
PANELS = [  # (intervention, pushed demand, pushed metric, alpha max, title)
    ("centroid_shrinkage", "identity", "silhouette_label", 0.75, "Push identity $\\uparrow$\n(centroid shrinkage)"),
    ("donor_centering", "invariance", "bras", 0.75, "Push context invariance $\\uparrow$\n(donor centering)"),
    ("latent_refinement", "fidelity", "recon_mse", 1.0, "Push expression fidelity $\\uparrow$\n(latent refinement)"),
]
RESP = {"identity": [("silhouette_label", "Label ASW", "-")], "invariance": [("bras", "BRAS", "-")],
        "fidelity": [("recon_mse", "−Recon. MSE", "-"), ("recon_pearson", "Recon. Pearson", ":")]}
LABEL = {"silhouette_label": "Label ASW", "bras": "BRAS", "recon_pearson": "Recon. Pearson", "recon_mse": "−Recon. MSE"}
SIGN = {"recon_mse": -1.0}  # lower MSE is better; plot as −ΔMSE so that "up = better" holds for every line


def curves(d: pd.DataFrame, out: Path, height: float = 2.05, stem: str = "trilemma_response_curves") -> None:
    apply_style({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(6.4, height))
    handles: dict[str, object] = {}
    for ax, (name, pushed, pm, amax, title) in zip(axes, PANELS):
        g = d[(d.intervention == name) & d.alpha.isin(ALPHAS)]
        x = g[g.metric == pm].set_index("alpha")["mean_delta"].sort_index() * SIGN.get(pm, 1.0)
        x = pd.concat([pd.Series({0.0: 0.0}), x])
        for dem in [k for k in COLOR if k != pushed]:
            for m, lab, ls in RESP[dem]:
                s = g[g.metric == m].set_index("alpha").reindex(x.index[1:])
                y = pd.concat([pd.Series({0.0: 0.0}), s["mean_delta"] * SIGN.get(m, 1.0)])
                e = pd.concat([pd.Series({0.0: 0.0}), s["sem_delta"]])
                h = ax.errorbar(x.values, y.values, yerr=e.values, marker="o", ms=3.2, lw=1.4, ls=ls, color=COLOR[dem], label=lab,
                                markeredgecolor="white", markeredgewidth=0.5, capsize=2, capthick=1.0, elinewidth=1.0)
                handles.setdefault(lab, h)
        ax.axhline(0, color="#9A9A9A", lw=0.7, ls="--", zorder=1)
        ax.axvline(0, color="#D6D6D6", lw=0.7, ls=":", zorder=0)
        # numeric ticks stay; each intervention point is marked by a light vertical guide with its alpha at the top
        pts = [(a, v) for a, v in zip(x.index, x.values) if a > 0]
        span = x.values.max() - x.values.min()
        for k, (a, v) in enumerate(pts):
            ax.axvline(v, color="#D6D6D6", lw=0.7, ls=":", zorder=0)
            # one label row; neighbours that sit close on the x axis are aligned away from each other
            near_left = k > 0 and (v - pts[k - 1][1]) < 0.2 * span
            near_right = k + 1 < len(pts) and (pts[k + 1][1] - v) < 0.2 * span
            ha = "left" if near_left and not near_right else ("right" if near_right and not near_left else "center")
            ax.text(v, 1.0, f"{a:g}", transform=ax.get_xaxis_transform(), ha=ha, va="bottom",
                    fontsize=7.5, fontweight="bold", color=PRIMARY_TEXT)
        ax.text(0.0, 1.0, "α:", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=PRIMARY_TEXT)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=5)
        ax.set_xlabel(f"$\\Delta$ {LABEL[pm]} (pushed)")
        ax.set_title(title, color=COLOR[pushed], pad=13)
        ax.grid(color="#ECECEC", linewidth=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    demote_ticks(axes, which="both")
    axes[0].set_ylabel("$\\Delta$ other demands")
    order = [lab for lab in ("Label ASW", "BRAS", "\u2212Recon. MSE", "Recon. Pearson") if lab in handles]
    demote_legend(fig.legend([handles[lab] for lab in order], order, loc="upper center", ncol=4, frameon=False,
                             bbox_to_anchor=(0.5, 1.0), prop={"size": 7.5, "weight": "bold"}, labelcolor="black",
                             handletextpad=0.4, columnspacing=1.4, handlelength=2.4))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=250, bbox_inches="tight")


def coupling_matrix(d: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for name, pushed, pm, amax, _ in PANELS:
        g = d[(d.intervention == name) & (d.alpha <= amax)]
        x = g[g.metric == pm].set_index("alpha")["mean_delta"].sort_index()
        for dem in COLOR:
            for m, lab, _ in RESP[dem]:
                if dem == pushed and m != pm:
                    continue
                s = g[g.metric == m].set_index("alpha").reindex(x.index)
                x0 = float(g[g.metric == pm]["base_mean"].iloc[0])
                y0 = float(s["base_mean"].iloc[0])
                xr = x.values / x0 * SIGN.get(pm, 1.0)
                yr = s["mean_delta"].values / y0 * SIGN.get(m, 1.0)
                slope = float((xr * yr).sum() / (xr ** 2).sum())  # relative response per relative push
                # consistency: share of datasets moving with the mean at the mid alpha
                mid = x.index[len(x) // 2]
                r = s.loc[mid]
                frac = (r["n_up"] if r["mean_delta"] > 0 else r["n"] - r["n_up"]) / r["n"]
                rows.append({"pushed": pushed, "pushed_metric": pm, "responding": dem, "metric": m, "slope": slope, "consistency": frac})
    t = pd.DataFrame(rows)
    t.to_csv(out / "coupling_matrix.csv", index=False)
    # draw: rows pushed, cols responding (identity, invariance, fidelity[Pearson], fidelity[MSE])
    cols = [("identity", "silhouette_label"), ("invariance", "bras"), ("fidelity", "recon_mse")]
    order = ["identity", "invariance", "fidelity"]
    fig, ax = plt.subplots(figsize=(3.2, 2.4), constrained_layout=True)
    M = np.full((3, 3), np.nan)
    for i, p in enumerate(order):
        for j, (dem, m) in enumerate(cols):
            r = t[(t.pushed == p) & (t.metric == m)]
            if len(r):
                M[i, j] = r.slope.iloc[0]
    disp = M.copy()
    vmax = np.nanmax(np.abs(disp[~np.eye(3, dtype=bool)]))
    ax.imshow(np.clip(disp, -vmax, vmax), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                continue
            r = t[(t.pushed == order[i]) & (t.metric == cols[j][1])].iloc[0]
            pushed_cell = (order[i] == cols[j][0]) and (cols[j][1] == PANELS[i][2])
            txt = "pushed" if pushed_cell else f"{M[i, j]:+.2f}\n({r.consistency:.0%})"
            dark = (not pushed_cell) and abs(disp[i, j]) > 0.6 * vmax
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else "#222")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["identity\n(ASW)", "invariance\n(BRAS)", "fidelity\n(−MSE)"], fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["push identity", "push invariance", "push fidelity"], fontsize=7.5)
    ax.set_xlabel("responding demand (relative change per relative change of the pushed metric)", fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"trilemma_coupling_matrix.{ext}", dpi=250, bbox_inches="tight")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-dir", type=Path, default=SUMMARY)
    args = ap.parse_args()
    d = pd.read_csv(args.summary_dir / "paired_deltas.csv")
    curves(d, args.summary_dir)
    t = coupling_matrix(d, args.summary_dir)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))



if __name__ == "__main__":
    main()
