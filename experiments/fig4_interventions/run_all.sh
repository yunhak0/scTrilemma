#!/usr/bin/env bash
# Reproduce Figure 4 / Table 20: demand-targeted latent interventions on the released checkpoint.
#
# Prerequisites: checkpoints/sctrilemma/final.ckpt and the sampled datasets from
#   pixi run python -m experiments.prepare_samples
#
# The paper combines five runs with different alpha grids; each run scores all 89 datasets and
# carries its own alpha=0 reference, so keep the run boundaries as below.
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT="${OUT:-outputs/experiments/fig4_interventions}"
RUN="pixi run python -m experiments.fig4_interventions.run_interventions"

$RUN --output-dir "$OUT"             --alphas 0.25 0.5 0.75 1.0        --interventions donor_centering centroid_shrinkage
$RUN --output-dir "$OUT/alpha_fine"  --alphas 0.05 0.1 0.15 0.2 0.3 0.4 --interventions donor_centering centroid_shrinkage
$RUN --output-dir "$OUT/alpha_mid"   --alphas 0.6 0.9                  --interventions centroid_shrinkage
$RUN --output-dir "$OUT/alpha_over"  --alphas 1.25 1.5 2.0             --interventions donor_centering
$RUN --output-dir "$OUT/refine_label" --alphas 0.25 0.5 0.75 1.0       --interventions latent_refinement label_shrinkage \
     --refine-steps 30 --refine-lr-scale 0.02 --refine-objective pearson

pixi run python -m experiments.fig4_interventions.summarize --dirs "$OUT" "$OUT/alpha_fine" "$OUT/alpha_over" "$OUT/alpha_mid" "$OUT/refine_label" --output-dir "$OUT/summary"
pixi run python -m experiments.fig4_interventions.make_figure --summary-dir "$OUT/summary"
