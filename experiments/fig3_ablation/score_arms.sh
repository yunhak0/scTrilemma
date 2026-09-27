#!/usr/bin/env bash
# Score the ablation arms with the Table 1 scorers and the reconstruction-fidelity modules.
# The released checkpoint ("Full") is scored by experiments/table1_benchmark/run_all.sh and
# experiments/table2_joint/run_all.sh; this script covers the retrained arms only.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARMS="${ARMS:-ablation_full_retrained ablation_no_egate ablation_no_croute ablation_no_pbcond ablation_pbcond_shuffled_codes ablation_pbcond_constant_code}"
export SCTRILEMMA_DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
SAMPLES_DIR="${SAMPLES_DIR:-outputs/experiments/scoring/metric_samples}"   # the metric samples of the released checkpoint (same cells)
OUT="${OUT:-outputs/experiments/fig3_ablation}"
RUN="pixi run python -m"

for arm in $ARMS; do
    cache="outputs/experiments/embeddings/$arm/embeddings"
    $RUN experiments.scoring.kmeans_repeat --cache-dir "$cache" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$OUT/scoring/$arm/kmeans_repeat" --model-name "$arm"
    $RUN experiments.scoring.scib_repeat   --cache-dir "$cache" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$OUT/scoring/$arm/scib_repeat"   --model-name "$arm"
    $RUN experiments.scoring.pcr_repeat    --cache-dir "$cache" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$OUT/scoring/$arm/pcr_repeat"    --model-name "$arm"
    $RUN experiments.reconstruction.hvg_fidelity --run "$arm" --checkpoint "checkpoints/$arm/final.ckpt" --output-dir "$OUT/hvg"
    $RUN experiments.reconstruction.generate --checkpoint "checkpoints/$arm/final.ckpt" --output-dir "$OUT/recon_cache/$arm"
    $RUN experiments.reconstruction.score_agreement --cache-dir "$OUT/recon_cache/$arm" --model-name "$arm" --output-dir "$OUT/recon/$arm"
done

$RUN experiments.fig3_ablation.build_long --output "$OUT/ablation_per_dataset_long.csv"
$RUN experiments.fig3_ablation.wilcoxon --input "$OUT/ablation_per_dataset_long.csv" --output "$OUT/ablation_wilcoxon_tests.csv"
$RUN experiments.fig3_ablation.plot --long "$OUT/ablation_per_dataset_long.csv" --tests "$OUT/ablation_wilcoxon_tests.csv" --output-dir "$OUT"
