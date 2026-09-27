#!/usr/bin/env bash
# Reproduce the scTrilemma row of Table 1 from the released checkpoint.
#   1. embedding caches for the 89 held-out datasets (GPU)   -> outputs/experiments/embeddings/sctrilemma/embeddings
#   2. 20 stratified metric samples per dataset               -> outputs/experiments/scoring/metric_samples
#   3. scorers: FAISS k-means repeats, scib-metrics repeats, PCR repeats
#   4. assemble the long file with the shipped baseline rows and build the table
set -euo pipefail
cd "$(dirname "$0")/../.."

export SCTRILEMMA_DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
CACHE_DIR="${CACHE_DIR:-outputs/experiments/embeddings/sctrilemma/embeddings}"
SAMPLES_DIR="${SAMPLES_DIR:-outputs/experiments/scoring/metric_samples}"
SCORING_DIR="${SCORING_DIR:-outputs/experiments/scoring}"
OUT_DIR="${OUT_DIR:-outputs/experiments/table1_benchmark}"
RUN="pixi run python -m"

bash scripts/evaluation/export_embeddings.sh
$RUN experiments.scoring.metric_samples --cache-dir "$CACHE_DIR" --output-dir "$SAMPLES_DIR"
$RUN experiments.scoring.kmeans_repeat  --cache-dir "$CACHE_DIR" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$SCORING_DIR/kmeans_repeat" --model-name sctrilemma
$RUN experiments.scoring.scib_repeat    --cache-dir "$CACHE_DIR" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$SCORING_DIR/scib_repeat"   --model-name sctrilemma
$RUN experiments.scoring.pcr_repeat     --cache-dir "$CACHE_DIR" --metric-samples-dir "$SAMPLES_DIR" --output-dir "$SCORING_DIR/pcr_repeat"    --model-name sctrilemma
$RUN experiments.table1_benchmark.assemble_long \
    --kmeans-long "$SCORING_DIR/kmeans_repeat/kmeans_repeat_long__sctrilemma.csv" \
    --scib-long   "$SCORING_DIR/scib_repeat/scib_repeat20_long__sctrilemma.csv" \
    --pcr-long    "$SCORING_DIR/pcr_repeat/pcr_repeat20_long.csv" \
    --output      "$OUT_DIR/table1_long.csv"
$RUN experiments.table1_benchmark.build_table --long "$OUT_DIR/table1_long.csv" --output-dir "$OUT_DIR"
