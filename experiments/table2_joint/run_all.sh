#!/usr/bin/env bash
# Reproduce Table 2 and appendix Tables 3, 4, 7, 18, 19 for the released checkpoint.
#
#   1. sampled cells (<= 2,500 per held-out dataset, seed 42)          CPU, Census data
#   2. embedding caches of the 89 held-out datasets                     GPU
#   3. reconstruction caches over the shared gene lists                 GPU (CPU: ~12 h)
#   4. Table 4: metadata audit of candidate contrasts                   CPU, Census data
#      (optional; joint.py defaults to the shipped results/cohort_audit/candidate_pairs.csv)
#   5. Table 2 / 3: five cohorts x 20 balanced repeats                  CPU, minutes
#   6. Table 2 pathway rows                                             CPU + Enrichr (network)
#   7. Tables 7 / 18: donor-supported seven-cohort panel                CPU + Enrichr (network)
#   8. Table 19: direct reconstruction agreement                        CPU, about a minute
#
# Baseline rows of the paper tables are shipped as CSV under results/ and are not recomputed.
set -euo pipefail
cd "$(dirname "$0")/../.."

export SCTRILEMMA_DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
OUT="${OUT:-outputs/experiments/table2_joint}"
RUN="pixi run python -m"
RUN_COHORT_AUDIT="${RUN_COHORT_AUDIT:-0}"   # 1 = re-derive candidate_pairs.csv from the Census metadata

$RUN experiments.prepare_samples
bash scripts/evaluation/export_embeddings.sh
$RUN experiments.reconstruction.generate

if [ "$RUN_COHORT_AUDIT" = "1" ]; then
    $RUN experiments.table2_joint.cohort_audit --output-dir "$OUT/cohort_audit"
    CANDIDATES=(--candidate-pairs "$OUT/cohort_audit/candidate_pairs.csv")
else
    CANDIDATES=()
fi

$RUN experiments.table2_joint.joint     --output-dir "$OUT/joint" ${CANDIDATES[@]+"${CANDIDATES[@]}"}
$RUN experiments.table2_joint.pathway   --output-dir "$OUT/pathway" --joint-dir "$OUT/joint"
$RUN experiments.table2_joint.rq4_panel all --output-dir "$OUT/rq4_panel"
$RUN experiments.reconstruction.score_agreement
