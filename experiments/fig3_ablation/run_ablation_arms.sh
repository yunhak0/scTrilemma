#!/usr/bin/env bash
# Train the six ablation arms with the released recipe (4 GPUs each, sequentially) and export
# their embedding caches. Checkpoints land in checkpoints/<arm>/final.ckpt.
set -euo pipefail
cd "$(dirname "$0")/../.."

ARMS="${ARMS:-ablation_full_retrained ablation_no_egate ablation_no_croute ablation_no_pbcond ablation_pbcond_shuffled_codes ablation_pbcond_constant_code}"
export SCTRILEMMA_DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"

# control tissue codes for the two PB-Cond control arms
bash scripts/preprocessing/6_build_control_codes.sh

for arm in $ARMS; do
    if [ ! -f "checkpoints/$arm/final.ckpt" ]; then
        bash scripts/training/train.sh "experiment=$arm"
    fi
    SCTRILEMMA_CKPT="checkpoints/$arm/final.ckpt" OUTPUT_DIR="outputs/experiments/embeddings/$arm" \
        bash scripts/evaluation/export_embeddings.sh
done
