#!/usr/bin/env bash
# Control tissue codes for the PB-Cond ablation arms (configs/experiment/ablation_pbcond_*.yaml).
set -euo pipefail
cd "$(dirname "$0")/../.."

CENSUS_TAG="${CENSUS_TAG:-20250130}"
EXTRA_CENSUS_TAG="${EXTRA_CENSUS_TAG:-20251108}"
DATA_ROOT="${DATA_ROOT:-${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}}"
K="${K:-32}"
SEED="${SEED:-42}"

pixi run python -m sctrilemma.preprocessing.build_control_codes \
  --codes "$DATA_ROOT/$CENSUS_TAG/tissue_codes_k${K}_with_${EXTRA_CENSUS_TAG}.pt" \
  --seed "$SEED"
