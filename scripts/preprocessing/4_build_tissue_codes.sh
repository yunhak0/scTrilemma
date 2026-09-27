#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CENSUS_TAG="${CENSUS_TAG:-20250130}"
DATA_ROOT="${DATA_ROOT:-${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}}"
K="${K:-32}"

pixi run python -m sctrilemma.preprocessing.build_tissue_codes \
  --data_root "$DATA_ROOT" \
  --census_version "$CENSUS_TAG" \
  --k "$K"
