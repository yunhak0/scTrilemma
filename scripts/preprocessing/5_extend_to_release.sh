#!/usr/bin/env bash
# Run steps 0-2 for EXTRA_CENSUS_TAG first, then merge its pseudo-bulk profiles
# into the training release and extend the tissue codes.
set -euo pipefail
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CENSUS_TAG="${CENSUS_TAG:-20250130}"
EXTRA_CENSUS_TAG="${EXTRA_CENSUS_TAG:-20251108}"
DATA_ROOT="${DATA_ROOT:-${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}}"
K="${K:-32}"

pixi run python -m sctrilemma.preprocessing.extend_tissue_codes \
  --data_root "$DATA_ROOT" \
  --census_version "$CENSUS_TAG" \
  --extra_census_version "$EXTRA_CENSUS_TAG" \
  --k "$K"
