#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export CPU_THREADS="${CPU_THREADS:-4}"

CENSUS_VERSION="${CENSUS_VERSION:-2025-01-30}"
ORGANISM="${ORGANISM:-homo_sapiens}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/$USER/datasets/cellxgene}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MIN_CELLS="${MIN_CELLS:-2}"
TEST_LIMIT="${TEST_LIMIT:-0}"

pixi run python -m sctrilemma.preprocessing.download_census \
  --census_version "$CENSUS_VERSION" \
  --organism "$ORGANISM" \
  --output_dir "$OUTPUT_DIR" \
  --max_workers "$MAX_WORKERS" \
  --min_cells "$MIN_CELLS" \
  --test_limit "$TEST_LIMIT"
