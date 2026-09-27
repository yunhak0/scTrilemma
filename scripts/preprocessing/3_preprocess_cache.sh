#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export CPU_THREADS="${CPU_THREADS:-4}"

CENSUS_TAG="${CENSUS_TAG:-20250130}"
DATA_ROOT="${DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
GENE_VOCAB_PATH="${GENE_VOCAB_PATH:-$DATA_ROOT/$CENSUS_TAG/gene_vocab_homo_sapiens_$CENSUS_TAG.json}"
CELL_TYPE_VOCAB_PATH="${CELL_TYPE_VOCAB_PATH:-$DATA_ROOT/$CENSUS_TAG/cell_type_vocab_homo_sapiens_$CENSUS_TAG.json}"
NUM_WORKERS="${NUM_WORKERS:-16}"

pixi run python -m sctrilemma.preprocessing.preprocess_cache \
  --data_root "$DATA_ROOT" \
  --census_version "$CENSUS_TAG" \
  --vocab_path "$GENE_VOCAB_PATH" \
  --cell_type_vocab_path "$CELL_TYPE_VOCAB_PATH" \
  --num_workers "$NUM_WORKERS"
