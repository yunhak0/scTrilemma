#!/usr/bin/env bash
# Minimal single-dataset example: checks that the environment, checkpoint and
# data layout are wired correctly before launching the full benchmark.
#
# This is a setup check, not a reproduction: it scores one held-out dataset on
# one GPU with a reduced cell cap, so the numbers will not match the paper.
# Use scripts/evaluation/run_benchmark_4gpu.sh for the reported results.
#
#   bash scripts/evaluation/run_single_dataset_example.sh
#   DATASET_ID=<uuid> MAX_CELLS=20000 bash scripts/evaluation/run_single_dataset_example.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET_ID="${DATASET_ID:-$(head -n 1 configs/zsb/full_89_ids.txt)}"
DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
TARGET_PATH="${TARGET_PATH:-${SCTRILEMMA_TARGET_PATH:-$DATA_ROOT/20251108/by_dataset}}"
GENE_VOCAB="${GENE_VOCAB:-${SCTRILEMMA_GENE_VOCAB:-$DATA_ROOT/20250130/gene_vocab_homo_sapiens_20250130.json}}"
SCTRILEMMA_CKPT="${SCTRILEMMA_CKPT:-checkpoints/sctrilemma/final.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/example/single_dataset}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_CELLS="${MAX_CELLS:-10000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
PYTHON="${PYTHON:-pixi run python}"

export CPU_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for f in "${SCTRILEMMA_CKPT}" "${GENE_VOCAB}"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
[ -d "${TARGET_PATH}" ] || { echo "missing data directory: ${TARGET_PATH}" >&2; exit 1; }

echo "scTrilemma single-dataset example"
echo "  dataset:    ${DATASET_ID}"
echo "  checkpoint: ${SCTRILEMMA_CKPT}"
echo "  data:       ${TARGET_PATH}"
echo "  gpu:        ${GPU}"
echo "  max cells:  ${MAX_CELLS}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON} -m sctrilemma.benchmark.run \
    --models sctrilemma \
    --dataset-ids "${DATASET_ID}" \
    --target-path "${TARGET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --sctrilemma-checkpoint "${SCTRILEMMA_CKPT}" \
    --sctrilemma-gene-vocab "${GENE_VOCAB}" \
    --batch-size "${BATCH_SIZE}" \
    --n-jobs 1 \
    --max-cells-per-dataset "${MAX_CELLS}" \
    --sample-seed "${SAMPLE_SEED}"

echo
echo "done. results:"
ls -1 "${OUTPUT_DIR}"
