#!/usr/bin/env bash
# Export scTrilemma embeddings of the 89 held-out datasets to per-dataset caches
# (outputs/experiments/embeddings/sctrilemma/embeddings/<dataset>.npz), the input of the
# Table 1 scoring modules in experiments/scoring/. Protocol: <= 100,000 cells per dataset
# (dataset-hash sampler, seed 0), batch size 256, posterior-mean embeddings (512-d).
#
#   bash scripts/evaluation/export_embeddings.sh
#   GPU=1 DATASET_IDS="<uuid> <uuid>" bash scripts/evaluation/export_embeddings.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

IDS_FILE="${IDS_FILE:-configs/zsb/full_89_ids.txt}"
DATASET_IDS="${DATASET_IDS:-}"
DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
TARGET_PATH="${TARGET_PATH:-${SCTRILEMMA_TARGET_PATH:-$DATA_ROOT/20251108/by_dataset}}"
GENE_VOCAB="${GENE_VOCAB:-${SCTRILEMMA_GENE_VOCAB:-$DATA_ROOT/20250130/gene_vocab_homo_sapiens_20250130.json}}"
SCTRILEMMA_CKPT="${SCTRILEMMA_CKPT:-checkpoints/sctrilemma/final.ckpt}"
TISSUE_CODE="${TISSUE_CODE:-}"
PSEUDO_BULK="${PSEUDO_BULK:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/experiments/embeddings/sctrilemma}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_CELLS="${MAX_CELLS:-100000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
PYTHON="${PYTHON:-pixi run python}"

export CPU_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for f in "${SCTRILEMMA_CKPT}" "${GENE_VOCAB}" "${IDS_FILE}"; do
    [ -e "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
[ -d "${TARGET_PATH}" ] || { echo "missing data directory: ${TARGET_PATH}" >&2; exit 1; }

EXTRA_ARGS=()
[ -n "${TISSUE_CODE}" ] && EXTRA_ARGS+=(--tissue-code "${TISSUE_CODE}")
[ -n "${PSEUDO_BULK}" ] && EXTRA_ARGS+=(--pseudo-bulk "${PSEUDO_BULK}")
for id in ${DATASET_IDS}; do
    EXTRA_ARGS+=(--dataset-id "${id}")
done

echo "scTrilemma embedding export"
echo "  ids:        ${DATASET_IDS:-${IDS_FILE}}"
echo "  checkpoint: ${SCTRILEMMA_CKPT}"
echo "  data:       ${TARGET_PATH}"
echo "  output:     ${OUTPUT_DIR}"
echo "  gpu:        ${GPU}"
echo "  max cells:  ${MAX_CELLS} (seed ${SAMPLE_SEED}), batch size ${BATCH_SIZE}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" ${PYTHON} -m sctrilemma.benchmark.export_embeddings \
    --dataset-ids-file "${IDS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${SCTRILEMMA_CKPT}" \
    --target-path "${TARGET_PATH}" \
    --gene-vocab "${GENE_VOCAB}" \
    --max-cells-per-dataset "${MAX_CELLS}" \
    --sample-seed "${SAMPLE_SEED}" \
    --batch-size "${BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"

echo
echo "done. caches:"
ls -1 "${OUTPUT_DIR}"
