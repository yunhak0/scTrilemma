#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

IDS_FILE="${IDS_FILE:-configs/zsb/full_89_ids.txt}"
DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"
TARGET_PATH="${TARGET_PATH:-${SCTRILEMMA_TARGET_PATH:-$DATA_ROOT/20251108/by_dataset}}"
GENE_VOCAB="${GENE_VOCAB:-${SCTRILEMMA_GENE_VOCAB:-$DATA_ROOT/20250130/gene_vocab_homo_sapiens_20250130.json}}"
SCTRILEMMA_CKPT="${SCTRILEMMA_CKPT:-checkpoints/sctrilemma/final.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/reproduction/zsb_full89_benchmark_4gpu}"
WORKERS="${WORKERS:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_JOBS="${N_JOBS:-1}"
CPU_THREADS="${CPU_THREADS:-4}"
MAX_CELLS_PER_DATASET="${MAX_CELLS_PER_DATASET:-100000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"

export CPU_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${CPU_THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${CPU_THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${CPU_THREADS}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SCTRILEMMA_SCGRAPH_TOTAL_MAX="${SCTRILEMMA_SCGRAPH_TOTAL_MAX:-5000}"
export SCTRILEMMA_SCGRAPH_MAX_PER_TYPE="${SCTRILEMMA_SCGRAPH_MAX_PER_TYPE:-250}"
export SCTRILEMMA_SCGRAPH_MAX_DENSE_GB="${SCTRILEMMA_SCGRAPH_MAX_DENSE_GB:-8.0}"

mkdir -p "${OUTPUT_ROOT}/splits" "${OUTPUT_ROOT}/shards" "${OUTPUT_ROOT}/merged"

echo "scTrilemma zero-shot benchmark reproduction (89 held-out datasets)"
echo "  ids: ${IDS_FILE}"
echo "  checkpoint: ${SCTRILEMMA_CKPT}"
echo "  output: ${OUTPUT_ROOT}"
echo "  workers: ${WORKERS}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  max_cells_per_dataset: ${MAX_CELLS_PER_DATASET}"
echo "  sample_seed: ${SAMPLE_SEED}"

IFS=',' read -r -a GPU_IDS <<< "${WORKERS}"

pixi run python - <<PY
from pathlib import Path
import csv
import sys

from sctrilemma.benchmark.data_utils import create_dataset_manifest

ids_file = Path("${IDS_FILE}")
target_path = Path("${TARGET_PATH}")
split_root = Path("${OUTPUT_ROOT}") / "splits"
gpu_ids = "${WORKERS}".split(",")

dataset_ids = [
    line.strip()
    for line in ids_file.read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
manifest = create_dataset_manifest(
    dataset_ids,
    str(target_path),
    str(split_root / "dataset_manifest.csv"),
)
items = [
    (str(row.dataset_id), int(row.n_cells), int(row.dataset_total_size_bytes))
    for row in manifest.itertuples(index=False)
]
bins = [{"gpu": gpu, "n_cells": 0, "n_bytes": 0, "ids": []} for gpu in gpu_ids]
for dataset_id, n_cells, n_bytes in sorted(items, key=lambda x: x[1], reverse=True):
    target = min(bins, key=lambda b: b["n_cells"])
    target["ids"].append(dataset_id)
    target["n_cells"] += n_cells
    target["n_bytes"] += n_bytes

split_root.mkdir(parents=True, exist_ok=True)
cell_lookup = {dataset_id: (n_cells, n_bytes) for dataset_id, n_cells, n_bytes in items}
for idx, bucket in enumerate(bins):
    bucket["ids"].sort(key=lambda dataset_id: cell_lookup[dataset_id][0])
    (split_root / f"ids_{idx}.txt").write_text("\\n".join(bucket["ids"]) + "\\n")

with (split_root / "balanced_assignment.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["split", "gpu", "dataset_id", "n_cells", "dataset_total_size_bytes"],
    )
    writer.writeheader()
    for idx, bucket in enumerate(bins):
        for dataset_id in bucket["ids"]:
            n_cells, n_bytes = cell_lookup[dataset_id]
            writer.writerow({
                "split": idx,
                "gpu": bucket["gpu"],
                "dataset_id": dataset_id,
                "n_cells": n_cells,
                "dataset_total_size_bytes": n_bytes,
            })

print("Balanced split summary:")
for idx, bucket in enumerate(bins):
    print(
        f"  split={idx} gpu={bucket['gpu']} "
        f"datasets={len(bucket['ids'])} cells={bucket['n_cells']:,} "
        f"bytes={bucket['n_bytes']:,}"
    )
PY

pids=()
for idx in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$idx]}"
    ids_path="${OUTPUT_ROOT}/splits/ids_${idx}.txt"
    shard_dir="${OUTPUT_ROOT}/shards/s$(printf '%02d' "${idx}")_g${gpu}"
    mkdir -p "${shard_dir}"
    mapfile -t shard_ids < "${ids_path}"
    if [[ "${#shard_ids[@]}" -eq 0 ]]; then
        continue
    fi
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        pixi run python -m sctrilemma.benchmark.run \
            --models sctrilemma \
            --dataset-ids "${shard_ids[@]}" \
            --target-path "${TARGET_PATH}" \
            --output-dir "${shard_dir}" \
            --sctrilemma-checkpoint "${SCTRILEMMA_CKPT}" \
            --sctrilemma-gene-vocab "${GENE_VOCAB}" \
            --batch-size "${BATCH_SIZE}" \
            --n-jobs "${N_JOBS}" \
            --profile \
            --max-cells-per-dataset "${MAX_CELLS_PER_DATASET}" \
            --sample-seed "${SAMPLE_SEED}"
    ) > "${shard_dir}/run.log" 2>&1 &
    pids+=("$!")
    echo "  launched shard ${idx} on GPU ${gpu}: ${#shard_ids[@]} datasets, pid ${pids[-1]}"
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failures=$((failures + 1))
    fi
done

pixi run python - <<PY
from pathlib import Path
import csv
import sys

from sctrilemma.benchmark.metrics import summarize_results_csv

root = Path("${OUTPUT_ROOT}")
merged = root / "merged"
merged.mkdir(parents=True, exist_ok=True)
out = merged / "results.csv"
header = None
rows = []
for path in sorted((root / "shards").glob("s*_g*/results.csv")):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if header is None:
            header = reader.fieldnames
        rows.extend(reader)
if header is None:
    raise SystemExit("No shard results.csv files found")
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)
summarize_results_csv(out, merged / "results_summary.csv")
print(f"[REPRO_BENCHMARK_OUTPUT] {out}")
print(f"[REPRO_BENCHMARK_SUMMARY] {merged / 'results_summary.csv'}")
PY

if [[ "${failures}" -ne 0 ]]; then
    echo "${failures} shard(s) failed" >&2
    exit 1
fi
