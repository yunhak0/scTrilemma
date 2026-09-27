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
MIN_CELLS="${MIN_CELLS:-2}"
ID_COLUMN="${ID_COLUMN:-feature_id}"
SYMBOL_COLUMN="${SYMBOL_COLUMN:-feature_name}"
SKIP_ENSEMBL_ENRICHMENT="${SKIP_ENSEMBL_ENRICHMENT:-0}"

if [[ -z "${ENSEMBL_RELEASE:-}" ]]; then
  case "${CENSUS_VERSION}" in
    2025-11-08|20251108) ENSEMBL_RELEASE=114 ;;
    *) ENSEMBL_RELEASE=110 ;;
  esac
fi

VERSION_TAG="${CENSUS_VERSION//-/}"
GTF_PATH="${GTF_PATH:-$OUTPUT_DIR/$VERSION_TAG/Homo_sapiens.GRCh38.${ENSEMBL_RELEASE}.gtf.gz}"
FASTA_PATH="${FASTA_PATH:-$OUTPUT_DIR/$VERSION_TAG/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"

args=(
  --census_version "$CENSUS_VERSION"
  --organism "$ORGANISM"
  --output_dir "$OUTPUT_DIR"
  --min_cells "$MIN_CELLS"
  --ensembl_release "$ENSEMBL_RELEASE"
  --id_column "$ID_COLUMN"
  --symbol_column "$SYMBOL_COLUMN"
)

if [[ "${ENSEMBL_RELEASE}" == "114" ]]; then
  if [[ ! -f "${GTF_PATH}" ]]; then
    echo "Ensembl 114 GTF not found: ${GTF_PATH}" >&2
    echo "Download it first:" >&2
    echo "  wget -P ${OUTPUT_DIR}/${VERSION_TAG} https://ftp.ensembl.org/pub/release-114/gtf/homo_sapiens/Homo_sapiens.GRCh38.114.gtf.gz" >&2
  fi
  args+=(--gtf_path "$GTF_PATH")
  if [[ -f "${FASTA_PATH}" ]]; then
    args+=(--fasta_path "$FASTA_PATH")
  fi
fi

if [[ "${SKIP_ENSEMBL_ENRICHMENT}" == "1" ]]; then
  args+=(--skip_ensembl_enrichment)
fi

pixi run python -m sctrilemma.preprocessing.download_metadata \
  "${args[@]}"
