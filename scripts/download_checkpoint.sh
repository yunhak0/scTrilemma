#!/usr/bin/env bash
# Download the released checkpoint from the Hugging Face Hub and verify its checksum.
#
#   bash scripts/download_checkpoint.sh            # curl (no Python dependency)
#   bash scripts/download_checkpoint.sh --python   # huggingface_hub (in the pixi environment; else pip install huggingface_hub)
#
# Overrides: HF_REPO_ID, HF_REVISION, HF_FILENAME, DEST
set -euo pipefail
cd "$(dirname "$0")/.."

HF_REPO_ID="${HF_REPO_ID:-yunhak0/scTrilemma}"
HF_REVISION="${HF_REVISION:-main}"
HF_FILENAME="${HF_FILENAME:-final.ckpt}"
DEST="${DEST:-checkpoints/sctrilemma/final.ckpt}"
SHA256_EXPECTED="58981767d498c5d9c4087276bc2853cb6d5c338fc5048d84b3d8f8c9cacf5f76"
URL="https://huggingface.co/${HF_REPO_ID}/resolve/${HF_REVISION}/${HF_FILENAME}"

mkdir -p "$(dirname "$DEST")"
if [ "${1:-}" = "--python" ]; then
    python - "$HF_REPO_ID" "$HF_FILENAME" "$HF_REVISION" "$DEST" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
repo_id, filename, revision, dest = sys.argv[1:5]
path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
shutil.copyfile(path, dest)
print(f"downloaded {repo_id}/{filename}@{revision} -> {dest}")
PY
else
    echo "downloading ${URL}"
    curl -L --fail --retry 3 -C - -o "$DEST" "$URL"
fi

echo "verifying checksum"
echo "${SHA256_EXPECTED}  ${DEST}" | sha256sum -c -
echo "ok: ${DEST}"
