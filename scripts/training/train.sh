#!/usr/bin/env bash
# Train scTrilemma with the released recipe (configs/sctrilemma.yaml; 4 GPUs, DDP).
#
#   bash scripts/training/train.sh                       # released recipe
#   bash scripts/training/train.sh experiment=<name>     # overlay from configs/experiment/
#   bash scripts/training/train.sh auto_resume=true      # resume the latest checkpoint of the experiment
#
# Extra arguments are passed to Hydra, e.g. trainer.devices=2 training.batch_size=128.
set -euo pipefail
cd "$(dirname "$0")/../.."

export SCTRILEMMA_DATA_ROOT="${SCTRILEMMA_DATA_ROOT:-/scratch/$USER/datasets/cellxgene}"

pixi run python -m sctrilemma.train_vae "$@"
