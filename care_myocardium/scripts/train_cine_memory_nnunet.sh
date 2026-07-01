#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_cine_memory_nnunet.sh <fold> [extra args]

Runs the SAM2-inspired cine memory trainer on the all-frame CARE dataset.

Expected dataset:
  CARE_DATASET_ID=602
  Dataset602_CARE_CineMyoPS_AllFrames

Environment overrides:
  GPU                             CUDA_VISIBLE_DEVICES (default 0)
  CARE_DATASET_ID                 default 602
  CINE_MEMORY_EMBED_DIM           default 8
  CINE_MEMORY_QUERY_FRAME_INDEX   default 0
  CINE_MEMORY_RESIDUAL_SCALE_INIT default 0.001
  CONTINUE=1                      resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-602}"
export TRAINER="${TRAINER:-CineMemoryTrainer}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export CINE_MEMORY_EMBED_DIM="${CINE_MEMORY_EMBED_DIM:-8}"
export CINE_MEMORY_QUERY_FRAME_INDEX="${CINE_MEMORY_QUERY_FRAME_INDEX:-0}"
export CINE_MEMORY_RESIDUAL_SCALE_INIT="${CINE_MEMORY_RESIDUAL_SCALE_INIT:-0.001}"

echo "[CARE cine memory] ext_trainer=${nnUNet_extTrainer}"
echo "[CARE cine memory] embed_dim=${CINE_MEMORY_EMBED_DIM} query_frame=${CINE_MEMORY_QUERY_FRAME_INDEX}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
