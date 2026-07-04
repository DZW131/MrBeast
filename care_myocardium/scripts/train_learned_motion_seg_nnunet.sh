#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_learned_motion_seg_nnunet.sh <fold> [extra args]

Runs the Seg-Net stage for learned motion + cine texture channels.

Environment overrides:
  GPU                         CUDA_VISIBLE_DEVICES (default 0)
  NUM_GPUS                    pass -num_gpus to nnUNetv2_train for DDP/multi-GPU training
  CARE_DATASET_ID             default 610
  LEARNED_MOTION_SEG_EPOCHS   default 400
  CONTINUE=1                  resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-610}"
export TRAINER="${TRAINER:-LearnedMotionSeg400EpochTrainer}"
export LEARNED_MOTION_SEG_EPOCHS="${LEARNED_MOTION_SEG_EPOCHS:-400}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE learned-motion Seg-Net] dataset=${CARE_DATASET_ID} trainer=${TRAINER} epochs=${LEARNED_MOTION_SEG_EPOCHS}"
echo "[CARE learned-motion Seg-Net] ext_trainer=${nnUNet_extTrainer}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
