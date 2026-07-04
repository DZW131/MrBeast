#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_motion_texture_ed_cycle_nnunet.sh <fold> [extra args]

Runs the motion-texture nnU-Net trainer with first/last ED cycle consistency.

Environment overrides:
  GPU                              CUDA_VISIBLE_DEVICES (default 0)
  NUM_GPUS                         pass -num_gpus to nnUNetv2_train for DDP/multi-GPU training
  CARE_DATASET_ID                  default 609
  MOTION_TEXTURE_ED_CYCLE_EPOCHS   default 300
  MOTION_TEXTURE_ED_CYCLE_WEIGHT   default 0.1
  ED_CYCLE_CONFIDENCE              default inherited from trainer
  ED_CYCLE_SCAR_WEIGHT             default inherited from trainer
  CONTINUE=1                       resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-609}"
export TRAINER="${TRAINER:-MotionTextureEDCycleTrainer}"
export MOTION_TEXTURE_ED_CYCLE_EPOCHS="${MOTION_TEXTURE_ED_CYCLE_EPOCHS:-300}"
export MOTION_TEXTURE_ED_CYCLE_WEIGHT="${MOTION_TEXTURE_ED_CYCLE_WEIGHT:-0.1}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE motion-texture ED-cycle] dataset=${CARE_DATASET_ID} trainer=${TRAINER} epochs=${MOTION_TEXTURE_ED_CYCLE_EPOCHS} weight=${MOTION_TEXTURE_ED_CYCLE_WEIGHT}"
echo "[CARE motion-texture ED-cycle] ext_trainer=${nnUNet_extTrainer}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
