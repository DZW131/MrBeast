#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_learned_motion_summary_ed_cycle_nnunet.sh <fold> [extra args]

Runs Dataset612 learned-motion summary Seg-Net with first/last ED consistency.

Environment overrides:
  GPU                              CUDA_VISIBLE_DEVICES (default 0)
  NUM_GPUS                         pass -num_gpus to nnUNetv2_train for DDP/multi-GPU training
  CARE_DATASET_ID                  default 612
  LEARNED_MOTION_ED_CYCLE_EPOCHS   default 300
  LEARNED_MOTION_ED_CYCLE_WEIGHT   default 0.1
  ED_CYCLE_RAMP_EPOCHS             default 40
  ED_CYCLE_CONFIDENCE              default 0.6
  ED_CYCLE_SCAR_WEIGHT             default 2.0
  CONTINUE=1                       resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-612}"
export TRAINER="${TRAINER:-LearnedMotionSummaryEDCycleTrainer}"
export LEARNED_MOTION_ED_CYCLE_EPOCHS="${LEARNED_MOTION_ED_CYCLE_EPOCHS:-300}"
export LEARNED_MOTION_ED_CYCLE_WEIGHT="${LEARNED_MOTION_ED_CYCLE_WEIGHT:-0.1}"
export ED_CYCLE_NUM_FRAMES="${ED_CYCLE_NUM_FRAMES:-30}"
export ED_CYCLE_RAMP_EPOCHS="${ED_CYCLE_RAMP_EPOCHS:-40}"
export ED_CYCLE_CONFIDENCE="${ED_CYCLE_CONFIDENCE:-0.6}"
export ED_CYCLE_SCAR_WEIGHT="${ED_CYCLE_SCAR_WEIGHT:-2.0}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE learned-motion summary ED-cycle] dataset=${CARE_DATASET_ID} trainer=${TRAINER} epochs=${LEARNED_MOTION_ED_CYCLE_EPOCHS}"
echo "[CARE learned-motion summary ED-cycle] weight=${LEARNED_MOTION_ED_CYCLE_WEIGHT} ramp=${ED_CYCLE_RAMP_EPOCHS} confidence=${ED_CYCLE_CONFIDENCE} scar_weight=${ED_CYCLE_SCAR_WEIGHT}"
echo "[CARE learned-motion summary ED-cycle] ext_trainer=${nnUNet_extTrainer}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
