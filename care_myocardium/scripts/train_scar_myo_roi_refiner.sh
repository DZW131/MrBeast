#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_scar_myo_roi_refiner.sh <fold> [extra args]

Runs the Stage 3 CARE scar refiner with Stage 2 myocardium prior on Dataset605.

Environment overrides:
  GPU             CUDA_VISIBLE_DEVICES (default 0)
  CARE_DATASET_ID default 605
  SCAR_ROI_EPOCHS default 300; ROI_EPOCHS also accepted
  CONTINUE=1      resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-605}"
export TRAINER="${TRAINER:-ScarROI300EpochTrainer}"
export SCAR_ROI_EPOCHS="${SCAR_ROI_EPOCHS:-${ROI_EPOCHS:-300}}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE Stage3 scar+myo ROI refiner] dataset=${CARE_DATASET_ID} trainer=${TRAINER} epochs=${SCAR_ROI_EPOCHS}"
echo "[CARE Stage3 scar+myo ROI refiner] ext_trainer=${nnUNet_extTrainer}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
