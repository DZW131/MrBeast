#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_motion_texture_nnunet.sh <fold> [extra args]

Runs the motion-texture nnU-Net baseline on Dataset608.

Environment overrides:
  GPU             CUDA_VISIBLE_DEVICES (default 0)
  CARE_DATASET_ID default 608
  TRAINER         default nnUNetTrainer
  CONTINUE=1      resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-608}"
export TRAINER="${TRAINER:-nnUNetTrainer}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE motion-texture] dataset=${CARE_DATASET_ID} trainer=${TRAINER}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
