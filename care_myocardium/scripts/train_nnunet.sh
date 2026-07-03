#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_nnunet.sh <fold> [extra args]

fold: 0 1 2 3 4 or all

Environment overrides:
  GPU           CUDA_VISIBLE_DEVICES (default 0)
  CONFIGURATION 3d_fullres (default)
  TRAINER       nnUNetTrainer (default)
  CONTINUE=1    resume from latest checkpoint
  nnUNet_n_proc_DA data augmentation worker count
  nnUNet_compile   torch.compile toggle
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

FOLD="$1"; shift || true
GPU="${GPU:-0}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
EXTRA=()
if [[ "${CONTINUE:-0}" == "1" ]]; then EXTRA+=(--c); fi
EXTRA+=("$@")

if [[ "${CARE_DATASET_ID}" == "602" || "${CARE_DATASET_ID}" == "608" ]]; then
  export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
  export nnUNet_compile="${nnUNet_compile:-f}"
fi

train_one() {
  local f="$1"
  echo "[CARE train] dataset_id=${CARE_DATASET_ID} fold=${f} gpu=${GPU} trainer=${TRAINER} n_proc_DA=${nnUNet_n_proc_DA:-default} compile=${nnUNet_compile:-default}"
  CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train \
    "${CARE_DATASET_ID}" "${CONFIGURATION}" "${f}" \
    -tr "${TRAINER}" "${EXTRA[@]}"
}

if [[ "${FOLD}" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "${f}"; done
else
  train_one "${FOLD}"
fi
