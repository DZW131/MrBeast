#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash cmr_multi_task1/scripts/train_nnunet.sh <view> <fold> [extra args]

view: sax | 2ch | 4ch
fold: 0 1 2 3 4 (or all)

Examples:
  bash cmr_multi_task1/scripts/train_nnunet.sh sax 0
  bash cmr_multi_task1/scripts/train_nnunet.sh 4ch all

Environment overrides:
  GPU           CUDA_VISIBLE_DEVICES (default 0)
  CONFIGURATION 3d_fullres | 3d_lowres (default 3d_fullres)
  TRAINER       nnUNetTrainer (default)
  CONTINUE=1    resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 2 ]]; then echo "ERROR: view and fold required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VIEW="${1,,}"; FOLD="${2}"; shift 2 || true
DATASET_ID="$(cine_dataset_id "${VIEW}")"
if [[ -z "${DATASET_ID}" ]]; then echo "ERROR: unknown view '${VIEW}'"; exit 1; fi

GPU="${GPU:-0}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
EXTRA=()
if [[ "${CONTINUE:-0}" == "1" ]]; then EXTRA+=(--c); fi
EXTRA+=("$@")

train_one() {
  local f="$1"
  echo "[train] view=${VIEW} dataset_id=${DATASET_ID} fold=${f} gpu=${GPU} trainer=${TRAINER}"
  CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train \
    "${DATASET_ID}" "${CONFIGURATION}" "${f}" \
    -tr "${TRAINER}" "${EXTRA[@]}"
}

if [[ "${FOLD}" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "${f}"; done
else
  train_one "${FOLD}"
fi
