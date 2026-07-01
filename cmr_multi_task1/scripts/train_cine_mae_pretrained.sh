#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 FOLD=0 bash cmr_multi_task1/scripts/train_cine_mae_pretrained.sh <view> <fold>

Fine-tune one Cine view from the nnSSL/MAE checkpoint with the TaWald nnU-Net
branch. Run preprocess_cine_mae_pretrained.sh first.

view: sax | 2ch | 4ch
fold: 0 1 2 3 4 (or all)

Environment overrides:
  GPU=0
  MAE_TRAINER=PretrainedTrainer
  MAE_PATCH_COMPAT=1   apply TaWald PretrainedTrainer compatibility patches
  CONTINUE=1           resume from latest checkpoint
  SAVE_NPZ=1           save npz for ensembling
  MAE_PLANS_NAME=      explicit ptPlans name (auto-detected if empty)
  MAE_REQUIRE_SAFE_PLAN=1  require a cine safe-patch ptPlans JSON
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 2 ]]; then echo "ERROR: view and fold required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env_mae.sh"

VIEW="${1,,}"; FOLD="${2}"; shift 2 || true
DATASET_ID="$(cine_dataset_id "${VIEW}")"
if [[ -z "${DATASET_ID}" ]]; then echo "ERROR: unknown view '${VIEW}'"; exit 1; fi

GPU="${GPU:-0}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
MAE_TRAINER="${MAE_TRAINER:-PretrainedTrainer}"
MAE_PATCH_COMPAT="${MAE_PATCH_COMPAT:-1}"
SAVE_NPZ="${SAVE_NPZ:-1}"
CONTINUE="${CONTINUE:-0}"
MAE_REQUIRE_SAFE_PLAN="${MAE_REQUIRE_SAFE_PLAN:-1}"

DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"
if [[ -z "${DATASET_DIR}" || ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: Could not find preprocessed Dataset${DATASET_ID}_* under ${nnUNet_preprocessed}" >&2; exit 1
fi

if [[ -z "${MAE_PLANS_NAME:-}" ]]; then
  MAE_PLANS_JSON="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name "ptPlans__${MAE_PRETRAINING_NAME}*.json" | sort | tail -n 1)"
  if [[ -z "${MAE_PLANS_JSON}" ]]; then
    echo "ERROR: No MAE plans found. Run preprocess_cine_mae_pretrained.sh first." >&2; exit 1
  fi
  MAE_PLANS_NAME="$(basename "${MAE_PLANS_JSON}" .json)"
fi
if [[ "${MAE_REQUIRE_SAFE_PLAN}" == "1" && "${MAE_PLANS_NAME}" != *"____Patch__"* ]]; then
  echo "ERROR: MAE plan '${MAE_PLANS_NAME}' is missing the cine safe-patch suffix." >&2
  echo "Run preprocess_cine_mae_pretrained.sh, or set MAE_REQUIRE_SAFE_PLAN=0 to use it anyway." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${SAVE_NPZ}" == "1" ]]; then EXTRA_ARGS+=(--npz); fi
if [[ "${CONTINUE}" == "1" ]]; then EXTRA_ARGS+=(--c); fi
EXTRA_ARGS+=("$@")

if [[ "${MAE_PATCH_COMPAT}" == "1" ]]; then
  bash "$(dirname "${BASH_SOURCE[0]}")/patch_mae_trainer_compat.sh"
fi

train_one() {
  local f="$1"
  cat <<EOF
[cine MAE fine-tune]
view=${VIEW} dataset_id=${DATASET_ID} fold=${f} gpu=${GPU}
trainer=${MAE_TRAINER} plans=${MAE_PLANS_NAME}
nnUNet_results=${nnUNet_results} continue=${CONTINUE} save_npz=${SAVE_NPZ}
EOF
  CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train_pretrained \
    "${DATASET_ID}" "${CONFIGURATION}" "${f}" \
    -tr "${MAE_TRAINER}" -p "${MAE_PLANS_NAME}" \
    "${EXTRA_ARGS[@]}"
}

if [[ "${FOLD}" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "${f}"; done
else
  train_one "${FOLD}"
fi
