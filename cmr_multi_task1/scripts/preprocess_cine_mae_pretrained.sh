#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash cmr_multi_task1/scripts/preprocess_cine_mae_pretrained.sh <view>

view: sax | 2ch | 4ch

Prepare one Cine view dataset for MAE-pretrained ResEnc fine-tuning using the
TaWald/nnU-Net nnSSL adaptation branch. Run convert_cine_to_nnunet.py and
plan_preprocess.sh first (a base nnUNetPlans.json must exist).

Environment overrides:
  MAE_CHECKPOINT=/path/to/checkpoint_final.pth
  MAE_PRETRAINING_NAME=ResEncL_OpenMind_MAE
  MAE_ADAPTATION_MODE=default_nnunet|like_pretrained|no_resample|fixed
  MAE_NUM_PROCESSES=4
  MAE_FORCE_PLAN=1   rerun base nnUNet planning even if plans exist
  MAE_SAFE_CINE_PATCH=1  create a cine-shaped MAE plan instead of generic 160^3
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: view required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env_mae.sh"

VIEW="${1,,}"
DATASET_ID="$(cine_dataset_id "${VIEW}")"
if [[ -z "${DATASET_ID}" ]]; then echo "ERROR: unknown view '${VIEW}'"; exit 1; fi

MAE_ADAPTATION_MODE="${MAE_ADAPTATION_MODE:-default_nnunet}"
MAE_NUM_PROCESSES="${MAE_NUM_PROCESSES:-4}"
MAE_FORCE_PLAN="${MAE_FORCE_PLAN:-0}"
MAE_SAFE_CINE_PATCH="${MAE_SAFE_CINE_PATCH:-1}"
MAE_PATCH_STRIDE_MULTIPLE="${MAE_PATCH_STRIDE_MULTIPLE:-32}"

if [[ ! -d "${MAE_NNUNET_ROOT}" ]]; then
  echo "ERROR: MAE_NNUNET_ROOT does not exist: ${MAE_NNUNET_ROOT}" >&2; exit 1
fi
if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
  echo "ERROR: MAE_CHECKPOINT does not exist: ${MAE_CHECKPOINT}" >&2; exit 1
fi

DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"

cat <<EOF
[cine MAE preprocess]
dataset_id=${DATASET_ID} view=${VIEW}
mae_nnunet_root=${MAE_NNUNET_ROOT}
checkpoint=${MAE_CHECKPOINT}
pretraining_name=${MAE_PRETRAINING_NAME}
adaptation_mode=${MAE_ADAPTATION_MODE}
safe_cine_patch=${MAE_SAFE_CINE_PATCH}
nnUNet_preprocessed=${nnUNet_preprocessed}
nnUNet_results=${nnUNet_results}
EOF

if [[ "${MAE_FORCE_PLAN}" == "1" || -z "${DATASET_DIR}" || ! -f "${DATASET_DIR}/nnUNetPlans.json" ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --no_pp
  DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"
else
  echo "Found existing nnUNet plans, skipping base planning: ${DATASET_DIR}/nnUNetPlans.json"
fi

nnUNetv2_preprocess_like_nnssl \
  -d "${DATASET_ID}" \
  -n "${MAE_PRETRAINING_NAME}" \
  -pc "${MAE_CHECKPOINT}" \
  -am "${MAE_ADAPTATION_MODE}" \
  -np "${MAE_NUM_PROCESSES}" \
  --verbose

if [[ "${MAE_SAFE_CINE_PATCH}" == "1" ]]; then
  python "$(dirname "${BASH_SOURCE[0]}")/make_cine_mae_safe_plan.py" \
    --dataset-dir "${DATASET_DIR}" \
    --pretraining-name "${MAE_PRETRAINING_NAME}" \
    --stride-multiple "${MAE_PATCH_STRIDE_MULTIPLE}" \
    --force
fi

echo "Available MAE plans:"
find "${DATASET_DIR}" -maxdepth 1 -type f -name "ptPlans__${MAE_PRETRAINING_NAME}*.json" -printf "%f\n" 2>/dev/null | sort
