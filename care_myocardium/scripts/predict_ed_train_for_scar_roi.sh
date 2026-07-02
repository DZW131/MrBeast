#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

GPU="${GPU:-0}"
FOLD="${FOLD:-0}"
STAGE1_DATASET_ID="${STAGE1_DATASET_ID:-601}"
STAGE1_CONFIGURATION="${STAGE1_CONFIGURATION:-3d_fullres}"
STAGE1_TRAINER="${STAGE1_TRAINER:-nnUNetTrainer}"
STAGE1_PLAN="${STAGE1_PLAN:-nnUNetPlans}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-checkpoint_final.pth}"
ED_DATASET_NAME="${ED_DATASET_NAME:-Dataset601_CARE_CineMyoPS_ED}"
STAGE1_PRED_DIR="${STAGE1_PRED_DIR:-${CARE_DATASET_ROOT}/scar_roi_stage1/${STAGE1_TRAINER}__${STAGE1_PLAN}__${STAGE1_CHECKPOINT}/train_predictions}"

INPUT_DIR="${nnUNet_raw}/${ED_DATASET_NAME}/imagesTr"
mkdir -p "${STAGE1_PRED_DIR}"

echo "[CARE scar ROI stage1 predict]"
echo "input=${INPUT_DIR}"
echo "output=${STAGE1_PRED_DIR}"
echo "dataset=${STAGE1_DATASET_ID} trainer=${STAGE1_TRAINER} plan=${STAGE1_PLAN} checkpoint=${STAGE1_CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${STAGE1_PRED_DIR}" \
  -d "${STAGE1_DATASET_ID}" \
  -c "${STAGE1_CONFIGURATION}" \
  -f "${FOLD}" \
  -tr "${STAGE1_TRAINER}" \
  -p "${STAGE1_PLAN}" \
  -chk "${STAGE1_CHECKPOINT}"

echo "${STAGE1_PRED_DIR}"
