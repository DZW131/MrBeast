#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

GPU="${GPU:-0}"
FOLD="${FOLD:-0}"
MYO_DATASET_ID="${MYO_DATASET_ID:-604}"
MYO_DATASET_NAME="${MYO_DATASET_NAME:-Dataset604_CARE_CineMyoPS_MyoROI_ED}"
MYO_CONFIGURATION="${MYO_CONFIGURATION:-3d_fullres}"
MYO_TRAINER="${MYO_TRAINER:-MyoROI300EpochTrainer}"
MYO_PLAN="${MYO_PLAN:-nnUNetPlans}"
MYO_CHECKPOINT="${MYO_CHECKPOINT:-checkpoint_final.pth}"
ED_DATASET_NAME="${ED_DATASET_NAME:-Dataset601_CARE_CineMyoPS_ED}"
PRED_ROOT="${PRED_ROOT:-${CARE_DATASET_ROOT}/scar_roi_stage2_myo/${MYO_TRAINER}__${MYO_PLAN}__${MYO_CHECKPOINT}}"
CROP_PRED_DIR="${CROP_PRED_DIR:-${PRED_ROOT}/crop_predictions}"
FULL_PRED_DIR="${FULL_PRED_DIR:-${PRED_ROOT}/full_predictions}"

export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"

INPUT_DIR="${nnUNet_raw}/${MYO_DATASET_NAME}/imagesTr"
ROI_DATASET_DIR="${nnUNet_raw}/${MYO_DATASET_NAME}"
REFERENCE_DATASET_DIR="${nnUNet_raw}/${ED_DATASET_NAME}"
mkdir -p "${CROP_PRED_DIR}" "${FULL_PRED_DIR}"

echo "[CARE Stage2 myo predict for Stage3]"
echo "input=${INPUT_DIR}"
echo "crop_output=${CROP_PRED_DIR}"
echo "full_output=${FULL_PRED_DIR}"
echo "dataset=${MYO_DATASET_ID} trainer=${MYO_TRAINER} checkpoint=${MYO_CHECKPOINT}"

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${CROP_PRED_DIR}" \
  -d "${MYO_DATASET_ID}" \
  -c "${MYO_CONFIGURATION}" \
  -f "${FOLD}" \
  -tr "${MYO_TRAINER}" \
  -p "${MYO_PLAN}" \
  -chk "${MYO_CHECKPOINT}"

python "${SCRIPT_DIR}/restore_roi_predictions_to_full.py" \
  --roi-dataset-dir "${ROI_DATASET_DIR}" \
  --crop-pred-dir "${CROP_PRED_DIR}" \
  --reference-dataset-dir "${REFERENCE_DATASET_DIR}" \
  --output-dir "${FULL_PRED_DIR}" \
  --overwrite

echo "${FULL_PRED_DIR}"
