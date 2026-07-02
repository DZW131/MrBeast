#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

export CARE_DATASET_ID="${CARE_DATASET_ID:-606}"
ED_DATASET_NAME="${ED_DATASET_NAME:-Dataset601_CARE_CineMyoPS_ED}"
OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-CARE_CineMyoPS_ScarMyoDilatedROI_ED}"
STAGE1_PRED_DIR="${STAGE1_PRED_DIR:-${CARE_DATASET_ROOT}/scar_roi_stage1/nnUNetTrainer__nnUNetPlans__checkpoint_final.pth/train_predictions}"
STAGE2_MYO_PRED_DIR="${STAGE2_MYO_PRED_DIR:-${CARE_DATASET_ROOT}/scar_roi_stage2_myo/MyoROI300EpochTrainer__nnUNetPlans__checkpoint_final.pth/full_predictions}"
MARGIN_XY="${MARGIN_XY:-48}"
MIN_XY="${MIN_XY:-128}"
PRIOR_DILATION_XY="${PRIOR_DILATION_XY:-4}"
NUM_PROC="${NUM_PROC:-4}"

python "${SCRIPT_DIR}/generate_scar_roi_dataset.py" \
  --target scar \
  --dataset-root "${CARE_DATASET_ROOT}" \
  --ed-dataset-name "${ED_DATASET_NAME}" \
  --stage1-pred-dir "${STAGE1_PRED_DIR}" \
  --stage2-myo-pred-dir "${STAGE2_MYO_PRED_DIR}" \
  --target-prior-mode dilate_xy \
  --prior-dilation-xy "${PRIOR_DILATION_XY}" \
  --output-dataset-id "${CARE_DATASET_ID}" \
  --output-dataset-name "${OUTPUT_DATASET_NAME}" \
  --margin-xy "${MARGIN_XY}" \
  --min-xy "${MIN_XY}" \
  --overwrite

CARE_DATASET_ID="${CARE_DATASET_ID}" NUM_PROC="${NUM_PROC}" bash "${SCRIPT_DIR}/plan_preprocess.sh"
