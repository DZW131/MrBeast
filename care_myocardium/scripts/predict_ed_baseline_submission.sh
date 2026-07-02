#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

TEAM_NAME="${TEAM_NAME:-Monster}"
GPU="${GPU:-0}"
FOLD="${FOLD:-0}"
FRAME_INDEX="${FRAME_INDEX:-0}"
PLAN="${PLAN:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainer}"
CHECKPOINT="${CHECKPOINT:-checkpoint_final.pth}"
DATASET_ID="${CARE_DATASET_ID:-601}"

if [[ -z "${CARE_CINEMYOPS_INFER_ROOT:-}" ]]; then
  echo "Set CARE_CINEMYOPS_INFER_ROOT to the official CineMyoPS validation/test image folder." >&2
  exit 2
fi

WORK_DIR="${CARE_SUBMISSION_WORK_DIR:-${CARE_DATASET_ROOT}/submission_work/ed_baseline_${TEAM_NAME}}"
IMAGES_TS="${WORK_DIR}/imagesTs"
MANIFEST="${WORK_DIR}/cine_myops_inference_manifest.json"
PRED_DIR="${WORK_DIR}/pred_nnunet"
SUBMISSION_ROOT="${CARE_SUBMISSION_ROOT:-${CARE_DATASET_ROOT}/submissions}"

mkdir -p "${WORK_DIR}" "${PRED_DIR}" "${SUBMISSION_ROOT}"

python "${SCRIPT_DIR}/prepare_cine_myops_inference.py" \
  --data-root "${CARE_CINEMYOPS_INFER_ROOT}" \
  --output-dir "${IMAGES_TS}" \
  --manifest "${MANIFEST}" \
  --frame-index "${FRAME_INDEX}" \
  --overwrite

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_predict \
  -i "${IMAGES_TS}" \
  -o "${PRED_DIR}" \
  -d "${DATASET_ID}" \
  -c 3d_fullres \
  -f "${FOLD}" \
  -tr "${TRAINER}" \
  -p "${PLAN}" \
  -chk "${CHECKPOINT}"

python "${SCRIPT_DIR}/package_care_myocardium_submission.py" \
  --pred-dir "${PRED_DIR}" \
  --manifest "${MANIFEST}" \
  --team-name "${TEAM_NAME}" \
  --output-root "${SUBMISSION_ROOT}" \
  --input-label-space auto
