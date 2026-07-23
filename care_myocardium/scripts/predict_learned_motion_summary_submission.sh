#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

TEAM_NAME="${TEAM_NAME:-Monster}"
GPU="${GPU:-0}"
FOLD="${FOLD:-0}"
PLAN="${PLAN:-nnUNetPlans}"
TRAINER="${TRAINER:-LearnedMotionSeg400EpochTrainer}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best_scar.pth}"
DATASET_ID="${LEARNED_MOTION_SUMMARY_DATASET_ID:-612}"
MOTION_CHECKPOINT="${MOTION_CHECKPOINT:-${CARE_DATASET_ROOT}/learned_motion/MotionNet_EDRef_1000/checkpoint_best.pth}"

if [[ -z "${CARE_CINEMYOPS_INFER_ROOT:-}" ]]; then
  echo "Set CARE_CINEMYOPS_INFER_ROOT to the official CineMyoPS validation/test image folder." >&2
  exit 2
fi

WORK_DIR="${CARE_SUBMISSION_WORK_DIR:-${CARE_DATASET_ROOT}/submission_work/learned_motion_summary_${TEAM_NAME}}"
IMAGES_TS="${WORK_DIR}/imagesTs"
MANIFEST="${WORK_DIR}/cine_myops_inference_manifest.json"
PRED_DIR="${WORK_DIR}/pred_nnunet"
SUBMISSION_ROOT="${CARE_SUBMISSION_ROOT:-${CARE_DATASET_ROOT}/submissions}"

mkdir -p "${WORK_DIR}" "${PRED_DIR}" "${SUBMISSION_ROOT}"

python "${SCRIPT_DIR}/prepare_learned_motion_summary_inference.py" \
  --data-root "${CARE_CINEMYOPS_INFER_ROOT}" \
  --checkpoint "${MOTION_CHECKPOINT}" \
  --output-dir "${IMAGES_TS}" \
  --manifest "${MANIFEST}" \
  --overwrite

export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
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
