#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash care_myocardium/scripts/prepare_learned_motion_dataset.sh

Exports Dataset610_CARE_CineMyoPS_LearnedMotionTexture:
  cine frames + per-frame learned ED-referenced displacement fields.

Environment overrides:
  CARE_DATA_ROOT          path to CAREdatasets
  CARE_DATASET_ID         default 610
  OUTPUT_DATASET_NAME     default CARE_CineMyoPS_LearnedMotionTexture
  MOTION_NET_CHECKPOINT   default learned_motion/MotionNet_EDRef_1000/checkpoint_best.pth
  LEARNED_MOTION_IMAGE_SIZE default 192, matching Motion-Net training
  NUM_PROC                nnU-Net preprocess workers, default 4
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

CARE_DATA_ROOT="${CARE_DATA_ROOT:-/data/sdb/jingkun/duyanhong/CAREdatasets}"
export CARE_DATASET_ID="${CARE_DATASET_ID:-610}"
OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-CARE_CineMyoPS_LearnedMotionTexture}"
MOTION_NET_CHECKPOINT="${MOTION_NET_CHECKPOINT:-${CARE_DATASET_ROOT}/learned_motion/MotionNet_EDRef_1000/checkpoint_best.pth}"
LEARNED_MOTION_IMAGE_SIZE="${LEARNED_MOTION_IMAGE_SIZE:-192}"
NUM_PROC="${NUM_PROC:-4}"

echo "[CARE learned-motion] data=${CARE_DATA_ROOT}"
echo "[CARE learned-motion] checkpoint=${MOTION_NET_CHECKPOINT}"
echo "[CARE learned-motion] dataset_id=${CARE_DATASET_ID} name=${OUTPUT_DATASET_NAME}"
echo "[CARE learned-motion] image_size=${LEARNED_MOTION_IMAGE_SIZE}"

cd "${REPO_ROOT}"
python -m care_myocardium.learned_motion.export_nnunet \
  --data-root "${CARE_DATA_ROOT}" \
  --dataset-root "${CARE_DATASET_ROOT}" \
  --checkpoint "${MOTION_NET_CHECKPOINT}" \
  --dataset-id "${CARE_DATASET_ID}" \
  --dataset-name "${OUTPUT_DATASET_NAME}" \
  --num-frames 30 \
  --image-size "${LEARNED_MOTION_IMAGE_SIZE}" \
  --overwrite

CARE_DATASET_ID="${CARE_DATASET_ID}" NUM_PROC="${NUM_PROC}" bash "${SCRIPT_DIR}/plan_preprocess.sh"
