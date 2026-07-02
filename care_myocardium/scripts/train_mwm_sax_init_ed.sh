#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_mwm_sax_init_ed.sh <fold> [extra args]

Fine-tune CARE Dataset601 ED from the MWM/CMR-MULTI SAX baseline checkpoint.
This uses a CARE plan with the MWM SAX network architecture so nnU-Net can load
the pretrained weights without shape mismatches.

Environment overrides:
  GPU                         CUDA_VISIBLE_DEVICES (default 0)
  CARE_DATASET_ID             default 601
  TRAINER                     default nnUNetTrainer
  PLAN_NAME                   default nnUNetPlans_MWMSAXArch
  PRETRAINED_WEIGHTS          default MWM SAX baseline final checkpoint
  MWM_PREPROCESSED            default cmr_multi_task1/DATASET/nnUNet_preprocessed
  MWMSAX_PATCH_SIZE           default "16 256 256"
  MWMSAX_BATCH_SIZE           default 2
  CONTINUE=1                  resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CARE_ROOT}/.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

FOLD="$1"; shift || true
GPU="${GPU:-0}"
CARE_DATASET_ID="${CARE_DATASET_ID:-601}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLAN_NAME="${PLAN_NAME:-nnUNetPlans_MWMSAXArch}"
MWM_PREPROCESSED="${MWM_PREPROCESSED:-${PROJECT_ROOT}/cmr_multi_task1/DATASET/nnUNet_preprocessed}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-${PROJECT_ROOT}/cmr_multi_task1/DATASET/nnUNet_result/Dataset501_CMRMULTI_CINE_SAX/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth}"
MWMSAX_PATCH_SIZE="${MWMSAX_PATCH_SIZE:-16 256 256}"
MWMSAX_BATCH_SIZE="${MWMSAX_BATCH_SIZE:-2}"

if [[ ! -f "${PRETRAINED_WEIGHTS}" ]]; then
  echo "ERROR: PRETRAINED_WEIGHTS not found: ${PRETRAINED_WEIGHTS}" >&2
  exit 1
fi

read -r PATCH_D PATCH_H PATCH_W <<< "${MWMSAX_PATCH_SIZE}"
python "${SCRIPT_DIR}/make_mwm_sax_init_plan.py" \
  --care-preprocessed "${nnUNet_preprocessed}" \
  --mwm-preprocessed "${MWM_PREPROCESSED}" \
  --care-dataset-id "${CARE_DATASET_ID}" \
  --mwm-dataset-id 501 \
  --configuration "${CONFIGURATION}" \
  --output-plans-name "${PLAN_NAME}" \
  --patch-size "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --batch-size "${MWMSAX_BATCH_SIZE}"

EXTRA=()
if [[ "${CONTINUE:-0}" == "1" ]]; then EXTRA+=(--c); fi
EXTRA+=("$@")

train_one() {
  local f="$1"
  cat <<EOF
[CARE MWM-SAX init train]
dataset_id=${CARE_DATASET_ID} fold=${f} gpu=${GPU}
trainer=${TRAINER} plans=${PLAN_NAME}
pretrained_weights=${PRETRAINED_WEIGHTS}
patch_size=${MWMSAX_PATCH_SIZE} batch_size=${MWMSAX_BATCH_SIZE}
EOF
  CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train \
    "${CARE_DATASET_ID}" "${CONFIGURATION}" "${f}" \
    -tr "${TRAINER}" -p "${PLAN_NAME}" \
    -pretrained_weights "${PRETRAINED_WEIGHTS}" \
    "${EXTRA[@]}"
}

if [[ "${FOLD}" == "all" ]]; then
  for f in 0 1 2 3 4; do train_one "${f}"; done
else
  train_one "${FOLD}"
fi
