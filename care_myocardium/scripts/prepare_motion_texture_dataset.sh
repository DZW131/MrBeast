#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash care_myocardium/scripts/prepare_motion_texture_dataset.sh [extra nnUNet plan args]

Creates Dataset608_CARE_CineMyoPS_MotionTexture:
  - all cine frames as texture channels
  - ED-referenced temporal difference motion channels
  - optional aggregated Farneback optical-flow motion channels

Environment overrides:
  CARE_SOURCE_ROOT     CAREdatasets root (default: repo sibling ../CAREdatasets)
  CARE_DATASET_ID      default 608
  OUTPUT_DATASET_NAME  default CARE_CineMyoPS_MotionTexture
  FRAME_INDEX          default 0
  FLOW_MODE            none or farneback_agg (default farneback_agg)
  FLOW_FRAME_STRIDE    default 1
  OVERWRITE            1 deletes existing raw Dataset608 first (default 1)
  NUM_PROC             nnU-Net preprocessing workers (default 4)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

CARE_SOURCE_ROOT="${CARE_SOURCE_ROOT:-${PROJECT_ROOT}/../CAREdatasets}"
export CARE_DATASET_ID="${CARE_DATASET_ID:-608}"
OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-CARE_CineMyoPS_MotionTexture}"
FRAME_INDEX="${FRAME_INDEX:-0}"
FLOW_MODE="${FLOW_MODE:-farneback_agg}"
FLOW_FRAME_STRIDE="${FLOW_FRAME_STRIDE:-1}"
NUM_PROC="${NUM_PROC:-4}"

CONVERT_ARGS=()
if [[ "${OVERWRITE:-1}" == "1" ]]; then
  CONVERT_ARGS+=(--overwrite)
fi

python "${SCRIPT_DIR}/convert_cine_myops_motion_texture_to_nnunet.py" \
  --data-root "${CARE_SOURCE_ROOT}" \
  --dataset-root "${CARE_DATASET_ROOT}" \
  --dataset-id "${CARE_DATASET_ID}" \
  --dataset-name "${OUTPUT_DATASET_NAME}" \
  --frame-index "${FRAME_INDEX}" \
  --flow-mode "${FLOW_MODE}" \
  --flow-frame-stride "${FLOW_FRAME_STRIDE}" \
  "${CONVERT_ARGS[@]}"

CARE_DATASET_ID="${CARE_DATASET_ID}" NUM_PROC="${NUM_PROC}" bash "${SCRIPT_DIR}/plan_preprocess.sh" "$@"
