#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-611}"
export OUTPUT_DATASET_NAME="${OUTPUT_DATASET_NAME:-CARE_CineMyoPS_LearnedMotionFramewise}"
export LEARNED_MOTION_FUSION_MODE="${LEARNED_MOTION_FUSION_MODE:-framewise_concat}"

bash "${SCRIPT_DIR}/prepare_learned_motion_dataset.sh"
