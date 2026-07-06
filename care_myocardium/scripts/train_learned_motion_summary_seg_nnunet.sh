#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-612}"
export TRAINER="${TRAINER:-LearnedMotionSeg400EpochTrainer}"

bash "${SCRIPT_DIR}/train_learned_motion_seg_nnunet.sh" "$@"
