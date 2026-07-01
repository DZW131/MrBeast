#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash care_myocardium/scripts/plan_preprocess.sh [extra nnUNet args]

Run after convert_cine_myops_to_nnunet.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
NUM_PROC="${NUM_PROC:-4}"

echo "[CARE plan_preprocess] dataset_id=${CARE_DATASET_ID} nnUNet_raw=${nnUNet_raw}"
nnUNetv2_plan_and_preprocess -d "${CARE_DATASET_ID}" -np "${NUM_PROC}" "$@"
