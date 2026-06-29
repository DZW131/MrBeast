#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash cmr_multi_task1/scripts/plan_preprocess.sh <view> [extra args]

view: sax | 2ch | 4ch

Plans and preprocesses the converted nnU-Net dataset for one Cine view.
Run convert_cine_to_nnunet.py first.

Environment overrides:
  TASK1_DATASET_ROOT  default cmr_multi_task1/DATASET
  NUM_PROC            number of preprocessing workers (default 4)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: view required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VIEW="${1,,}"
DATASET_ID="$(cine_dataset_id "${VIEW}")"
if [[ -z "${DATASET_ID}" ]]; then echo "ERROR: unknown view '${VIEW}'"; exit 1; fi
shift || true
NUM_PROC="${NUM_PROC:-4}"

echo "[plan_preprocess] view=${VIEW} dataset_id=${DATASET_ID} nnUNet_raw=${nnUNet_raw}"
nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" -np "${NUM_PROC}" "$@"
