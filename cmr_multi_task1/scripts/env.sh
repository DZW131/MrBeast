#!/usr/bin/env bash
# Base environment for CMR-MULTI Task 1 (Cine MRI) nnU-Net workflow.
# Source this from the project root before running plan/train/predict scripts.

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _ENV_FILE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  eval '_ENV_FILE="${(%):-%x}"'
else
  _ENV_FILE="$0"
fi

TASK1_ROOT="${TASK1_ROOT:-$(cd "$(dirname "${_ENV_FILE}")/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${TASK1_ROOT}/.." && pwd)}"

export TASK1_ROOT
export PROJECT_ROOT

TASK1_DATASET_ROOT="${TASK1_DATASET_ROOT:-${TASK1_ROOT}/DATASET}"
export TASK1_DATASET_ROOT
export nnUNet_raw="${nnUNet_raw:-${TASK1_DATASET_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${TASK1_DATASET_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${TASK1_DATASET_ROOT}/nnUNet_result}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# View -> dataset id map (kept in sync with configs/cine_labels.json)
declare -gA CINE_DATASET_IDS=(
  [sax]=501
  [2ch]=502
  [4ch]=503
)

cine_dataset_id() {
  local view="${1,,}"
  echo "${CINE_DATASET_IDS[$view]:-}"
}

unset _ENV_FILE
