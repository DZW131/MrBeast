#!/usr/bin/env bash
set -euo pipefail

CARE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${CARE_ROOT}/.." && pwd)"
CARE_DATASET_ROOT="${CARE_DATASET_ROOT:-${CARE_ROOT}/DATASET}"
CARE_DATASET_ID="${CARE_DATASET_ID:-601}"
CARE_DATASET_NAME="${CARE_DATASET_NAME:-Dataset601_CARE_CineMyoPS}"

export CARE_ROOT
export PROJECT_ROOT
export CARE_DATASET_ROOT
export CARE_DATASET_ID
export CARE_DATASET_NAME
export nnUNet_raw="${CARE_DATASET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${CARE_DATASET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${CARE_DATASET_ROOT}/nnUNet_results"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"
