#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  bash cmr_multi_task1/scripts/postprocess.sh <view> <input-dir> <output-dir>

Apply class-aware MR connected-component postprocessing to predicted masks.

view: sax | 2ch | 4ch

Examples:
  bash cmr_multi_task1/scripts/postprocess.sh sax \
    cmr_multi_task1/DATASET/predictions/sax_test_mae \
    cmr_multi_task1/DATASET/predictions/sax_test_mae_pp

Optional:
  RULES_OVERRIDE=/path/to/overrides.json  augment per-label rules
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 3 ]]; then echo "ERROR: view, input-dir, output-dir required"; show_help; exit 1; fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

VIEW="${1,,}"; INPUT_DIR="$2"; OUTPUT_DIR="$3"; shift 3 || true
EXTRA=()
if [[ -n "${RULES_OVERRIDE:-}" ]]; then EXTRA+=(--rules-override "${RULES_OVERRIDE}"); fi

echo "[postprocess] view=${VIEW} input=${INPUT_DIR} output=${OUTPUT_DIR}"
python "${TASK1_ROOT}/scripts/postprocess_cine.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --view "${VIEW}" \
  "${EXTRA[@]}"
