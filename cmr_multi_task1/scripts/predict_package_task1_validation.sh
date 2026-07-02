#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

SPLIT="${SPLIT:-val}"
TAG="${TAG:-baseline}"
FOLDS="${FOLDS:-0}"
GPU_SAX="${GPU_SAX:-0}"
GPU_2CH="${GPU_2CH:-1}"
GPU_4CH="${GPU_4CH:-5}"
OUT_ROOT="${OUT_ROOT:-${TASK1_DATASET_ROOT}/submission_work/${SPLIT}_${TAG}}"
POSTPROCESS="${POSTPROCESS:-1}"
ZIP_NAME="${ZIP_NAME:-submission_task1_cine_${SPLIT}_${TAG}.zip}"

mkdir -p "${OUT_ROOT}/logs"

run_predict() {
  local view="$1"
  local gpu="$2"
  local out="${OUT_ROOT}/${view}_pred"
  mkdir -p "${out}"
  GPU="${gpu}" FOLDS="${FOLDS}" RESULT_DIR="${out}" \
    bash "${SCRIPT_DIR}/predict.sh" "${view}" "${SPLIT}" \
    > "${OUT_ROOT}/logs/predict_${view}.log" 2>&1
}

echo "[task1 submission] split=${SPLIT} tag=${TAG} folds=${FOLDS}"
run_predict sax "${GPU_SAX}" &
pid_sax=$!
run_predict 2ch "${GPU_2CH}" &
pid_2ch=$!
run_predict 4ch "${GPU_4CH}" &
pid_4ch=$!
wait "${pid_sax}" "${pid_2ch}" "${pid_4ch}"

for view in sax 2ch 4ch; do
  in_dir="${OUT_ROOT}/${view}_pred"
  out_dir="${OUT_ROOT}/${view}_pred_pp"
  if [[ "${POSTPROCESS}" == "1" ]]; then
    bash "${SCRIPT_DIR}/postprocess.sh" "${view}" "${in_dir}" "${out_dir}" \
      > "${OUT_ROOT}/logs/postprocess_${view}.log" 2>&1
  else
    out_dir="${in_dir}"
  fi
done

SAX_FOR_EF="${OUT_ROOT}/sax_pred_pp"
if [[ "${POSTPROCESS}" != "1" ]]; then SAX_FOR_EF="${OUT_ROOT}/sax_pred"; fi
if [[ "${SPLIT}" == "val" ]]; then
  SLICE_INFO="${SLICE_INFO:-/data/sdb/jingkun/duyanhong/data/CMR_MULTI_dl/CINE_MULTI/id_slice_info_valid.json}"
else
  SLICE_INFO="${SLICE_INFO:-/data/sdb/jingkun/duyanhong/data/CMR_MULTI_dl/CINE_MULTI/sax_slice_info_test.json}"
fi
EF_JSON="${OUT_ROOT}/lv_ef_results.json"
python "${SCRIPT_DIR}/calculate_lvef.py" \
  --pred-dir "${SAX_FOR_EF}" \
  --slice-info "${SLICE_INFO}" \
  --output "${EF_JSON}" \
  > "${OUT_ROOT}/logs/calculate_lvef.log" 2>&1

python "${SCRIPT_DIR}/package_task1_submission.py" \
  --sax-dir "${OUT_ROOT}/sax_pred_pp" \
  --2ch-dir "${OUT_ROOT}/2ch_pred_pp" \
  --4ch-dir "${OUT_ROOT}/4ch_pred_pp" \
  --ef-json "${EF_JSON}" \
  --output-root "${TASK1_DATASET_ROOT}/submissions" \
  --zip-name "${ZIP_NAME}" \
  > "${OUT_ROOT}/logs/package.log" 2>&1

cat "${OUT_ROOT}/logs/package.log"
