#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_mae.sh"

TARGET="${MAE_NNUNET_ROOT}/nnunetv2/training/nnUNetTrainer/pretraining/pretrainedTrainer.py"
if [[ ! -f "${TARGET}" ]]; then
  echo "ERROR: Cannot find PretrainedTrainer source: ${TARGET}" >&2; exit 1
fi

if grep -Fq 'for c in cit["apa_citations"]:' "${TARGET}"; then
  cp -n "${TARGET}" "${TARGET}.orig"
  perl -0pi -e 's/for c in cit\["apa_citations"\]:/for c in cit.get("apa_citations", cit.get("bibtex_citations", [])):/g' "${TARGET}"
  echo "Patched MAE PretrainedTrainer citation compatibility: ${TARGET}"
else
  echo "MAE PretrainedTrainer already patched or upstream changed: ${TARGET}"
fi

if ! grep -Fq 'lpe_in_encoder = False' "${TARGET}"; then
  cp -n "${TARGET}" "${TARGET}.orig"
  perl -0pi -e 's/        lpe_in_stem = False\n/        lpe_in_encoder = False\n        lpe_in_stem = False\n/g' "${TARGET}"
  echo "Patched MAE PretrainedTrainer LPE guard for non-LPE ResEnc checkpoints: ${TARGET}"
else
  echo "MAE PretrainedTrainer LPE guard already patched or upstream changed: ${TARGET}"
fi
