#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0 bash care_myocardium/scripts/train_ed_cycle_nnunet.sh <fold> [extra args]

Runs the first-ED <-> last-ED cycle-consistency trainer on Dataset608.

Expected dataset:
  CARE_DATASET_ID=608
  Dataset608_CARE_CineMyoPS_MotionTexture

Environment overrides:
  GPU                   CUDA_VISIBLE_DEVICES (default 0)
  CARE_DATASET_ID       default 608
  ED_CYCLE_EPOCHS       default 300
  ED_CYCLE_WEIGHT       default 0.2
  ED_CYCLE_RAMP_EPOCHS  default 40
  ED_CYCLE_CONFIDENCE   default 0.6
  ED_CYCLE_SCAR_WEIGHT  default 2.0
  CONTINUE=1            resume from latest checkpoint
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi
if [[ $# -lt 1 ]]; then echo "ERROR: fold required"; show_help; exit 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CARE_DATASET_ID="${CARE_DATASET_ID:-608}"
export TRAINER="${TRAINER:-EDCycleConsistencyTrainer}"
export nnUNet_extTrainer="${CARE_ROOT}/nnunet_ext${nnUNet_extTrainer:+:${nnUNet_extTrainer}}"
export ED_CYCLE_EPOCHS="${ED_CYCLE_EPOCHS:-300}"
export ED_CYCLE_NUM_FRAMES="${ED_CYCLE_NUM_FRAMES:-30}"
export ED_CYCLE_WEIGHT="${ED_CYCLE_WEIGHT:-0.2}"
export ED_CYCLE_RAMP_EPOCHS="${ED_CYCLE_RAMP_EPOCHS:-40}"
export ED_CYCLE_CONFIDENCE="${ED_CYCLE_CONFIDENCE:-0.6}"
export ED_CYCLE_SCAR_WEIGHT="${ED_CYCLE_SCAR_WEIGHT:-2.0}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export nnUNet_compile="${nnUNet_compile:-f}"

echo "[CARE ED-cycle] dataset=${CARE_DATASET_ID} trainer=${TRAINER} epochs=${ED_CYCLE_EPOCHS}"
echo "[CARE ED-cycle] weight=${ED_CYCLE_WEIGHT} ramp=${ED_CYCLE_RAMP_EPOCHS} confidence=${ED_CYCLE_CONFIDENCE} scar_weight=${ED_CYCLE_SCAR_WEIGHT}"
echo "[CARE ED-cycle] ext_trainer=${nnUNet_extTrainer}"
bash "${SCRIPT_DIR}/train_nnunet.sh" "$@"
