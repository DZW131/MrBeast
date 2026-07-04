#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<EOF
Usage:
  GPU=0,1,2,3 bash care_myocardium/scripts/train_motion_net.sh

Trains the learned ED-referenced Motion-Net used by the strict MTI-MyoScarSeg route.

Environment overrides:
  CARE_DATA_ROOT       path to CAREdatasets
  MOTION_NET_OUT       output checkpoint directory
  MOTION_NET_EPOCHS    default 1000
  MOTION_NET_BATCH     default 16 per process
  MOTION_NET_GPUS      number of DDP processes; default derived from GPU
  MOTION_NET_WORKERS   default 4
  MOTION_NET_LR        default 5e-4
  MOTION_NET_IMAGE_SIZE default 192
  SMOOTH_WEIGHT        default 0.05
  CACHE_IN_MEMORY=1    keep cine volumes in RAM per worker
  RESUME=1             resume checkpoint_latest.pth
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then show_help; exit 0; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

GPU="${GPU:-0}"
IFS=',' read -r -a GPU_LIST <<< "${GPU}"
NUM_GPUS="${MOTION_NET_GPUS:-${#GPU_LIST[@]}}"
CARE_DATA_ROOT="${CARE_DATA_ROOT:-${CARE_DATASET_ROOT:-/data/sdb/jingkun/duyanhong/CAREdatasets}}"
MOTION_NET_OUT="${MOTION_NET_OUT:-${CARE_DATASET_ROOT}/learned_motion/MotionNet_EDRef_1000}"
MOTION_NET_EPOCHS="${MOTION_NET_EPOCHS:-1000}"
MOTION_NET_BATCH="${MOTION_NET_BATCH:-16}"
MOTION_NET_WORKERS="${MOTION_NET_WORKERS:-4}"
MOTION_NET_LR="${MOTION_NET_LR:-0.0005}"
MOTION_NET_IMAGE_SIZE="${MOTION_NET_IMAGE_SIZE:-192}"
SMOOTH_WEIGHT="${SMOOTH_WEIGHT:-0.05}"

EXTRA=()
if [[ "${CACHE_IN_MEMORY:-0}" == "1" ]]; then EXTRA+=(--cache-in-memory); fi
if [[ "${RESUME:-0}" == "1" ]]; then EXTRA+=(--resume); fi

echo "[CARE Motion-Net] data=${CARE_DATA_ROOT}"
echo "[CARE Motion-Net] out=${MOTION_NET_OUT}"
echo "[CARE Motion-Net] epochs=${MOTION_NET_EPOCHS} gpu=${GPU} num_gpus=${NUM_GPUS} batch=${MOTION_NET_BATCH}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  -m care_myocardium.learned_motion.train_motion_net \
  --data-root "${CARE_DATA_ROOT}" \
  --output-dir "${MOTION_NET_OUT}" \
  --epochs "${MOTION_NET_EPOCHS}" \
  --batch-size "${MOTION_NET_BATCH}" \
  --lr "${MOTION_NET_LR}" \
  --smooth-weight "${SMOOTH_WEIGHT}" \
  --image-size "${MOTION_NET_IMAGE_SIZE}" \
  --num-workers "${MOTION_NET_WORKERS}" \
  "${EXTRA[@]}"
