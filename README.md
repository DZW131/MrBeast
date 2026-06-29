# MrBeast ? CMR-MULTI Task 1 (Cine MRI)

MICCAI 2026 / 1st Workshop on Medical World Models ? CMR-MULTI Challenge,
**Task 1: Cine multi-sequence segmentation & wall motion (LVEF) analysis**.

- Challenge: https://mwm2026.github.io/cmr-multi
- Codabench: https://www.codabench.org/competitions/15533/
- Dataset: https://huggingface.co/datasets/TaipingQu/CMR-MULTI
- Official baseline: https://github.com/qutaiping/CMR_multi_baseline

## Pipeline

nnU-Net v2 initialized from the third-party **ResEncL-OpenMind-MAE**
self-supervised checkpoint ? fine-tune on Cine data ? **class-aware MR
postprocessing** ? LVEF quantification. See
[`cmr_multi_task1/README.md`](cmr_multi_task1/README.md) for the full runbook.

```
convert_cine_to_nnunet.py        # CINE_MULTI -> 3 nnU-Net datasets (SAX/2CH/4CH)
preprocess_cine_mae_pretrained.sh# align datasets to the MAE checkpoint plan
train_cine_mae_pretrained.sh     # MAE fine-tune (ResEncL) per view / fold
predict.sh                       # export val/test masks (MAE=1)
postprocess.sh                   # class-aware MR connected-component cleanup
calculate_lvef.py                # LVEF from postprocessed SAX masks
evaluate_val.py                  # Dice / HD95 on held-out validation
```

## Repository layout

```
cmr_multi_task1/
  README.md                      # full runbook
  configs/
    cine_labels.json             # per-view label space & dataset ids
    cine_postprocess_rules.json  # per-class MR postprocessing rules
  scripts/                       # conversion / env / train / predict / postprocess / metrics
third_party/                     # NOT tracked ? see below
pretrained/                      # NOT tracked ? see below
```

## One-time external dependencies (not committed)

These are gitignored. Set them up at the repo root before running:

```bash
# 1. TaWald nnU-Net nnSSL fine-tuning branch (provides nnUNetv2_train_pretrained)
git clone -b nnssl_finetuning_inclusion https://github.com/TaWald/nnUNet.git third_party/TaWald_nnUNet_nnssl

# 2. ResEncL-OpenMind-MAE checkpoint (MIC-DKFZ, ~468 MB)
pip install huggingface_hub
python - <<'PY'
import os
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
from huggingface_hub import snapshot_download
snapshot_download(repo_id="MIC-DKFZ/ResEncL-OpenMind-MAE",
                  local_dir="pretrained/MAE/ResEncL-OpenMind-MAE")
PY
```

Expected layout after setup:

```
third_party/TaWald_nnUNet_nnssl/
pretrained/MAE/ResEncL-OpenMind-MAE/checkpoint_final.pth
pretrained/MAE/ResEncL-OpenMind-MAE/adaptation_plan.json
```

## Environment (server)

Two conda envs:

- `cmr_multi` ? stock nnU-Net v2 (conversion, eval, plain baseline).
- `cmr_multi_mae` ? the TaWald nnSSL branch installed editable
  (`pip install -e third_party/TaWald_nnUNet_nnssl`). `env_mae.sh` points
  `MAE_CONDA_PREFIX` here.

See `cmr_multi_task1/README.md` for the step-by-step commands.
