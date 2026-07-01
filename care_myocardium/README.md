# CARE CineMyoPS

This folder prepares the MICCAI CARE 2026 myocardial pathology cine task for
nnU-Net experiments.

## Dataset Facts

Local training data inspected at `D:/work/CMR-MULTI/CAREdatasets` contains two
subtasks:

- `CineMyoPS_train`: cine-only task, 64 cases total.
  - `center_alpha`: 40 cases.
  - `center_beta`: 24 cases.
  - Each case has one 4D cine image `CaseXXXX_Cine.nii.gz` with shape
    `(H, W, Z, 30)` and one 3D label `CaseXXXX_gd.nii.gz` with shape
    `(H, W, Z)`.
- `MyoPS_train`: separate multi-modal task with LGE/C0/T2 combinations. Do not
  mix this into the first CineMyoPS baseline.

For CineMyoPS, labels in the released masks are:

| Source value | Meaning |
| --- | --- |
| 0 | background |
| 200 | LV normal myocardium |
| 500 | LV blood pool |
| 2221 | LV myocardial scar |

The source labels are remapped to contiguous nnU-Net labels:

| nnU-Net value | Meaning | Source value |
| --- | --- | --- |
| 0 | background | 0 |
| 1 | LV normal myocardium | 200 |
| 2 | LV blood pool | 500 |
| 3 | LV myocardial scar | 2221 |

One training case has no scar voxels; the other 63 have scar label 2221.

## Important Annotation Detail

The cine image is 4D, but the ground truth is only 3D. The challenge describes
the target as the end-diastolic frame. The local NIfTI headers do not expose an
explicit ED frame index, so `convert_cine_myops_to_nnunet.py` makes the frame
choice configurable and defaults to frame 0.

Use `--mode ed` for a simple 3D baseline. Use `--mode all_frames` to export all
30 cine frames as 30 nnU-Net input channels. Keep these as separate nnU-Net
dataset ids so preprocessing and checkpoints do not overwrite each other.

## Runbook

From the repo root:

```bash
python care_myocardium/scripts/convert_cine_myops_to_nnunet.py \
  --data-root /path/to/CAREdatasets \
  --dataset-root care_myocardium/DATASET \
  --mode ed \
  --dataset-id 601 \
  --dataset-name CARE_CineMyoPS_ED \
  --frame-index 0

CARE_DATASET_ID=601 bash care_myocardium/scripts/plan_preprocess.sh
GPU=0 CARE_DATASET_ID=601 bash care_myocardium/scripts/train_nnunet.sh 0
```

Motion-information variant:

```bash
python care_myocardium/scripts/convert_cine_myops_to_nnunet.py \
  --data-root /path/to/CAREdatasets \
  --dataset-root care_myocardium/DATASET \
  --mode all_frames \
  --dataset-id 602 \
  --dataset-name CARE_CineMyoPS_AllFrames

CARE_DATASET_ID=602 bash care_myocardium/scripts/plan_preprocess.sh
GPU=0 CARE_DATASET_ID=602 bash care_myocardium/scripts/train_nnunet.sh 0
```

Before submission, map predictions back to source label values:

```bash
python care_myocardium/scripts/restore_cine_myops_labels.py \
  --input-dir care_myocardium/DATASET/predictions/cine_myops_test \
  --output-dir care_myocardium/DATASET/predictions/cine_myops_test_source_labels
```
