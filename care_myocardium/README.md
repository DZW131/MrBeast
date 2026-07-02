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

MWM/CMR-MULTI SAX-initialized CARE ED fine-tune:

The default CARE ED nnU-Net architecture is not identical to the MWM SAX
baseline architecture, so direct `-pretrained_weights` loading would fail on
shape checks. This run writes a CARE plan named `nnUNetPlans_MWMSAXArch` that
keeps the CARE preprocessing, labels, spacing and normalization, but uses the
MWM SAX network architecture and a compatible CARE patch size. Results are
therefore stored separately from the CARE ED scratch baseline.

```bash
GPU=0 bash care_myocardium/scripts/train_mwm_sax_init_ed.sh 0
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

`Dataset602` stores all 30 cine frames as channels, so each 3D sample is much
larger than the ED-only baseline. The training script defaults this dataset to
`nnUNet_n_proc_DA=2` and `nnUNet_compile=f` unless you override them. A short
server benchmark found this stable on the CARE all-frame `.b2nd` cases while
being substantially faster than single-threaded augmentation.

SAM2-inspired cine memory variant:

This keeps nnU-Net as the segmentation backbone but prepends a lightweight
temporal memory reader. The ED frame is the query, the remaining cine frames are
read as a memory bank with voxel-wise temporal attention, and the network then
segments the enhanced ED volume. This is intended for `Dataset602` because it
needs all 30 frames as input channels.

```bash
# after creating and preprocessing Dataset602_CARE_CineMyoPS_AllFrames
GPU=0 bash care_myocardium/scripts/train_cine_memory_nnunet.sh 0
```

Useful knobs:

```bash
CINE_MEMORY_EMBED_DIM=12 \
CINE_MEMORY_QUERY_FRAME_INDEX=0 \
GPU=0 bash care_myocardium/scripts/train_cine_memory_nnunet.sh 0
```

For prediction with this trainer, export the external trainer path before
calling `nnUNetv2_predict`:

```bash
export nnUNet_extTrainer="$(pwd)/care_myocardium/nnunet_ext"
```

Before submission, map predictions back to source label values:

```bash
python care_myocardium/scripts/restore_cine_myops_labels.py \
  --input-dir care_myocardium/DATASET/predictions/cine_myops_test \
  --output-dir care_myocardium/DATASET/predictions/cine_myops_test_source_labels
```

## Validation Leaderboard Packaging

The CARE validation leaderboard expects a zip named like
`CARE-Myocardium-TeamName.zip`. For the CineMyoPS-only baseline, package only
the `CineMyoPS` folder:

```text
CARE-Myocardium-Monster.zip
└── CineMyoPS
    └── Anonymous Center
        └── Case****
            └── Case****_pred.nii.gz
```

Once the official anonymous CineMyoPS validation images are available on the
server, run the ED baseline submission pipeline:

```bash
CARE_CINEMYOPS_INFER_ROOT=/path/to/anonymous/CineMyoPS_validation \
TEAM_NAME=Monster \
GPU=0 \
bash care_myocardium/scripts/predict_ed_baseline_submission.sh
```

The script extracts frame 0 into nnU-Net `imagesTs`, runs the completed
`Dataset601` ED baseline checkpoint, restores nnU-Net labels `0/1/2/3` to CARE
labels `0/200/500/2221`, and writes:

```text
care_myocardium/DATASET/submissions/CARE-Myocardium-Monster.zip
```

Current local/server data contains only `Myo_train`; the anonymous validation
images must be downloaded separately before a real leaderboard upload package
can be generated.
