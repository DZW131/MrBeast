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

Motion-texture fusion variant:

This is the nnU-Net adaptation of the contrast-free cine scar segmentation idea
from arXiv:2501.05241. It keeps nnU-Net as the segmentation backbone, but makes
the input closer to the paper: all cine frames provide temporal texture, and
extra channels encode ED-referenced motion cues. By default Dataset608 adds five
temporal difference/statistics channels plus four aggregated Farneback optical
flow channels:

| Channel family | Meaning |
| --- | --- |
| `cine_t00..cine_t29` | Full cine texture sequence |
| `temporal_std` | Per-voxel temporal intensity variation |
| `max_abs_diff_from_ed` | Largest absolute difference from the ED frame |
| `mean_abs_diff_from_ed` | Average absolute difference from the ED frame |
| `signed_diff_at_max_abs_diff` | Signed intensity change at the strongest motion frame |
| `normalized_max_diff_frame` | Frame index of the strongest ED-referenced change |
| `farneback_*_to_ed` | 2D slice-wise optical-flow summary from cine frames to ED |

```bash
# Create and preprocess Dataset608_CARE_CineMyoPS_MotionTexture.
bash care_myocardium/scripts/prepare_motion_texture_dataset.sh

# Train the motion-texture nnU-Net baseline.
GPU=0 bash care_myocardium/scripts/train_motion_texture_nnunet.sh 0

# Quick 400-epoch OpenCV/Farneback motion-texture ablation.
CARE_DATASET_ID=609 GPU=0 bash care_myocardium/scripts/train_motion_texture_400_nnunet.sh 0

# Paper-length 1000-epoch OpenCV/Farneback motion-texture run.
CARE_DATASET_ID=609 GPU=0 bash care_myocardium/scripts/train_motion_texture_1000_nnunet.sh 0

# 300-epoch OpenCV/Farneback motion-texture run with first/last ED cycle consistency.
CARE_DATASET_ID=609 GPU=0 bash care_myocardium/scripts/train_motion_texture_ed_cycle_nnunet.sh 0

# Strict learned-motion route from the MTI-MyoScarSeg paper:
# 1) train Motion-Net with ED-to-frame registration for 1000 epochs.
GPU=0,1,2,3 MOTION_NET_GPUS=4 bash care_myocardium/scripts/train_motion_net.sh

# 2) export cine frames + per-frame learned displacement fields as Dataset610.
bash care_myocardium/scripts/prepare_learned_motion_dataset.sh

# 3) train Seg-Net for 400 epochs on learned motion + cine texture channels.
CARE_DATASET_ID=610 GPU=0 bash care_myocardium/scripts/train_learned_motion_seg_nnunet.sh 0
```

Useful knobs:

```bash
# Dependency-free ablation: only cine frames + ED difference motion proxies.
FLOW_MODE=none bash care_myocardium/scripts/prepare_motion_texture_dataset.sh

# Faster optical-flow preprocessing: use every second cine frame for flow.
FLOW_FRAME_STRIDE=2 bash care_myocardium/scripts/prepare_motion_texture_dataset.sh
```

First/last ED cycle-consistency variant:

The official CineMyoPS labels annotate the first ED frame. The final cine frame
is also visually close to ED but has no label. This trainer keeps Dataset608 as
input and adds a closed-loop ED consistency regularizer: the normal forward view
uses frame 0 as the ED reference, while the reversed cine view uses the final
frame as the ED reference and recomputes the deterministic motion proxy
channels. The trainer then enforces bidirectional first-ED <-> last-ED soft
prediction consistency with confidence gating and extra scar weighting. The
supervised label loss on frame 0 remains unchanged.

```bash
# after Dataset608 has been created and preprocessed
GPU=0 bash care_myocardium/scripts/train_ed_cycle_nnunet.sh 0
```

Useful knobs:

```bash
ED_CYCLE_EPOCHS=300 \
ED_CYCLE_WEIGHT=0.2 \
ED_CYCLE_RAMP_EPOCHS=40 \
ED_CYCLE_CONFIDENCE=0.6 \
ED_CYCLE_SCAR_WEIGHT=2.0 \
GPU=0 bash care_myocardium/scripts/train_ed_cycle_nnunet.sh 0
```

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

Staged myocardium-to-scar ROI refinement:

This route keeps the stable ED full-image model as stage 1. Stage 2 trains a
binary myocardium refiner on local crops where myocardium means label 1 plus
scar label 3. Stage 3 then trains the binary scar refiner inside the myocardial
context. This matches the anatomy better than refining scar alone, because scar
should be constrained by the myocardium.

Stage 2 ROI samples have three channels: ED image crop, stage-1 myocardium
prior, and stage-1 foreground context prior. Stage 3-v1/Dataset605 uses ED
image crop, exact stage-1 scar prior, and Stage 2 refined myocardium prior, but
diagnostics showed it can collapse into copying the exact scar prior. Stage
3-v2/Dataset606 replaces the exact scar prior with a dilated stage-1 scar
proposal, but still crops around the stage-1 scar. The preferred Stage
3-v3/Dataset607 crops around the Stage 2 refined myocardium instead, then uses
the dilated scar proposal only as a hint. This is intended to recover cases
where the stage-1 scar proposal misses the real scar. Dataset603 remains the
scar-only ROI baseline for ablation. The ROI trainers default to 300 epochs
because these are small second/third-stage datasets.

```bash
# 1) Predict stage-1 masks on Dataset601 imagesTr with the finished ED baseline.
GPU=4 bash care_myocardium/scripts/predict_ed_train_for_scar_roi.sh

# 2) Create Dataset604_CARE_CineMyoPS_MyoROI_ED and preprocess it.
CARE_DATASET_ID=604 bash care_myocardium/scripts/prepare_myo_roi_dataset.sh

# 3) Train the 300-epoch myocardium ROI refiner.
GPU=4 MYO_ROI_EPOCHS=300 bash care_myocardium/scripts/train_myo_roi_refiner.sh 0

# 4) Predict Stage 2 myocardium ROI masks and paste them back to full image space.
GPU=4 bash care_myocardium/scripts/predict_myo_roi_train_for_scar_roi.sh

# 5) Create Dataset607_CARE_CineMyoPS_ScarMyoSearchROI_ED and preprocess it.
CARE_DATASET_ID=607 bash care_myocardium/scripts/prepare_scar_myo_search_roi_dataset.sh

# 6) Train the 300-epoch myocardium-centered scar search refiner.
GPU=4 CARE_DATASET_ID=607 SCAR_ROI_EPOCHS=300 bash care_myocardium/scripts/train_scar_myo_roi_refiner.sh 0
```

To compare shorter runs:

```bash
GPU=4 MYO_ROI_EPOCHS=200 bash care_myocardium/scripts/train_myo_roi_refiner.sh 0
GPU=4 CARE_DATASET_ID=607 SCAR_ROI_EPOCHS=200 bash care_myocardium/scripts/train_scar_myo_roi_refiner.sh 0
```

Earlier Stage 3 and scar-only ablations:

```bash
CARE_DATASET_ID=606 bash care_myocardium/scripts/prepare_scar_myo_dilated_roi_dataset.sh
GPU=4 CARE_DATASET_ID=606 SCAR_ROI_EPOCHS=300 bash care_myocardium/scripts/train_scar_myo_roi_refiner.sh 0

CARE_DATASET_ID=605 bash care_myocardium/scripts/prepare_scar_myo_roi_dataset.sh
GPU=4 CARE_DATASET_ID=605 SCAR_ROI_EPOCHS=300 bash care_myocardium/scripts/train_scar_myo_roi_refiner.sh 0

CARE_DATASET_ID=603 bash care_myocardium/scripts/prepare_scar_roi_dataset.sh
GPU=4 SCAR_ROI_EPOCHS=300 bash care_myocardium/scripts/train_scar_roi_refiner.sh 0
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
