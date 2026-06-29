# CMR-MULTI Task 1 ? Cine MRI Segmentation & LVEF

MICCAI 2026 / 1st Workshop on Medical World Models ? CMR-MULTI Challenge, Task 1.
This folder adapts the OTBD nnU-Net + MAE-pretrained workflow to the Cine MRI
multi-view segmentation task (SAX / 2CH / 4CH) plus LVEF quantification.

- Challenge: https://mwm2026.github.io/cmr-multi
- Dataset: https://huggingface.co/datasets/TaipingQu/CMR-MULTI
- Baseline: https://github.com/qutaiping/CMR_multi_baseline

## Task

Segment cardiac structures from 3D+t Cine (SSFP) MRI across three views, then
quantify Left Ventricular Ejection Fraction (LVEF) from SAX.

| View | Labels (already contiguous, no remap needed) |
| --- | --- |
| SAX | 0 bg, 1 LV_Myo, 2 LV_Cavity, 3 RV_Cavity |
| 2CH | 0 bg, 1 LV_Cavity, 2 LV_Myo |
| 4CH | 0 bg, 1 LV_Cavity, 2 LV_Myo, 3 RV_Cavity, 4 RA, 5 LA |

Each view has a different label space, so each view is converted to its own
nnU-Net dataset:

| View | Dataset ID | Dataset name |
| --- | --- | --- |
| SAX | 501 | Dataset501_CMRMULTI_CINE_SAX |
| 2CH | 502 | Dataset502_CMRMULTI_CINE_2CH |
| 4CH | 503 | Dataset503_CMRMULTI_CINE_4CH |

## Data counts (verified)

| View | TR (labeled) | VAL (labeled, held-out) | TST (no label) |
| --- | --- | --- | --- |
| SAX | 105 | 15 | 30 |
| 2CH | 105 | 15 | 30 |
| 4CH | 105 | 15 | 30 |

## Main pipeline

The primary submission route is nnU-Net v2 initialized from the third-party
ResEncL-OpenMind-MAE self-supervised checkpoint, fine-tuned on Cine data, then
refined with class-aware MR postprocessing:

1. **Data conversion** ? `convert_cine_to_nnunet.py` turns CINE_MULTI into three
   nnU-Net raw datasets (SAX/2CH/4CH).
2. **MAE-adapted preprocessing** ? `preprocess_cine_mae_pretrained.sh` aligns each
   dataset to the MAE checkpoint's architecture plan via the TaWald nnSSL branch.
3. **MAE fine-tuning** ? `train_cine_mae_pretrained.sh` fine-tunes the ResEncL
   encoder + decoder from the pretrained weights for segmentation.
4. **Prediction** ? `predict.sh` (MAE=1) exports masks for the val/test split.
5. **Class-aware MR postprocessing** ? `postprocess.sh` runs per-class
   connected-component cleanup (keep top-k, fill holes, reject components far
   from the heart body) tuned per view via `configs/cine_postprocess_rules.json`.
6. **Functional metric** ? `calculate_lvef.py` derives LVEF from the postprocessed
   SAX masks.

The plain nnU-Net baseline (steps 1, 2 base planning, `train_nnunet.sh`) is kept
as an ablation comparison in `nnUNet_result`, while MAE fine-tuning writes to the
separate `nnUNet_result_mae`.
## Layout produced

```
cmr_multi_task1/DATASET/
  nnUNet_raw/
    Dataset501_CMRMULTI_CINE_SAX/
      imagesTr/  labelsTr/  imagesTs/  val_labels/  dataset.json
    Dataset502_CMRMULTI_CINE_2CH/  ...
    Dataset503_CMRMULTI_CINE_4CH/  ...
  nnUNet_preprocessed/
  nnUNet_result/           # plain nnU-Net checkpoints
  nnUNet_result_mae/       # MAE fine-tuned checkpoints (separate)
  test_images/{sax,2ch,4ch}/  # Codabench submission inputs
  predictions/             # predict.sh outputs
```

- `imagesTr/labelsTr` ? `*_TR` (train, 105).
- `imagesTs` ? `*_VAL` images (held-out val for prediction); matching GT kept in
  `val_labels/` for `evaluate_val.py`.
- `test_images/<view>/` ? `*_TST` images (no labels, for final submission).

## One-time external dependencies (already prepared locally)

```
third_party/TaWald_nnUNet_nnssl/                 # TaWald/nnUNet nnssl_finetuning_inclusion
pretrained/MAE/ResEncL-OpenMind-MAE/
  checkpoint_final.pth   (468 MB, MIC-DKFZ/ResEncL-OpenMind-MAE)
  adaptation_plan.json
  config.json
```

## Server environment (configure later)

Two conda envs are recommended:

1. `cmr_multi` ? stock nnU-Net v2 for the plain baseline + conversion/eval utils.
2. `cmr_multi_mae` ? the TaWald nnSSL branch (`third_party/TaWald_nnUNet_nnssl`)
   installed editable, providing `nnUNetv2_preprocess_like_nnssl` and
   `nnUNetv2_train_pretrained`. `env_mae.sh` points `MAE_CONDA_PREFIX` here and
   prepends its `bin` to PATH.

Both need `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` (set by
`env.sh`).

## Runbook

Run from the OTBD project root (`$PROJECT_ROOT`).

### 1. Convert data to nnU-Net format

```bash
python cmr_multi_task1/scripts/convert_cine_to_nnunet.py \
  --data-root /path/to/CINE_MULTI \
  --dataset-root cmr_multi_task1/DATASET
# dry-run first:
python cmr_multi_task1/scripts/convert_cine_to_nnunet.py \
  --data-root /path/to/CINE_MULTI --dry-run
```

### 2. Plan & preprocess (per view)

```bash
bash cmr_multi_task1/scripts/plan_preprocess.sh sax
bash cmr_multi_task1/scripts/plan_preprocess.sh 2ch
bash cmr_multi_task1/scripts/plan_preprocess.sh 4ch
```

### 3a. Plain nnU-Net baseline (per view / fold)

```bash
GPU=0 bash cmr_multi_task1/scripts/train_nnunet.sh sax 0
GPU=0 bash cmr_multi_task1/scripts/train_nnunet.sh 2ch 0
GPU=0 bash cmr_multi_task1/scripts/train_nnunet.sh 4ch 0
# full 5-fold:
GPU=0 bash cmr_multi_task1/scripts/train_nnunet.sh sax all
```

### 3b. MAE-pretrained fine-tune (per view / fold)

```bash
bash cmr_multi_task1/scripts/preprocess_cine_mae_pretrained.sh sax
GPU=0 FOLD=0 bash cmr_multi_task1/scripts/train_cine_mae_pretrained.sh sax 0
# 5-fold:
GPU=0 bash cmr_multi_task1/scripts/train_cine_mae_pretrained.sh sax all
```

### 4. Predict

```bash
# held-out validation
FOLDS="0" bash cmr_multi_task1/scripts/predict.sh sax val
FOLDS="0 1 2 3 4" bash cmr_multi_task1/scripts/predict.sh sax val
# MAE checkpoints:
MAE=1 FOLDS="0" bash cmr_multi_task1/scripts/predict.sh sax val
# final Codabench test
FOLDS="0 1 2 3 4" bash cmr_multi_task1/scripts/predict.sh sax test
```

### 4b. Class-aware MR postprocessing

```bash
# Clean the predicted masks per view (val or test)
bash cmr_multi_task1/scripts/postprocess.sh sax \
  cmr_multi_task1/DATASET/predictions/sax_test_mae \
  cmr_multi_task1/DATASET/predictions/sax_test_mae_pp
bash cmr_multi_task1/scripts/postprocess.sh 4ch \
  cmr_multi_task1/DATASET/predictions/4ch_val_mae \
  cmr_multi_task1/DATASET/predictions/4ch_val_mae_pp
```

Tune per-class rules in `cmr_multi_task1/configs/cine_postprocess_rules.json`
(keep_top_k, fill_holes, max_distance_to_heart_mm, min_component_size).
### 5. Evaluate validation (Dice / HD95)

```bash
python cmr_multi_task1/scripts/evaluate_val.py \
  --pred-dir cmr_multi_task1/DATASET/predictions/sax_val_plain \
  --gt-dir   cmr_multi_task1/DATASET/nnUNet_raw/Dataset501_CMRMULTI_CINE_SAX/val_labels
```

### 6. LVEF (SAX only)

```bash
python cmr_multi_task1/scripts/calculate_lvef.py \
  --pred-dir cmr_multi_task1/DATASET/predictions/sax_test_plain \
  --slice-info /path/to/CINE_MULTI/sax_slice_info_test.json \
  --output cmr_multi_task1/DATASET/predictions/sax_test_plain/lv_ef_results.json
```

## Notes

- Labels are already small contiguous ints; conversion copies + verifies only.
- The 3rd axis of a Cine volume stacks spatial slices x temporal phases; nnU-Net
  treats it as a 3D volume (same as the official baseline).
- MAE fine-tuning writes to `nnUNet_result_mae` so it never overwrites the plain
  baseline checkpoints in `nnUNet_result`.
- `calculate_lvef.py` mirrors the official baseline algorithm exactly so the
  LVEF metric is comparable.


