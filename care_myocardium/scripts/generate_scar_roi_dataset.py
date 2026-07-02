#!/usr/bin/env python3
"""Build staged CARE myocardium or scar ROI nnU-Net datasets.

The first-stage model segments the full ED frame. This script crops a focused
ROI around the chosen predicted target proposal, adds simple first-stage priors
as input channels, and trains a binary ROI refiner on the cropped region.

Channels:
    0000: ED image crop
    0001: first-stage target prior or proposal
    0002: context prior. For scar Stage 3, this can be a refined Stage 2
          myocardium prior; otherwise it is the first-stage foreground.

Labels:
    0 background
    1 selected target, either LV myocardium including scar or LV scar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "DATASET"
DEFAULT_ED_DATASET = "Dataset601_CARE_CineMyoPS_ED"

TARGET_CONFIGS = {
    "scar": {
        "default_output_id": 603,
        "default_output_name": "CARE_CineMyoPS_ScarROI_ED",
        "proposal_label_ids": (3,),
        "target_label_ids": (3,),
        "target_channel_name": "stage1_scar_prior",
        "context_channel_name": "stage1_heart_prior",
        "output_label_name": "LV_myocardial_scar",
        "proposal_source": "stage1_scar",
        "gt_source": "gt_scar_fallback",
        "voxel_key": "scar_voxels",
        "prior_voxel_key": "stage1_scar_voxels_in_crop",
        "manifest_name": "scar_roi_manifest.json",
    },
    "myo": {
        "default_output_id": 604,
        "default_output_name": "CARE_CineMyoPS_MyoROI_ED",
        "proposal_label_ids": (1, 3),
        "target_label_ids": (1, 3),
        "target_channel_name": "stage1_myo_prior",
        "context_channel_name": "stage1_heart_prior",
        "output_label_name": "LV_myocardium_including_scar",
        "proposal_source": "stage1_myo",
        "gt_source": "gt_myo_fallback",
        "voxel_key": "myo_voxels",
        "prior_voxel_key": "stage1_myo_voxels_in_crop",
        "manifest_name": "myo_roi_manifest.json",
    },
}


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {name}")


def case_id_from_image(path: Path) -> str:
    stem = strip_nii_suffix(path.name)
    return stem[:-5] if stem.endswith("_0000") else stem


def load_optional_pred(
    pred_dir: Path | None,
    case_id: str,
    shape: tuple[int, ...],
    *,
    required: bool = False,
    label: str = "prediction",
) -> np.ndarray:
    if pred_dir is None:
        if required:
            raise FileNotFoundError(f"Missing required {label} directory for {case_id}")
        return np.zeros(shape, dtype=np.int16)
    pred_path = pred_dir / f"{case_id}.nii.gz"
    if not pred_path.is_file():
        pred_path = pred_dir / f"{case_id}_pred.nii.gz"
    if not pred_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing {label} for {case_id} in {pred_dir}")
        return np.zeros(shape, dtype=np.int16)
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj).astype(np.int16)
    if pred.shape != shape:
        raise ValueError(f"{label} {pred_path}: shape {pred.shape} does not match image shape {shape}")
    return pred


def bbox_from_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.min(axis=0), coords.max(axis=0) + 1


def mask_from_labels(data: np.ndarray, label_ids: tuple[int, ...]) -> np.ndarray:
    if len(label_ids) == 1:
        return data == label_ids[0]
    return np.isin(data, np.asarray(label_ids, dtype=data.dtype))


def dilate_xy(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=False)
        expanded = np.zeros_like(out, dtype=bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                expanded |= padded[1 + dx:1 + dx + out.shape[0], 1 + dy:1 + dy + out.shape[1], :]
        out = expanded
    return out


def transform_target_prior(mask: np.ndarray, mode: str, dilation_xy: int) -> np.ndarray:
    if mode == "binary":
        return mask.astype(bool)
    if mode == "dilate_xy":
        return dilate_xy(mask, dilation_xy)
    if mode == "none":
        return np.zeros_like(mask, dtype=bool)
    raise ValueError(f"Unsupported target prior mode: {mode}")


def target_channel_name(config: dict, mode: str) -> str:
    if mode == "binary":
        return config["target_channel_name"]
    if mode == "dilate_xy":
        return config["target_channel_name"].replace("_prior", "_dilated_prior")
    if mode == "none":
        return f"empty_{config['target_channel_name']}"
    raise ValueError(f"Unsupported target prior mode: {mode}")


def centered_bounds(
    start: int,
    end: int,
    limit: int,
    margin: int,
    min_size: int,
) -> tuple[int, int]:
    start = max(0, int(start) - margin)
    end = min(limit, int(end) + margin)
    size = end - start
    if size < min_size:
        center = (start + end) // 2
        start = center - min_size // 2
        end = start + min_size
        if start < 0:
            end -= start
            start = 0
        if end > limit:
            start -= end - limit
            end = limit
        start = max(0, start)
    return int(start), int(end)


def choose_crop(
    pred: np.ndarray,
    label: np.ndarray | None,
    proposal_label_ids: tuple[int, ...],
    target_label_ids: tuple[int, ...],
    proposal_source: str,
    gt_source: str,
    margin_xy: int,
    min_xy: int,
    full_z: bool,
    primary_mask: np.ndarray | None = None,
    primary_source: str | None = None,
) -> tuple[slice, slice, slice, str]:
    if primary_mask is not None:
        proposal = bbox_from_mask(primary_mask)
        source = primary_source or "primary_mask"
    else:
        proposal = bbox_from_mask(mask_from_labels(pred, proposal_label_ids))
        source = proposal_source
    if proposal is None and label is not None:
        proposal = bbox_from_mask(mask_from_labels(label, target_label_ids))
        source = gt_source
    if proposal is None:
        proposal = bbox_from_mask(pred > 0)
        source = "stage1_foreground_fallback"
    if proposal is None and label is not None:
        proposal = bbox_from_mask(label > 0)
        source = "gt_foreground_fallback"
    if proposal is None:
        shape = pred.shape
        start = np.array([shape[0] // 2, shape[1] // 2, 0], dtype=int)
        end = start + 1
        source = "center_fallback"
    else:
        start, end = proposal

    x0, x1 = centered_bounds(start[0], end[0], pred.shape[0], margin_xy, min_xy)
    y0, y1 = centered_bounds(start[1], end[1], pred.shape[1], margin_xy, min_xy)
    if full_z:
        z0, z1 = 0, pred.shape[2]
    else:
        z0, z1 = centered_bounds(start[2], end[2], pred.shape[2], 1, 1)
    return slice(x0, x1), slice(y0, y1), slice(z0, z1), source


def crop_affine(affine: np.ndarray, slices: tuple[slice, slice, slice]) -> np.ndarray:
    out = affine.copy()
    offset = np.array([s.start for s in slices], dtype=float)
    out[:3, 3] = affine[:3, 3] + affine[:3, :3] @ offset
    return out


def save_like(
    data: np.ndarray,
    ref_img: nib.Nifti1Image,
    slices: tuple[slice, slice, slice],
    out_path: Path,
    dtype: np.dtype | type,
) -> None:
    header = ref_img.header.copy()
    out = nib.Nifti1Image(data.astype(dtype), crop_affine(ref_img.affine, slices), header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(ref_img.header.get_zooms()[:3])
    out.header.set_data_dtype(dtype)
    nib.save(out, str(out_path))


def write_dataset_json(
    dataset_dir: Path,
    num_training: int,
    target_channel_name_value: str,
    config: dict,
    context_channel_name: str,
) -> None:
    payload = {
        "channel_names": {
            "0": "cine_ed_roi",
            "1": target_channel_name_value,
            "2": context_channel_name,
        },
        "labels": {
            "background": 0,
            config["output_label_name"]: 1,
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create CARE myocardium or scar ROI refiner dataset.")
    p.add_argument("--target", choices=sorted(TARGET_CONFIGS), default="scar")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--ed-dataset-name", default=DEFAULT_ED_DATASET)
    p.add_argument("--stage1-pred-dir", type=Path, default=None)
    p.add_argument(
        "--stage2-myo-pred-dir",
        type=Path,
        default=None,
        help="Optional full-image Stage 2 myocardium predictions for scar Stage 3.",
    )
    p.add_argument("--output-dataset-id", type=int, default=None)
    p.add_argument("--output-dataset-name", default=None)
    p.add_argument("--margin-xy", type=int, default=48)
    p.add_argument("--min-xy", type=int, default=128)
    p.add_argument(
        "--crop-source",
        choices=("stage1_target", "stage2_myo", "gt_target"),
        default="stage1_target",
        help="Mask used to localize ROI crop before fallbacks.",
    )
    p.add_argument(
        "--target-prior-mode",
        choices=("binary", "dilate_xy", "none"),
        default="binary",
        help="How to encode the stage-1 target prior channel.",
    )
    p.add_argument("--prior-dilation-xy", type=int, default=4)
    p.add_argument("--crop-full-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = TARGET_CONFIGS[args.target]
    output_dataset_id = args.output_dataset_id or config["default_output_id"]
    output_dataset_name = args.output_dataset_name or config["default_output_name"]
    use_stage2_myo_context = args.target == "scar" and args.stage2_myo_pred_dir is not None
    context_channel_name = "stage2_myo_prior" if use_stage2_myo_context else config["context_channel_name"]
    target_channel_name_value = target_channel_name(config, args.target_prior_mode)

    ed_dir = args.dataset_root / "nnUNet_raw" / args.ed_dataset_name
    images_tr = ed_dir / "imagesTr"
    labels_tr = ed_dir / "labelsTr"
    if not images_tr.is_dir() or not labels_tr.is_dir():
        raise FileNotFoundError(f"Expected imagesTr/labelsTr under {ed_dir}")

    dataset_name = output_dataset_name
    if not dataset_name.startswith(f"Dataset{output_dataset_id:03d}_"):
        dataset_name = f"Dataset{output_dataset_id:03d}_{dataset_name}"
    out_dir = args.dataset_root / "nnUNet_raw" / dataset_name
    out_images = out_dir / "imagesTr"
    out_labels = out_dir / "labelsTr"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{out_dir} exists; pass --overwrite to replace files")

    image_files = sorted(images_tr.glob("*_0000.nii.gz"))
    if not image_files:
        raise FileNotFoundError(f"No *_0000.nii.gz files in {images_tr}")

    manifest = []
    print(f"target       = {args.target}")
    print(f"ed_dir       = {ed_dir}")
    print(f"stage1_pred  = {args.stage1_pred_dir}")
    print(f"stage2_myo   = {args.stage2_myo_pred_dir}")
    print(f"crop_source  = {args.crop_source}")
    print(f"prior_mode   = {args.target_prior_mode}")
    print(f"out_dir      = {out_dir}")
    print(f"cases        = {len(image_files)}")
    if args.dry_run:
        return

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    for img_path in image_files:
        case_id = case_id_from_image(img_path)
        label_path = labels_tr / f"{case_id}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")

        img = nib.load(str(img_path))
        label_img = nib.load(str(label_path))
        image = np.asanyarray(img.dataobj)
        label = np.asanyarray(label_img.dataobj).astype(np.int16)
        if image.shape != label.shape:
            raise ValueError(f"{case_id}: image shape {image.shape} != label shape {label.shape}")
        pred = load_optional_pred(args.stage1_pred_dir, case_id, label.shape, label="stage1 prediction")
        stage2_myo = load_optional_pred(
            args.stage2_myo_pred_dir,
            case_id,
            label.shape,
            required=use_stage2_myo_context or args.crop_source == "stage2_myo",
            label="stage2 myocardium prediction",
        )

        primary_mask = None
        primary_source = None
        if args.crop_source == "stage2_myo":
            primary_mask = stage2_myo > 0
            primary_source = "stage2_myo"
        elif args.crop_source == "gt_target":
            primary_mask = mask_from_labels(label, config["target_label_ids"])
            primary_source = config["gt_source"]

        sx, sy, sz, source = choose_crop(
            pred=pred,
            label=label,
            proposal_label_ids=config["proposal_label_ids"],
            target_label_ids=config["target_label_ids"],
            proposal_source=config["proposal_source"],
            gt_source=config["gt_source"],
            margin_xy=args.margin_xy,
            min_xy=args.min_xy,
            full_z=args.crop_full_z,
            primary_mask=primary_mask,
            primary_source=primary_source,
        )
        slices = (sx, sy, sz)
        ed_crop = image[slices]
        binary_target_prior = mask_from_labels(pred[slices], config["proposal_label_ids"])
        target_prior = transform_target_prior(
            binary_target_prior,
            args.target_prior_mode,
            args.prior_dilation_xy,
        ).astype(np.float32)
        context_prior = (
            (stage2_myo[slices] > 0).astype(np.float32)
            if use_stage2_myo_context
            else (pred[slices] > 0).astype(np.float32)
        )
        target_label = mask_from_labels(label[slices], config["target_label_ids"]).astype(np.int16)

        save_like(ed_crop, img, slices, out_images / f"{case_id}_0000.nii.gz", img.get_data_dtype())
        save_like(target_prior, img, slices, out_images / f"{case_id}_0001.nii.gz", np.float32)
        save_like(context_prior, img, slices, out_images / f"{case_id}_0002.nii.gz", np.float32)
        save_like(target_label, label_img, slices, out_labels / f"{case_id}.nii.gz", np.int16)

        crop_box = [[s.start, s.stop] for s in slices]
        manifest.append({
            "target": args.target,
            "case_id": case_id,
            "source_image": str(img_path),
            "source_label": str(label_path),
            "stage1_prediction": str((args.stage1_pred_dir / f"{case_id}.nii.gz") if args.stage1_pred_dir else ""),
            "stage2_myo_prediction": str((args.stage2_myo_pred_dir / f"{case_id}.nii.gz") if use_stage2_myo_context else ""),
            "crop_source_requested": args.crop_source,
            "proposal_source": source,
            "crop_box_xyz": crop_box,
            "source_shape": list(label.shape),
            "crop_shape": list(target_label.shape),
            config["voxel_key"]: int(target_label.sum()),
            config["prior_voxel_key"]: int(target_prior.sum()),
            "target_prior_mode": args.target_prior_mode,
            "prior_dilation_xy": args.prior_dilation_xy,
            "context_prior_channel": context_channel_name,
            "stage2_myo_voxels_in_crop": int(context_prior.sum()) if use_stage2_myo_context else 0,
        })
        print(f"{case_id}: {source} crop={crop_box} {config['voxel_key']}={int(target_label.sum())}")

    write_dataset_json(out_dir, len(manifest), target_channel_name_value, config, context_channel_name)
    manifest_payload = {
            "target": args.target,
            "ed_dataset": str(ed_dir),
            "stage1_pred_dir": str(args.stage1_pred_dir) if args.stage1_pred_dir else None,
            "stage2_myo_pred_dir": str(args.stage2_myo_pred_dir) if use_stage2_myo_context else None,
            "crop_source": args.crop_source,
            "target_prior_mode": args.target_prior_mode,
            "prior_dilation_xy": args.prior_dilation_xy,
            "context_prior_channel": context_channel_name,
            "margin_xy": args.margin_xy,
            "min_xy": args.min_xy,
            "crop_full_z": args.crop_full_z,
            "cases": manifest,
    }
    with (out_dir / config["manifest_name"]).open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    with (out_dir / "roi_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    print(f"Created {out_dir}")
    print(f"cases={len(manifest)}")


if __name__ == "__main__":
    main()
