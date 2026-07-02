#!/usr/bin/env python3
"""Build staged CARE myocardium or scar ROI nnU-Net datasets.

The first-stage model segments the full ED frame. This script crops a focused
ROI around the chosen predicted target proposal, adds simple first-stage priors
as input channels, and trains a binary ROI refiner on the cropped region.

Channels:
    0000: ED image crop
    0001: first-stage target prior, binary
    0002: first-stage heart/context prior, binary foreground

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


def load_optional_pred(pred_dir: Path | None, case_id: str, shape: tuple[int, ...]) -> np.ndarray:
    if pred_dir is None:
        return np.zeros(shape, dtype=np.int16)
    pred_path = pred_dir / f"{case_id}.nii.gz"
    if not pred_path.is_file():
        pred_path = pred_dir / f"{case_id}_pred.nii.gz"
    if not pred_path.is_file():
        return np.zeros(shape, dtype=np.int16)
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj).astype(np.int16)
    if pred.shape != shape:
        raise ValueError(f"{pred_path}: shape {pred.shape} does not match image shape {shape}")
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
) -> tuple[slice, slice, slice, str]:
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


def write_dataset_json(dataset_dir: Path, num_training: int, config: dict) -> None:
    payload = {
        "channel_names": {
            "0": "cine_ed_roi",
            "1": config["target_channel_name"],
            "2": config["context_channel_name"],
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
    p.add_argument("--output-dataset-id", type=int, default=None)
    p.add_argument("--output-dataset-name", default=None)
    p.add_argument("--margin-xy", type=int, default=48)
    p.add_argument("--min-xy", type=int, default=128)
    p.add_argument("--crop-full-z", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = TARGET_CONFIGS[args.target]
    output_dataset_id = args.output_dataset_id or config["default_output_id"]
    output_dataset_name = args.output_dataset_name or config["default_output_name"]

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
        pred = load_optional_pred(args.stage1_pred_dir, case_id, label.shape)

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
        )
        slices = (sx, sy, sz)
        ed_crop = image[slices]
        target_prior = mask_from_labels(pred[slices], config["proposal_label_ids"]).astype(np.float32)
        heart_prior = (pred[slices] > 0).astype(np.float32)
        target_label = mask_from_labels(label[slices], config["target_label_ids"]).astype(np.int16)

        save_like(ed_crop, img, slices, out_images / f"{case_id}_0000.nii.gz", img.get_data_dtype())
        save_like(target_prior, img, slices, out_images / f"{case_id}_0001.nii.gz", np.float32)
        save_like(heart_prior, img, slices, out_images / f"{case_id}_0002.nii.gz", np.float32)
        save_like(target_label, label_img, slices, out_labels / f"{case_id}.nii.gz", np.int16)

        crop_box = [[s.start, s.stop] for s in slices]
        manifest.append({
            "target": args.target,
            "case_id": case_id,
            "source_image": str(img_path),
            "source_label": str(label_path),
            "stage1_prediction": str((args.stage1_pred_dir / f"{case_id}.nii.gz") if args.stage1_pred_dir else ""),
            "proposal_source": source,
            "crop_box_xyz": crop_box,
            "source_shape": list(label.shape),
            "crop_shape": list(target_label.shape),
            config["voxel_key"]: int(target_label.sum()),
            config["prior_voxel_key"]: int(target_prior.sum()),
        })
        print(f"{case_id}: {source} crop={crop_box} {config['voxel_key']}={int(target_label.sum())}")

    write_dataset_json(out_dir, len(manifest), config)
    manifest_payload = {
            "target": args.target,
            "ed_dataset": str(ed_dir),
            "stage1_pred_dir": str(args.stage1_pred_dir) if args.stage1_pred_dir else None,
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
