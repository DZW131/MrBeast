#!/usr/bin/env python3
"""Restore cropped ROI predictions back into full Dataset601 image space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {name}")


def load_prediction(pred_dir: Path, case_id: str) -> np.ndarray:
    candidates = [
        pred_dir / f"{case_id}.nii.gz",
        pred_dir / f"{case_id}_pred.nii.gz",
    ]
    for path in candidates:
        if path.is_file():
            return np.asanyarray(nib.load(str(path)).dataobj).astype(np.int16)
    raise FileNotFoundError(f"Missing ROI prediction for {case_id} in {pred_dir}")


def save_full_prediction(data: np.ndarray, reference_img: nib.Nifti1Image, out_path: Path) -> None:
    header = reference_img.header.copy()
    out = nib.Nifti1Image(data.astype(np.int16), reference_img.affine, header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(reference_img.header.get_zooms()[:3])
    out.header.set_data_dtype(np.int16)
    nib.save(out, str(out_path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paste cropped ROI predictions back into full image space.")
    p.add_argument("--roi-dataset-dir", type=Path, required=True)
    p.add_argument("--crop-pred-dir", type=Path, required=True)
    p.add_argument("--reference-dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest-name", default="roi_manifest.json")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.roi_dataset_dir / args.manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ROI manifest: {manifest_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists; pass --overwrite to replace files")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError(f"No cases found in {manifest_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_images = args.reference_dataset_dir / "imagesTr"
    restored = []
    for case in cases:
        case_id = case["case_id"]
        crop_box = case["crop_box_xyz"]
        source_shape = tuple(int(v) for v in case["source_shape"])
        ref_path = ref_images / f"{case_id}_0000.nii.gz"
        if not ref_path.is_file():
            raise FileNotFoundError(f"Missing reference image for {case_id}: {ref_path}")

        crop_pred = load_prediction(args.crop_pred_dir, case_id)
        slices = tuple(slice(int(start), int(stop)) for start, stop in crop_box)
        expected_shape = tuple(s.stop - s.start for s in slices)
        if crop_pred.shape != expected_shape:
            raise ValueError(f"{case_id}: prediction shape {crop_pred.shape} != crop shape {expected_shape}")

        full = np.zeros(source_shape, dtype=np.int16)
        full[slices] = crop_pred.astype(np.int16)
        save_full_prediction(full, nib.load(str(ref_path)), args.output_dir / f"{case_id}.nii.gz")
        restored.append({"case_id": case_id, "crop_shape": list(crop_pred.shape), "full_voxels": int(full.sum())})
        print(f"{case_id}: pasted crop={crop_box} voxels={int(full.sum())}")

    with (args.output_dir / "restore_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "roi_dataset_dir": str(args.roi_dataset_dir),
                "crop_pred_dir": str(args.crop_pred_dir),
                "reference_dataset_dir": str(args.reference_dataset_dir),
                "cases": restored,
            },
            f,
            indent=2,
        )
    print(f"restored={len(restored)} output={args.output_dir}")


if __name__ == "__main__":
    main()
