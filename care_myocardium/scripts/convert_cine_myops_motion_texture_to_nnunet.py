#!/usr/bin/env python3
"""Create a motion-texture nnU-Net dataset for CARE CineMyoPS.

This is an nnU-Net-friendly adaptation of the motion/texture fusion idea used
for contrast-free cine scar segmentation: the full cine sequence is exported as
texture channels, and ED-referenced motion cues are exported as extra channels.
The segmentation target remains the original 3D ED mask with myocardium, LV
blood pool, and scar labels, so nnU-Net keeps the multi-class supervision.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_cine_myops_to_nnunet import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    iter_cases,
    load_json,
    remap_label,
    resolve_cine_root,
)


DIFF_MOTION_FEATURE_NAMES = [
    "temporal_std",
    "max_abs_diff_from_ed",
    "mean_abs_diff_from_ed",
    "signed_diff_at_max_abs_diff",
    "normalized_max_diff_frame",
]

FARNEBACK_FEATURE_NAMES = [
    "farneback_mean_dx_to_ed",
    "farneback_mean_dy_to_ed",
    "farneback_mean_magnitude_to_ed",
    "farneback_max_magnitude_to_ed",
]


def make_3d_image(data: np.ndarray, ref_img: nib.Nifti1Image, dtype: np.dtype | type) -> nib.Nifti1Image:
    header = ref_img.header.copy()
    zooms = header.get_zooms()[:3]
    out = nib.Nifti1Image(data.astype(dtype), ref_img.affine, header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(zooms)
    out.header.set_data_dtype(dtype)
    return out


def normalized_for_flow(slice_2d: np.ndarray) -> np.ndarray:
    arr = slice_2d.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def compute_diff_motion_features(cine: np.ndarray, frame_index: int) -> list[tuple[str, np.ndarray]]:
    cine_f = cine.astype(np.float32, copy=False)
    ed = cine_f[..., frame_index]
    diffs = cine_f - ed[..., None]
    absdiff = np.abs(diffs)
    max_idx = np.argmax(absdiff, axis=3)
    signed_at_max = np.take_along_axis(diffs, max_idx[..., None], axis=3)[..., 0]
    denom = max(cine_f.shape[3] - 1, 1)
    return [
        ("temporal_std", np.std(cine_f, axis=3)),
        ("max_abs_diff_from_ed", np.max(absdiff, axis=3)),
        ("mean_abs_diff_from_ed", np.mean(absdiff, axis=3)),
        ("signed_diff_at_max_abs_diff", signed_at_max),
        ("normalized_max_diff_frame", max_idx.astype(np.float32) / float(denom)),
    ]


def compute_farneback_features(
    cine: np.ndarray,
    frame_index: int,
    frame_stride: int,
) -> list[tuple[str, np.ndarray]]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("--flow-mode farneback_agg requires opencv-python/cv2") from exc

    cine_f = cine.astype(np.float32, copy=False)
    h, w, z, n_frames = cine_f.shape
    sum_dx = np.zeros((h, w, z), dtype=np.float32)
    sum_dy = np.zeros((h, w, z), dtype=np.float32)
    sum_mag = np.zeros((h, w, z), dtype=np.float32)
    max_mag = np.zeros((h, w, z), dtype=np.float32)
    count = 0

    for t in range(0, n_frames, frame_stride):
        if t == frame_index:
            continue
        for zi in range(z):
            ref = normalized_for_flow(cine_f[:, :, zi, frame_index])
            mov = normalized_for_flow(cine_f[:, :, zi, t])
            flow = cv2.calcOpticalFlowFarneback(
                ref,
                mov,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            dx = flow[..., 0].astype(np.float32)
            dy = flow[..., 1].astype(np.float32)
            mag = np.sqrt(dx * dx + dy * dy).astype(np.float32)
            sum_dx[:, :, zi] += dx
            sum_dy[:, :, zi] += dy
            sum_mag[:, :, zi] += mag
            max_mag[:, :, zi] = np.maximum(max_mag[:, :, zi], mag)
        count += 1

    denom = float(max(count, 1))
    return [
        ("farneback_mean_dx_to_ed", sum_dx / denom),
        ("farneback_mean_dy_to_ed", sum_dy / denom),
        ("farneback_mean_magnitude_to_ed", sum_mag / denom),
        ("farneback_max_magnitude_to_ed", max_mag),
    ]


def write_dataset_json(
    dataset_dir: Path,
    cfg: dict,
    num_training: int,
    channel_names: list[str],
) -> None:
    labels = {name: int(value) for name, value in cfg["nnunet_labels"].items()}
    payload = {
        "channel_names": {str(i): name for i, name in enumerate(channel_names)},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dataset_name_with_id(dataset_id: int, dataset_name: str) -> str:
    if dataset_name.startswith(f"Dataset{dataset_id:03d}_"):
        return dataset_name
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CARE CineMyoPS to motion-texture nnU-Net format.")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Path to CAREdatasets, Myo_train, or CineMyoPS_train.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--frame-index", type=int, default=0,
                   help="Reference ED frame. The released NIfTI files do not expose ED index; default 0.")
    p.add_argument("--dataset-id", type=int, default=608)
    p.add_argument("--dataset-name", default="CARE_CineMyoPS_MotionTexture")
    p.add_argument("--flow-mode", choices=["none", "farneback_agg"], default="none",
                   help="Optional optical-flow motion summary channels.")
    p.add_argument("--flow-frame-stride", type=int, default=1,
                   help="Use every Nth cine frame for optical-flow aggregation.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.flow_frame_stride < 1:
        raise ValueError("--flow-frame-stride must be >= 1")

    cfg = load_json(args.config)
    dataset_name = dataset_name_with_id(args.dataset_id, args.dataset_name)
    cine_root = resolve_cine_root(args.data_root)
    dataset_dir = args.dataset_root / "nnUNet_raw" / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    remap = {int(k): int(v) for k, v in cfg["label_remap_source_to_nnunet"].items()}
    cases = list(iter_cases(cine_root))

    print(f"cine_root    = {cine_root}")
    print(f"dataset_dir  = {dataset_dir}")
    print(f"frame_index  = {args.frame_index}")
    print(f"flow_mode    = {args.flow_mode}")
    print(f"cases        = {len(cases)}")
    if args.dry_run:
        for center, case_id, cine_path, gd_path in cases[:10]:
            print(f"  {center}/{case_id}: {cine_path.name}, {gd_path.name}")
        return

    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    channel_names: list[str] | None = None
    manifest = []
    for center, case_id, cine_path, gd_path in cases:
        cine_img = nib.load(str(cine_path))
        gd_img = nib.load(str(gd_path))
        if len(cine_img.shape) != 4:
            raise ValueError(f"Expected 4D cine image for {case_id}, got {cine_img.shape}")
        if args.frame_index < 0 or args.frame_index >= cine_img.shape[3]:
            raise IndexError(f"frame_index={args.frame_index} outside 0..{cine_img.shape[3] - 1}")
        if cine_img.shape[:3] != gd_img.shape:
            raise ValueError(f"Shape mismatch for {case_id}: cine={cine_img.shape}, gd={gd_img.shape}")

        cine = np.asanyarray(cine_img.dataobj).astype(np.float32)
        n_frames = cine.shape[3]
        case_channel_names = [f"cine_t{t:02d}" for t in range(n_frames)]

        for t in range(n_frames):
            nib.save(
                make_3d_image(cine[..., t], cine_img, np.float32),
                str(images_tr / f"{case_id}_{t:04d}.nii.gz"),
            )

        motion_features = compute_diff_motion_features(cine, args.frame_index)
        if args.flow_mode == "farneback_agg":
            motion_features.extend(compute_farneback_features(cine, args.frame_index, args.flow_frame_stride))

        for offset, (name, feature) in enumerate(motion_features, start=n_frames):
            case_channel_names.append(name)
            nib.save(
                make_3d_image(feature, cine_img, np.float32),
                str(images_tr / f"{case_id}_{offset:04d}.nii.gz"),
            )

        if channel_names is None:
            channel_names = case_channel_names
        elif channel_names != case_channel_names:
            raise ValueError(f"Channel names/count mismatch for {case_id}")

        label_arr = remap_label(np.asanyarray(gd_img.dataobj), remap)
        label_nii = nib.Nifti1Image(label_arr, gd_img.affine, gd_img.header)
        label_nii.header.set_data_dtype(np.int16)
        nib.save(label_nii, str(labels_tr / f"{case_id}.nii.gz"))

        manifest.append({
            "center": center,
            "case_id": case_id,
            "source_cine": str(cine_path),
            "source_label": str(gd_path),
            "cine_shape": list(cine_img.shape),
            "label_shape": list(gd_img.shape),
            "reference_frame_index": args.frame_index,
            "num_cine_frames": n_frames,
            "num_channels": len(case_channel_names),
        })

    channel_names = channel_names or []
    write_dataset_json(dataset_dir, cfg, len(manifest), channel_names)
    motion_feature_names = DIFF_MOTION_FEATURE_NAMES.copy()
    if args.flow_mode == "farneback_agg":
        motion_feature_names.extend(FARNEBACK_FEATURE_NAMES)
    with (dataset_dir / "care_motion_texture_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_root": str(cine_root),
            "reference_frame_index": args.frame_index,
            "flow_mode": args.flow_mode,
            "flow_frame_stride": args.flow_frame_stride,
            "texture_channel_names": channel_names[: manifest[0]["num_cine_frames"]] if manifest else [],
            "motion_feature_names": motion_feature_names,
            "label_remap_source_to_nnunet": cfg["label_remap_source_to_nnunet"],
            "cases": manifest,
        }, f, indent=2)
    print(f"Converted {len(manifest)} cases with {len(channel_names)} channel(s).")
    print(f"nnU-Net dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
