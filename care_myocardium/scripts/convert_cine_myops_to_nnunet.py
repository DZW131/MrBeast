#!/usr/bin/env python3
"""Convert CARE CineMyoPS train data to nnU-Net v2 raw format.

The source CineMyoPS files are one 4D cine image per case:

    Case1001_Cine.nii.gz  -> shape (H, W, Z, 30)
    Case1001_gd.nii.gz    -> shape (H, W, Z)

The 3D gd mask labels the reference end-diastolic frame. The released files do
not store an explicit ED frame index in the NIfTI header, so this converter keeps
the frame choice configurable. The conservative default is frame 0.

Source labels are sparse challenge ids:
    200  LV normal myocardium
    500  LV blood pool
    2221 LV myocardial scar

For nnU-Net training they are remapped to contiguous labels:
    1 normal myocardium, 2 LV blood pool, 3 scar
Use restore_cine_myops_labels.py before packaging predictions for submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cine_myops_labels.json"
DEFAULT_DATASET_ROOT = ROOT / "DATASET"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_cine_root(data_root: Path) -> Path:
    if (data_root / "center_alpha").is_dir() and (data_root / "center_beta").is_dir():
        return data_root
    cand = data_root / "Myo_train" / "CineMyoPS_train"
    if cand.is_dir():
        return cand
    cand = data_root / "CineMyoPS_train"
    if cand.is_dir():
        return cand
    raise FileNotFoundError(
        f"Could not find CineMyoPS_train under {data_root}. "
        "Expected center_alpha/center_beta or Myo_train/CineMyoPS_train."
    )


def iter_cases(cine_root: Path) -> Iterable[tuple[str, str, Path, Path]]:
    for center_dir in sorted(p for p in cine_root.iterdir() if p.is_dir()):
        for cine_path in sorted(center_dir.glob("*_Cine.nii.gz")):
            case_id = cine_path.name.replace("_Cine.nii.gz", "")
            gd_path = cine_path.with_name(f"{case_id}_gd.nii.gz")
            if not gd_path.is_file():
                raise FileNotFoundError(f"Missing gd mask for {cine_path}: {gd_path}")
            yield center_dir.name, case_id, cine_path, gd_path


def remap_label(arr: np.ndarray, remap: dict[int, int]) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.int16)
    seen = set(int(v) for v in np.unique(arr))
    allowed = set(remap)
    extra = seen - allowed
    if extra:
        raise ValueError(f"Unexpected labels {sorted(extra)}; allowed {sorted(allowed)}")
    for src, dst in remap.items():
        if src != 0:
            out[arr == src] = dst
    return out


def make_3d_image(data: np.ndarray, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    header = ref_img.header.copy()
    zooms = header.get_zooms()[:3]
    out = nib.Nifti1Image(data.astype(ref_img.get_data_dtype()), ref_img.affine, header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(zooms)
    out.header.set_data_dtype(ref_img.get_data_dtype())
    return out


def save_frame(cine_img: nib.Nifti1Image, frame_index: int, out_path: Path) -> None:
    if len(cine_img.shape) != 4:
        raise ValueError(f"Expected 4D cine image, got shape {cine_img.shape}")
    if frame_index < 0 or frame_index >= cine_img.shape[3]:
        raise IndexError(f"frame_index={frame_index} outside 0..{cine_img.shape[3] - 1}")
    data = np.asanyarray(cine_img.dataobj[..., frame_index])
    nib.save(make_3d_image(data, cine_img), str(out_path))


def save_all_frames(cine_img: nib.Nifti1Image, out_pattern: str, out_dir: Path) -> int:
    if len(cine_img.shape) != 4:
        raise ValueError(f"Expected 4D cine image, got shape {cine_img.shape}")
    n_frames = cine_img.shape[3]
    for frame_index in range(n_frames):
        data = np.asanyarray(cine_img.dataobj[..., frame_index])
        nib.save(make_3d_image(data, cine_img), str(out_dir / out_pattern.format(frame_index)))
    return n_frames


def write_dataset_json(dataset_dir: Path, cfg: dict, num_training: int, n_channels: int) -> None:
    labels = {name: int(value) for name, value in cfg["nnunet_labels"].items()}
    channel_names = {str(i): ("cine_ed" if n_channels == 1 else f"cine_t{i:02d}") for i in range(n_channels)}
    payload = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CARE CineMyoPS to nnU-Net raw format.")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Path to CAREdatasets, Myo_train, or CineMyoPS_train.")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--mode", choices=["ed", "all_frames"], default="ed",
                   help="ed: one configured frame as one channel; all_frames: 30 cine frames as channels.")
    p.add_argument("--frame-index", type=int, default=0,
                   help="Frame used in --mode ed. The dataset does not expose an ED index; default 0.")
    p.add_argument("--dataset-id", type=int, default=None)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    dataset_id = args.dataset_id or int(cfg["dataset_id"])
    dataset_name = args.dataset_name or cfg["dataset_name"]
    if not dataset_name.startswith(f"Dataset{dataset_id:03d}_"):
        dataset_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    cine_root = resolve_cine_root(args.data_root)
    dataset_dir = args.dataset_root / "nnUNet_raw" / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    manifest = []
    remap = {int(k): int(v) for k, v in cfg["label_remap_source_to_nnunet"].items()}

    cases = list(iter_cases(cine_root))
    print(f"cine_root    = {cine_root}")
    print(f"dataset_dir  = {dataset_dir}")
    print(f"mode         = {args.mode}")
    print(f"frame_index  = {args.frame_index}")
    print(f"cases        = {len(cases)}")
    if args.dry_run:
        for center, case_id, cine_path, gd_path in cases[:10]:
            print(f"  {center}/{case_id}: {cine_path.name}, {gd_path.name}")
        return

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    n_channels = None
    for center, case_id, cine_path, gd_path in cases:
        cine_img = nib.load(str(cine_path))
        gd_img = nib.load(str(gd_path))
        if cine_img.shape[:3] != gd_img.shape:
            raise ValueError(f"Shape mismatch for {case_id}: cine={cine_img.shape}, gd={gd_img.shape}")
        label_arr = remap_label(np.asanyarray(gd_img.dataobj), remap)
        label_out = labels_tr / f"{case_id}.nii.gz"
        label_nii = nib.Nifti1Image(label_arr, gd_img.affine, gd_img.header)
        label_nii.header.set_data_dtype(np.int16)
        nib.save(label_nii, str(label_out))

        if args.mode == "ed":
            save_frame(cine_img, args.frame_index, images_tr / f"{case_id}_0000.nii.gz")
            n_case_channels = 1
        else:
            n_case_channels = save_all_frames(cine_img, f"{case_id}_{{:04d}}.nii.gz", images_tr)
        if n_channels is None:
            n_channels = n_case_channels
        elif n_channels != n_case_channels:
            raise ValueError(f"Channel count mismatch: expected {n_channels}, got {n_case_channels} for {case_id}")
        manifest.append({
            "center": center,
            "case_id": case_id,
            "source_cine": str(cine_path),
            "source_label": str(gd_path),
            "cine_shape": list(cine_img.shape),
            "label_shape": list(gd_img.shape),
        })

    write_dataset_json(dataset_dir, cfg, len(manifest), int(n_channels or 1))
    with (dataset_dir / "care_cinemyops_manifest.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_root": str(cine_root),
            "mode": args.mode,
            "frame_index": args.frame_index,
            "label_remap_source_to_nnunet": cfg["label_remap_source_to_nnunet"],
            "cases": manifest,
        }, f, indent=2)
    print(f"Converted {len(manifest)} cases with {n_channels} channel(s).")
    print(f"nnU-Net dataset: {dataset_dir}")


if __name__ == "__main__":
    main()
