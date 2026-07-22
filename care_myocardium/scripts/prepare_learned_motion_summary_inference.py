#!/usr/bin/env python3
"""Prepare unlabeled CineMyoPS validation images for Dataset612 inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from care_myocardium.learned_motion.export_nnunet import (  # noqa: E402
    FUSION_MODE_MOTION_SUMMARY,
    build_learned_motion_channel_names,
    center_crop_or_pad_volume,
    iter_export_channels,
    load_motion_model,
    make_3d_image,
    predict_case_frame_flows,
)

DEFAULT_OUTPUT_DIR = ROOT / "DATASET" / "submission_work" / "learned_motion_summary_imagesTs"
DEFAULT_MANIFEST = ROOT / "DATASET" / "submission_work" / "learned_motion_summary_manifest.json"


def case_id_from_path(path: Path) -> str:
    if not path.name.endswith("_Cine.nii.gz"):
        raise ValueError(f"Not a CineMyoPS cine file: {path}")
    return path.name.removesuffix("_Cine.nii.gz")


def infer_center(path: Path, case_id: str) -> str:
    if path.parent.name == case_id and path.parent.parent.name:
        return path.parent.parent.name
    return path.parent.name


def iter_cine_files(data_root: Path) -> list[Path]:
    files = sorted(p for p in data_root.rglob("*_Cine.nii.gz") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No *_Cine.nii.gz files found under {data_root}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export unlabeled CineMyoPS cine cases as Dataset612 41-channel imagesTs."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained Motion-Net checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--ed-frame-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_motion_model(args.checkpoint.resolve(), device)
    channel_names = build_learned_motion_channel_names(args.num_frames, FUSION_MODE_MOTION_SUMMARY)

    seen: set[str] = set()
    cases = []
    for cine_path in iter_cine_files(data_root):
        case_id = case_id_from_path(cine_path)
        if case_id in seen:
            raise ValueError(f"Duplicate case id {case_id}: {cine_path}")
        seen.add(case_id)

        existing = list(out_dir.glob(f"{case_id}_*.nii.gz"))
        if existing and not args.overwrite:
            raise FileExistsError(f"{case_id}: output channels already exist; pass --overwrite")
        for path in existing:
            path.unlink()

        cine_img = nib.load(str(cine_path))
        if len(cine_img.shape) != 4:
            raise ValueError(f"{cine_path}: expected 4D cine image, got shape {cine_img.shape}")
        n_frames = min(cine_img.shape[3], args.num_frames)
        if n_frames != args.num_frames:
            raise ValueError(f"{case_id}: has {n_frames} frames, expected {args.num_frames}")
        if args.ed_frame_index < 0 or args.ed_frame_index >= cine_img.shape[3]:
            raise IndexError(f"{case_id}: invalid ED frame index {args.ed_frame_index}")

        cine = np.asanyarray(cine_img.dataobj).astype(np.float32)
        cine_frames = [
            center_crop_or_pad_volume(cine[..., t], args.image_size, np.float32)
            for t in range(args.num_frames)
        ]
        frame_flows = predict_case_frame_flows(
            model,
            cine,
            device,
            args.num_frames,
            args.ed_frame_index,
            args.image_size,
        )
        export_channels = iter_export_channels(cine_frames, frame_flows, FUSION_MODE_MOTION_SUMMARY, device)
        if len(export_channels) != len(channel_names):
            raise RuntimeError(f"{case_id}: exported {len(export_channels)} channels, expected {len(channel_names)}")

        for offset, channel in enumerate(export_channels):
            out_path = out_dir / f"{case_id}_{offset:04d}.nii.gz"
            nib.save(make_3d_image(channel, cine_img, np.float32), str(out_path))

        cases.append({
            "case_id": case_id,
            "source_center": infer_center(cine_path, case_id),
            "source_cine": str(cine_path.resolve()),
            "cine_shape": list(cine_img.shape),
            "ed_frame_index": args.ed_frame_index,
            "num_frames": args.num_frames,
            "image_size": args.image_size,
            "channel_count": len(channel_names),
        })
        print(f"{case_id}: exported {len(channel_names)} channels")

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "data_root": str(data_root),
            "output_dir": str(out_dir),
            "motion_checkpoint": str(args.checkpoint.resolve()),
            "fusion_mode": FUSION_MODE_MOTION_SUMMARY,
            "channel_names": channel_names,
            "cases": cases,
        }, f, indent=2)
    print(f"Prepared {len(cases)} cases.")
    print(f"manifest = {manifest_path}")


if __name__ == "__main__":
    main()
