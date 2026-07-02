#!/usr/bin/env python3
"""Prepare CARE CineMyoPS cine images for ED-only nnU-Net inference.

The validation/test release is expected to contain 4D files named
`CaseXXXX_Cine.nii.gz`, usually without labels. This script extracts the
configured frame into nnU-Net's `imagesTs` naming convention:

    CaseXXXX_Cine.nii.gz -> CaseXXXX_0000.nii.gz

It also writes a manifest that the submission packager uses to restore the
official case folder structure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "DATASET" / "submission_work" / "cine_myops_ed_imagesTs"
DEFAULT_MANIFEST = ROOT / "DATASET" / "submission_work" / "cine_myops_inference_manifest.json"


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


def make_3d_image(data: np.ndarray, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    header = ref_img.header.copy()
    zooms = header.get_zooms()[:3]
    out = nib.Nifti1Image(data.astype(ref_img.get_data_dtype()), ref_img.affine, header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(zooms)
    out.header.set_data_dtype(ref_img.get_data_dtype())
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one cine frame per case into nnU-Net imagesTs format."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Folder containing validation/test *_Cine.nii.gz files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output imagesTs-style directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="JSON manifest with case ids and source paths.")
    parser.add_argument("--frame-index", type=int, default=0,
                        help="Frame index used by the ED baseline. Default: 0.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    manifest_path = args.manifest.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    cases = []
    for cine_path in iter_cine_files(data_root):
        case_id = case_id_from_path(cine_path)
        if case_id in seen:
            raise ValueError(f"Duplicate case id {case_id}: {cine_path}")
        seen.add(case_id)

        img = nib.load(str(cine_path))
        if len(img.shape) != 4:
            raise ValueError(f"{cine_path}: expected 4D cine image, got shape {img.shape}")
        if args.frame_index < 0 or args.frame_index >= img.shape[3]:
            raise IndexError(
                f"{cine_path}: frame_index={args.frame_index} outside 0..{img.shape[3] - 1}"
            )

        out_path = out_dir / f"{case_id}_0000.nii.gz"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite to replace it")
        data = np.asanyarray(img.dataobj[..., args.frame_index])
        nib.save(make_3d_image(data, img), str(out_path))
        cases.append({
            "case_id": case_id,
            "source_center": infer_center(cine_path, case_id),
            "source_cine": str(cine_path),
            "output_image": str(out_path),
            "cine_shape": list(img.shape),
            "frame_index": args.frame_index,
        })
        print(f"{case_id}: {cine_path} -> {out_path}")

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({
            "data_root": str(data_root),
            "output_dir": str(out_dir),
            "frame_index": args.frame_index,
            "cases": cases,
        }, f, indent=2)
    print(f"Prepared {len(cases)} cases.")
    print(f"manifest = {manifest_path}")


if __name__ == "__main__":
    main()
