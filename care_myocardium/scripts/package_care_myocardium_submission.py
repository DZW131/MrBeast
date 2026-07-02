#!/usr/bin/env python3
"""Package CARE Myocardium predictions for validation leaderboard upload."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cine_myops_labels.json"
DEFAULT_MANIFEST = ROOT / "DATASET" / "submission_work" / "cine_myops_inference_manifest.json"
DEFAULT_OUTPUT_ROOT = ROOT / "DATASET" / "submissions"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file name: {name}")


def resolve_case_id(pred_path: Path) -> str:
    stem = strip_nii_suffix(pred_path.name)
    if stem.endswith("_pred"):
        stem = stem[:-5]
    return stem


def infer_label_space(values: set[int], source_values: set[int], nnunet_values: set[int]) -> str:
    if values <= nnunet_values and any(v in values for v in (1, 2, 3)):
        return "nnunet"
    if values <= source_values:
        return "source"
    if values <= nnunet_values:
        return "nnunet"
    raise ValueError(f"Unexpected label values {sorted(values)}")


def restore_or_copy(
    pred_path: Path,
    out_path: Path,
    label_space: str,
    source_values: set[int],
    nnunet_values: set[int],
    remap: dict[int, int],
) -> str:
    img = nib.load(str(pred_path))
    arr = np.asanyarray(img.dataobj).astype(np.int16)
    values = set(int(v) for v in np.unique(arr))
    if label_space == "auto":
        effective_space = infer_label_space(values, source_values, nnunet_values)
    else:
        effective_space = label_space

    if effective_space == "source":
        if not values <= source_values:
            raise ValueError(f"{pred_path.name}: source-space labels expected, got {sorted(values)}")
        shutil.copy2(pred_path, out_path)
        return "source"

    if effective_space != "nnunet":
        raise ValueError(f"Unsupported label space: {effective_space}")
    if not values <= nnunet_values:
        raise ValueError(f"{pred_path.name}: nnU-Net labels expected, got {sorted(values)}")
    out = np.zeros(arr.shape, dtype=np.int16)
    for src, dst in remap.items():
        out[arr == src] = dst
    nii = nib.Nifti1Image(out, img.affine, img.header)
    nii.header.set_data_dtype(np.int16)
    nib.save(nii, str(out_path))
    return "nnunet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CARE-Myocardium-TeamName.zip from CineMyoPS predictions."
    )
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory containing nnU-Net outputs, e.g. CaseXXXX.nii.gz.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Manifest produced by prepare_cine_myops_inference.py.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--team-name", default="Monster")
    parser.add_argument("--center-name", default="Anonymous Center")
    parser.add_argument("--input-label-space", choices=["auto", "nnunet", "source"], default="auto")
    parser.add_argument("--allow-extra", action="store_true",
                        help="Do not fail if pred-dir contains predictions outside the manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.resolve()
    manifest = load_json(args.manifest.resolve())
    cfg = load_json(args.config.resolve())
    source_values = set(int(v) for v in cfg["source_labels"].values())
    nnunet_values = set(int(v) for v in cfg["nnunet_labels"].values())
    remap = {int(k): int(v) for k, v in cfg["label_remap_nnunet_to_source"].items()}

    pred_files = sorted(p for p in pred_dir.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    pred_by_case = {resolve_case_id(p): p for p in pred_files}
    if len(pred_by_case) != len(pred_files):
        raise ValueError(f"Duplicate case predictions in {pred_dir}")

    expected_cases = [case["case_id"] for case in manifest["cases"]]
    missing = [case_id for case_id in expected_cases if case_id not in pred_by_case]
    extra = sorted(set(pred_by_case) - set(expected_cases))
    if missing:
        raise FileNotFoundError(f"Missing predictions for {len(missing)} cases: {missing[:10]}")
    if extra and not args.allow_extra:
        raise ValueError(f"Unexpected predictions outside manifest: {extra[:10]}")

    output_root = args.output_root.resolve()
    package_name = f"CARE-Myocardium-{args.team_name}"
    stage_root = output_root / package_name
    zip_path = output_root / f"{package_name}.zip"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_spaces = []
    for case_id in expected_cases:
        case_dir = stage_root / "CineMyoPS" / args.center_name / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / f"{case_id}_pred.nii.gz"
        label_spaces.append(restore_or_copy(
            pred_by_case[case_id],
            out_path,
            args.input_label_space,
            source_values,
            nnunet_values,
            remap,
        ))

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage_root))

    entries = zipfile.ZipFile(zip_path).namelist()
    expected_count = len(expected_cases)
    if len(entries) != expected_count:
        raise RuntimeError(f"Zip has {len(entries)} files, expected {expected_count}")
    print(f"Packaged {expected_count} CineMyoPS predictions.")
    print(f"Input label spaces used: {sorted(set(label_spaces))}")
    print(f"stage_root = {stage_root}")
    print(f"zip_path   = {zip_path}")


if __name__ == "__main__":
    main()
