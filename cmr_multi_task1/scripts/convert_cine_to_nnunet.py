#!/usr/bin/env python3
"""Convert CMR-MULTI CINE_MULTI data to nnU-Net v2 raw format.

Each view (SAX / 2CH / 4CH) becomes one nnU-Net dataset because the label
spaces differ per view. Labels in the source data are already contiguous
integers starting at 0, so no remapping is required - we copy and verify.

Layout produced under <dataset_root>/nnUNet_raw/<dataset_name>/:

    imagesTr/<case>_0000.nii.gz      (from *_TR/image)
    labelsTr/<case>.nii.gz           (from *_TR/anno)
    imagesTs/<case>_0000.nii.gz      (from *_VAL/image, held-out validation)
    val_labels/<case>.nii.gz         (from *_VAL/anno, for evaluation only)

Test images (no annotations) are collected under
<dataset_root>/test_images/<view>/ for final Codabench submission.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cine_labels.json"
DEFAULT_DATASET_ROOT = ROOT / "DATASET"
NII_SUFFIXES = (".nii.gz", ".nii")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def list_nii(directory: Path) -> List[Path]:
    if not directory or not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.name.endswith(NII_SUFFIXES) and p.is_file())


def find_split_dir(data_root: Path, view_folder: str, split: str) -> Path:
    # source folders are named e.g. SAX_TR, 2CH_VAL, 4CH_TST
    cand = data_root / f"{view_folder}_{split}"
    if cand.exists():
        return cand
    # tolerate lowercase / case differences
    matches = [p for p in data_root.iterdir()
               if p.is_dir() and p.name.upper() == f"{view_folder}_{split}".upper()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {view_folder}_{split} under {data_root}")


def verify_labels(label_path: Path, allowed: set[int]) -> None:
    """Lightweight label check using nibabel (no copy-time remap needed)."""
    import nibabel as nib
    arr = nib.load(str(label_path)).get_fdata()
    uniq = set(int(v) for v in np.unique(arr))
    extra = uniq - allowed
    if extra:
        raise ValueError(f"Unexpected label values {sorted(extra)} in {label_path} "
                         f"(allowed {sorted(allowed)})")


def copy_cases(images: List[Path], anno_dir: Path | None, dst_img_dir: Path,
               dst_label_dir: Path | None, allowed_labels: set[int],
               verify: bool, dry_run: bool) -> List[dict]:
    records = []
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    if dst_label_dir is not None:
        dst_label_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        case_id = strip_nii(img.name)
        dst_img = dst_img_dir / f"{case_id}_0000.nii.gz"
        records.append({"case_id": case_id, "source_image": str(img), "nnunet_image": str(dst_img)})
        if dry_run:
            continue
        shutil.copy2(img, dst_img)
        if dst_label_dir is not None and anno_dir is not None:
            label = anno_dir / img.name
            if not label.exists():
                raise FileNotFoundError(f"No annotation {label} for image {img.name}")
            dst_label = dst_label_dir / f"{case_id}.nii.gz"
            if verify:
                verify_labels(label, allowed_labels)
            shutil.copy2(label, dst_label)
            records[-1]["source_label"] = str(label)
            records[-1]["nnunet_label"] = str(dst_label)
    return records


def make_dataset_json(dataset_dir: Path, view_cfg: dict, num_training: int) -> None:
    labels = {str(k): v for k, v in view_cfg["labels"].items()}
    # nnU-Net expects label names as keys; background must be 0.
    payload = {
        "channel_names": {"0": view_cfg.get("channel_name", "MR")},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    out = dataset_dir / "dataset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def convert_view(view_key: str, view_cfg: dict, data_root: Path,
                 dataset_root: Path, splits: List[str], verify: bool,
                 dry_run: bool) -> dict:
    view_folder = view_cfg["folder"]
    raw_root = dataset_root / "nnUNet_raw"
    ds_dir = raw_root / view_cfg["dataset_name"]
    allowed = set(int(v) for v in view_cfg["labels"].values())

    summary = {"view": view_key, "dataset_name": view_cfg["dataset_name"],
               "dataset_id": view_cfg["dataset_id"], "splits": {}}

    for split in splits:
        split_dir = find_split_dir(data_root, view_folder, split)
        images = list_nii(split_dir / "image")
        anno_dir = split_dir / "anno"
        if not anno_dir.exists():
            anno_dir = None

        if split == "TR":
            recs = copy_cases(images, anno_dir, ds_dir / "imagesTr",
                              ds_dir / "labelsTr", allowed, verify, dry_run)
            summary["splits"]["train"] = {"images": len(recs), "with_label": sum(1 for r in recs if "nnunet_label" in r)}
        elif split == "VAL":
            recs = copy_cases(images, anno_dir, ds_dir / "imagesTs",
                              ds_dir / "val_labels", allowed, verify, dry_run)
            summary["splits"]["val"] = {"images": len(recs), "with_label": sum(1 for r in recs if "nnunet_label" in r)}
        elif split == "TST":
            test_dir = dataset_root / "test_images" / view_key
            recs = copy_cases(images, None, test_dir, None, allowed, False, dry_run)
            summary["splits"]["test"] = {"images": len(recs)}
        else:
            raise ValueError(f"Unknown split {split}")

    if "TR" in splits and not dry_run:
        make_dataset_json(ds_dir, view_cfg, summary["splits"]["train"]["images"])
    elif "TR" in splits and dry_run:
        # still report expected numTraining
        summary["expected_numTraining"] = summary["splits"]["train"]["images"]
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert CINE_MULTI to nnU-Net raw format.")
    p.add_argument("--data-root", type=Path, required=True,
                   help="Path to CINE_MULTI (containing SAX_TR, 2CH_VAL, ...).")
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT,
                   help="Output root with nnUNet_raw / test_images.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--views", nargs="*", default=None,
                   help="Subset of views to convert (e.g. sax 2ch 4ch). Default: all.")
    p.add_argument("--splits", nargs="*", default=["TR", "VAL", "TST"],
                   choices=["TR", "VAL", "TST"])
    p.add_argument("--dry-run", action="store_true",
                   help="Report discovered cases only; do not copy or verify.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip label value verification (faster, needs nibabel otherwise).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    views = args.views or list(cfg["views"].keys())
    print(f"data_root      = {args.data_root}")
    print(f"dataset_root   = {args.dataset_root}")
    print(f"views          = {views}")
    print(f"splits         = {args.splits}")
    print(f"dry_run        = {args.dry_run}")
    print(f"verify_labels  = {not args.no_verify}")
    print("-" * 60)

    total_summary = []
    for vk in views:
        vc = cfg["views"][vk]
        s = convert_view(vk, vc, args.data_root, args.dataset_root,
                         args.splits, verify=not args.no_verify, dry_run=args.dry_run)
        total_summary.append(s)
        print(f"[{vk}] {s['dataset_name']}")
        for k, v in s["splits"].items():
            print(f"    {k:5s}: {v}")
    print("-" * 60)
    print(json.dumps(total_summary, indent=2))


if __name__ == "__main__":
    main()

