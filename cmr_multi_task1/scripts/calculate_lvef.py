#!/usr/bin/env python3
"""Compute Left Ventricular Ejection Fraction (LVEF) from SAX Cine segmentation masks.

Ported from the official CMR-MULTI baseline (3D_seg/calculate_lv_ef.py) so the
metric matches the challenge evaluation. The 3rd axis of a Cine SAX volume is
spatial-slices x temporal-phases stacked; phases are recovered via the
sax_slice_info mapping (case_id -> number of cardiac phases).

SAX label convention (see configs/cine_labels.json):
    0 background, 1 LV_Myo, 2 LV_Cavity, 3 RV_Cavity
LV blood pool = label 2.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import mode

LV_BLOOD_POOL_ID = 2


def create_3d_blocks(data: np.ndarray, num_blocks: int):
    """Split the Cine volume into temporal phases (baseline logic)."""
    total_slices = data.shape[2]
    if total_slices < num_blocks:
        return None
    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        return None
    blocks = []
    for block_idx in range(num_blocks):
        slice_indices = [block_idx + i * num_blocks for i in range(slices_per_block)]
        if len(slice_indices) > 10:
            effective_indices = slice_indices[1:-1]
        else:
            effective_indices = slice_indices
        blocks.append(data[:, :, effective_indices])
    return blocks


def calculate_lv_ef(mask_path: Path, slice_num: int):
    try:
        nii = nib.load(str(mask_path))
        pred = nii.get_fdata()
        spacing = nii.header.get_zooms()
        blocks = create_3d_blocks(pred, slice_num)
        if not blocks:
            return None
        valid_z_lengths = []
        for block in blocks:
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            count = int(np.sum(has_lv))
            if count > 0:
                valid_z_lengths.append(count)
        if not valid_z_lengths:
            return None
        target_z_length = mode(valid_z_lengths, keepdims=False).mode
        block_volumes = []
        for block in blocks:
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z = np.where(has_lv)[0]
            if len(valid_z) != target_z_length:
                continue
            lv_voxels = np.sum(block[..., valid_z] == LV_BLOOD_POOL_ID)
            lv_vol = lv_voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0
            block_volumes.append(lv_vol)
        if not block_volumes:
            return None
        block_volumes.sort()
        es_lv_vol = block_volumes[0]
        ed_lv_vol = block_volumes[-1]
        lv_ef = ((ed_lv_vol - es_lv_vol) / ed_lv_vol * 100) if ed_lv_vol > 0 else 0.0
        return {"LV_EF": float(lv_ef), "LV_EDV": float(ed_lv_vol), "LV_ESV": float(es_lv_vol)}
    except Exception as e:
        logging.error("Error processing %s: %s", mask_path, e)
        return None


def case_id_from_name(name: str) -> str:
    """CINE_SAX_001.nii.gz -> 001 (last underscore-delimited numeric chunk)."""
    base = name
    for suf in (".nii.gz", ".nii"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    parts = base.split("_")
    for p in reversed(parts):
        if p.isdigit():
            return p
    return base[-3:]


def convert(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(i) for i in obj]
    return obj


def parse_args():
    p = argparse.ArgumentParser(description="Calculate LVEF from SAX Cine masks.")
    p.add_argument("--pred-dir", type=Path, required=True,
                   help="Directory of predicted SAX masks (.nii.gz).")
    p.add_argument("--slice-info", type=Path, required=True,
                   help="JSON mapping case_id -> number of cardiac phases "
                        "(e.g. sax_slice_info.json / sax_slice_info_test.json).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output JSON path (default <pred-dir>/lv_ef_results.json).")
    p.add_argument("--lv-label", type=int, default=LV_BLOOD_POOL_ID,
                   help="LV blood pool label id (SAX default 2).")
    return p.parse_args()


def main():
    args = parse_args()
    global LV_BLOOD_POOL_ID
    LV_BLOOD_POOL_ID = args.lv_label
    with args.slice_info.open("r") as f:
        slice_info = json.load(f)
    out = args.output or (args.pred_dir / "lv_ef_results.json")
    results, errors = [], []
    if not args.pred_dir.exists():
        print(f"Error: {args.pred_dir} does not exist")
        return
    files = sorted(p for p in args.pred_dir.iterdir() if p.name.endswith(".nii.gz"))
    print(f"Processing {len(files)} masks in {args.pred_dir} ...")
    for fp in files:
        case_id = case_id_from_name(fp.name)
        slice_num = slice_info.get(str(case_id))
        if slice_num is None:
            errors.append({"id": case_id, "error": f"{case_id} not in {args.slice_info.name}"})
            continue
        m = calculate_lv_ef(fp, int(slice_num))
        if m:
            results.append({"id": case_id, **m})
            print(f"  {case_id}: EF={m['LV_EF']:.2f}%")
        else:
            errors.append({"id": case_id, "error": "calculation returned None"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(convert(results), f, ensure_ascii=False, indent=4)
    err_out = out.with_name(out.name.replace("results", "errors"))
    with err_out.open("w", encoding="utf-8") as f:
        json.dump(convert(errors), f, ensure_ascii=False, indent=4)
    print(f"\nDone. {len(results)} ok, {len(errors)} errors.")
    print(f"Results -> {out}")
    print(f"Errors  -> {err_out}")


if __name__ == "__main__":
    main()
