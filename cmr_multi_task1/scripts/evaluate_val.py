#!/usr/bin/env python3
"""Evaluate predicted Cine masks against held-out validation labels.

Computes per-class Dice and HD95 (in mm) for one view, comparing every file in
the prediction directory to the matching ground truth in the val_labels dir.
Case matching is by stripped filename (CINE_SAX_001.nii.gz).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def strip_nii(name: str) -> str:
    for suf in (".nii.gz", ".nii"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    denom = int(pred.sum() + ref.sum())
    if denom == 0:
        return math.nan
    return float(2 * np.logical_and(pred, ref).sum() / denom)


def hd95_mm(pred: np.ndarray, ref: np.ndarray, spacing_xyz: Sequence[float]) -> float:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    if pred.sum() == 0 and ref.sum() == 0:
        return math.nan
    if pred.sum() == 0 or ref.sum() == 0:
        return math.inf
    pred_surf = pred ^ binary_erosion(pred)
    ref_surf = ref ^ binary_erosion(ref)
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    d_ref = distance_transform_edt(~ref_surf, sampling=spacing_zyx)
    d_pred = distance_transform_edt(~pred_surf, sampling=spacing_zyx)
    dists = np.concatenate([d_ref[pred_surf], d_pred[ref_surf]])
    return float(np.percentile(dists, 95))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Cine val predictions.")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True, help="val_labels directory.")
    p.add_argument("--labels", type=int, nargs="*", default=None,
                   help="Foreground label ids to score (default: all >0 found in GT).")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    import nibabel as nib
    args = parse_args()
    pred_files = sorted(p for p in args.pred_dir.iterdir() if p.name.endswith(".nii.gz"))
    if not pred_files:
        print(f"No predictions in {args.pred_dir}"); return
    # determine label set
    label_set = set(args.labels) if args.labels else None
    per_case = []
    all_dice = {l: [] for l in (label_set or [])}
    all_hd = {l: [] for l in (label_set or [])}
    for pf in pred_files:
        case = strip_nii(pf.name)
        gt = args.gt_dir / f"{case}.nii.gz"
        if not gt.exists():
            print(f"  WARN: no GT for {case}, skip"); continue
        pr = nib.load(str(pf)).get_fdata()
        gt_arr = nib.load(str(gt)).get_fdata()
        spacing = nib.load(str(pf)).header.get_zooms()
        if label_set is None:
            label_set = sorted(int(v) for v in np.unique(gt_arr) if v > 0)
            all_dice = {l: [] for l in label_set}
            all_hd = {l: [] for l in label_set}
        row = {"case": case}
        for l in label_set:
            d = dice(pr == l, gt_arr == l)
            h = hd95_mm(pr == l, gt_arr == l, spacing)
            row[f"dice_{l}"] = d
            row[f"hd95_{l}"] = h
            all_dice[l].append(d)
            all_hd[l].append(h)
        per_case.append(row)
    summary = {}
    for l in label_set:
        ds = [x for x in all_dice[l] if not math.isnan(x)]
        hs = [x for x in all_hd[l] if not (math.isnan(x) or math.isinf(x))]
        summary[l] = {
            "dice_mean": float(np.mean(ds)) if ds else None,
            "hd95_mean": float(np.mean(hs)) if hs else None,
            "n": len(per_case),
        }
    out = args.output or (args.pred_dir / "eval_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_case": per_case}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
