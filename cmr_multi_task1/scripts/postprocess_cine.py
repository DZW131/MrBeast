#!/usr/bin/env python3
"""Class-aware MR postprocessing for CMR-MULTI Task 1 Cine predictions.

For each view, per-class connected-component cleanup is applied:
  - keep the top-k largest components,
  - reject components farther than max_distance_to_heart_mm from the heart body
    (the largest union-of-foreground component),
  - optionally fill holes within the kept mask (only into background voxels).

Rules are read from configs/cine_postprocess_rules.json and can be tuned per
view / label. Input predictions keep the original geometry (spacing/affine).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = ROOT / "configs" / "cine_postprocess_rules.json"
DEFAULT_CONFIG = ROOT / "configs" / "cine_labels.json"


def largest_component(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num <= 1:
        return mask
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == int(sizes.argmax())


def largest_foreground_component(seg: np.ndarray, labels: list[int]) -> np.ndarray:
    fg = np.isin(seg, labels)
    return largest_component(fg)


def remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask
    sizes = np.bincount(cc.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[sizes >= min_size] = True
    keep[0] = False
    return keep[cc]


def normalize_spacing(spacing: Iterable[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None:
        return tuple(1.0 for _ in range(ndim))
    values = tuple(float(v) for v in spacing)
    return values if len(values) == ndim else tuple(values[:ndim]) + (1.0,) * (ndim - len(values))


def component_filter(mask: np.ndarray, rule: dict, heart_distance: np.ndarray | None) -> np.ndarray:
    if not np.any(mask):
        return mask
    structure = ndimage.generate_binary_structure(mask.ndim, mask.ndim)
    cc, num = ndimage.label(mask, structure=structure)
    if num == 0:
        return mask
    sizes = np.bincount(cc.ravel())
    comp_ids = np.arange(1, num + 1)
    min_size = int(rule.get("min_component_size", 0) or 0)
    keep_ids = comp_ids[sizes[comp_ids] >= min_size]
    max_dist = rule.get("max_distance_to_heart_mm")
    if max_dist is not None and heart_distance is not None and keep_ids.size:
        dists = np.atleast_1d(ndimage.minimum(heart_distance, labels=cc, index=keep_ids)).astype(float)
        keep_ids = keep_ids[dists <= float(max_dist)]
    if not keep_ids.size:
        return np.zeros_like(mask, dtype=bool)
    top_k = int(rule.get("keep_top_k", rule.get("keep_largest", False) and 1 or 0))
    if top_k > 0 and keep_ids.size > top_k:
        order = np.argsort(sizes[keep_ids])[::-1]
        keep_ids = keep_ids[order[:top_k]]
    return np.isin(cc, keep_ids)


def postprocess_label_array(seg: np.ndarray, labels: list[int],
                             class_rules: dict[int, dict],
                             spacing: Iterable[float] | None) -> np.ndarray:
    cleaned = seg.copy()
    heart_distance = None
    heart_body = largest_foreground_component(seg, labels)
    if np.any(heart_body):
        heart_distance = ndimage.distance_transform_edt(
            ~heart_body, sampling=normalize_spacing(spacing, seg.ndim))
    for label in labels:
        mask = seg == int(label)
        rule = class_rules.get(int(label), {"keep_top_k": 1})
        mask = component_filter(mask, rule, heart_distance=heart_distance)
        if bool(rule.get("fill_holes", False)):
            filled = ndimage.binary_fill_holes(mask)
            mask = mask | (filled & (seg == 0))
        cleaned[seg == int(label)] = 0
        cleaned[mask] = int(label)
    return cleaned


def load_nii(path: Path):
    import nibabel as nib
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.affine, nii.header, nib


def parse_args():
    p = argparse.ArgumentParser(description="Class-aware MR postprocessing for Cine predictions.")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory of predicted .nii.gz masks.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory for cleaned masks.")
    p.add_argument("--view", required=True, choices=["sax", "2ch", "4ch"],
                   help="View decides the label set and rules.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    p.add_argument("--rules-override", type=Path, default=None,
                   help="Optional JSON overriding/augmenting per-label rules.")
    return p.parse_args()


def main():
    import nibabel as nib
    args = parse_args()
    cfg = json.load(args.config.open("r", encoding="utf-8"))
    view_cfg = cfg["views"][args.view]
    labels = [int(v) for v in view_cfg["labels"].values() if int(v) > 0]
    rules = json.load(args.rules.open("r", encoding="utf-8"))[args.view]
    class_rules = {int(k): dict(v) for k, v in rules.items()}
    if args.rules_override is not None:
        ov = json.load(args.rules_override.open("r", encoding="utf-8"))
        for k, v in ov.get(args.view, {}).items():
            class_rules.setdefault(int(k), {}).update(v)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.name.endswith(".nii.gz"))
    print(f"Postprocessing view={args.view} labels={labels} files={len(files)}")
    for fp in files:
        arr, affine, header, _ = load_nii(fp)
        spacing = header.get_zooms()
        cleaned = postprocess_label_array(arr.astype(np.int16), labels, class_rules, spacing)
        out = nib.Nifti1Image(cleaned.astype(np.int16), affine, header)
        nib.save(out, str(args.output_dir / fp.name))
        print(f"  {fp.name}")
    print(f"Done -> {args.output_dir}")


if __name__ == "__main__":
    main()
