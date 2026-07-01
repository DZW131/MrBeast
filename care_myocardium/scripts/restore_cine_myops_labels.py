#!/usr/bin/env python3
"""Restore CARE CineMyoPS original label ids for submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cine_myops_labels.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Map nnU-Net labels 1/2/3 back to CARE labels 200/500/2221.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.load(args.config.open("r", encoding="utf-8"))
    remap = {int(k): int(v) for k, v in cfg["label_remap_nnunet_to_source"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.input_dir.iterdir() if p.name.endswith(".nii.gz"))
    print(f"Restoring {len(files)} masks -> {args.output_dir}")
    for fp in files:
        img = nib.load(str(fp))
        arr = np.asanyarray(img.dataobj).astype(np.int16)
        out = np.zeros(arr.shape, dtype=np.int16)
        seen = set(int(v) for v in np.unique(arr))
        extra = seen - set(remap)
        if extra:
            raise ValueError(f"{fp.name}: unexpected labels {sorted(extra)}")
        for src, dst in remap.items():
            out[arr == src] = dst
        nii = nib.Nifti1Image(out, img.affine, img.header)
        nii.header.set_data_dtype(np.int16)
        nib.save(nii, str(args.output_dir / fp.name))
        print(f"  {fp.name}")


if __name__ == "__main__":
    main()
