#!/usr/bin/env python3
"""Package CMR-MULTI Task 1 predictions for Codabench validation upload."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "DATASET" / "submissions"

VIEW_DIRS = {
    "sax": "SAX",
    "2ch": "2CH",
    "4ch": "4CH",
}

LABEL_SETS = {
    "sax": {0, 1, 2, 3},
    "2ch": {0, 1, 2},
    "4ch": {0, 1, 2, 3, 4, 5},
}


def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Not a NIfTI file: {name}")


def prediction_files(path: Path) -> list[Path]:
    files = sorted(p for p in path.iterdir() if p.name.endswith((".nii", ".nii.gz")))
    if not files:
        raise FileNotFoundError(f"No NIfTI predictions found in {path}")
    return files


def validate_labels(pred_path: Path, allowed: set[int]) -> None:
    img = nib.load(str(pred_path))
    values = set(int(v) for v in np.unique(np.asanyarray(img.dataobj)))
    extra = values - allowed
    if extra:
        raise ValueError(f"{pred_path}: unexpected labels {sorted(extra)}, allowed {sorted(allowed)}")


def normalize_ef_key(case_id: str) -> str:
    case_id = str(case_id)
    if case_id.startswith("CINE_SAX_"):
        return case_id
    return f"CINE_SAX_{int(case_id):03d}" if case_id.isdigit() else case_id


def load_ef_predictions(path: Path) -> dict[str, float]:
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            data = data["results"]
        else:
            return {normalize_ef_key(k): float(v) for k, v in data.items()}
    if not isinstance(data, list):
        raise ValueError(f"Unsupported EF JSON shape in {path}")
    out: dict[str, float] = {}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError(f"Unsupported EF row in {path}: {row!r}")
        key = row.get("case_id", row.get("id"))
        if key is None:
            raise ValueError(f"EF row lacks id/case_id: {row}")
        value = row.get("LV_EF", row.get("ef", row.get("EF")))
        if value is None:
            raise ValueError(f"EF row lacks LV_EF/ef/EF: {row}")
        out[normalize_ef_key(str(key))] = float(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create submission.zip containing task1_cine predictions."
    )
    parser.add_argument("--sax-dir", type=Path, required=True)
    parser.add_argument("--2ch-dir", type=Path, required=True)
    parser.add_argument("--4ch-dir", type=Path, required=True)
    parser.add_argument("--ef-json", type=Path, required=True,
                        help="lv_ef_results.json or already-normalized ef_predictions.json.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--zip-name", default="submission_task1_cine.zip")
    parser.add_argument("--include-empty-task2", action="store_true",
                        help="Also create empty task2_lge skeleton and mass_predictions.json.")
    parser.add_argument("--skip-label-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dirs = {
        "sax": args.sax_dir.resolve(),
        "2ch": args.__dict__["2ch_dir"].resolve(),
        "4ch": args.__dict__["4ch_dir"].resolve(),
    }
    ef_json = args.ef_json.resolve()
    out_root = args.output_root.resolve()
    stage_root = out_root / args.zip_name.removesuffix(".zip")
    zip_path = out_root / args.zip_name

    if stage_root.exists():
        shutil.rmtree(stage_root)
    out_root.mkdir(parents=True, exist_ok=True)

    counts = {}
    for view, src_dir in pred_dirs.items():
        dst_dir = stage_root / "task1_cine" / VIEW_DIRS[view]
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = prediction_files(src_dir)
        counts[view] = len(files)
        for fp in files:
            if not args.skip_label_check:
                validate_labels(fp, LABEL_SETS[view])
            shutil.copy2(fp, dst_dir / fp.name)

    ef_predictions = load_ef_predictions(ef_json)
    sax_case_ids = {strip_nii_suffix(p.name) for p in prediction_files(pred_dirs["sax"])}
    missing_ef = sorted(sax_case_ids - set(ef_predictions))
    extra_ef = sorted(set(ef_predictions) - sax_case_ids)
    if missing_ef:
        raise ValueError(f"Missing EF predictions for SAX cases: {missing_ef[:10]}")
    if extra_ef:
        ef_predictions = {k: v for k, v in ef_predictions.items() if k in sax_case_ids}
    with (stage_root / "task1_cine" / "ef_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(dict(sorted(ef_predictions.items())), f, indent=2)

    if args.include_empty_task2:
        task2 = stage_root / "task2_lge"
        for name in ("SAX", "2CH", "4CH", "RAS"):
            (task2 / name).mkdir(parents=True, exist_ok=True)
        with (task2 / "mass_predictions.json").open("w", encoding="utf-8") as f:
            json.dump({}, f)

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(stage_root.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage_root))

    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.namelist()
    print(f"Packaged Task 1 Cine predictions: {counts}")
    print(f"EF predictions: {len(ef_predictions)}")
    print(f"stage_root = {stage_root}")
    print(f"zip_path   = {zip_path}")
    print(f"entries    = {len(entries)}")


if __name__ == "__main__":
    main()
