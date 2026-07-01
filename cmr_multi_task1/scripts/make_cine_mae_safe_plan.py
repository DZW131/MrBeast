#!/usr/bin/env python3
"""Create a CMR cine-friendly MAE plan from the TaWald nnSSL plan."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the latest TaWald MAE ptPlans JSON and replace its 160^3 "
            "patch with a CMR cine patch derived from nnUNetPlans.json."
        )
    )
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--pretraining-name", required=True)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--source-plan", type=Path, default=None)
    parser.add_argument("--base-plan", default="nnUNetPlans.json")
    parser.add_argument("--stride-multiple", default=32, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)
        f.write("\n")


def round_patch(patch_size: list[int], multiple: int) -> list[int]:
    if multiple < 1:
        raise ValueError("--stride-multiple must be >= 1")
    return [int(math.ceil(int(v) / multiple) * multiple) for v in patch_size]


def find_source_plan(dataset_dir: Path, pretraining_name: str) -> Path:
    candidates = sorted(dataset_dir.glob(f"ptPlans__{pretraining_name}*.json"))
    unpatched = [p for p in candidates if "____Patch__" not in p.stem]
    if unpatched:
        return unpatched[-1]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No ptPlans__{pretraining_name}*.json found in {dataset_dir}"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    base_plan_path = dataset_dir / args.base_plan
    source_plan_path = args.source_plan or find_source_plan(
        dataset_dir, args.pretraining_name
    )

    base_plan = load_json(base_plan_path)
    source_plan = load_json(source_plan_path)
    config_name = args.configuration
    base_config = base_plan["configurations"][config_name]
    source_config = source_plan["configurations"][config_name]

    base_patch = [int(v) for v in base_config["patch_size"]]
    safe_patch = round_patch(base_patch, args.stride_multiple)
    source_patch = [int(v) for v in source_config["patch_size"]]

    new_plan = json.loads(json.dumps(source_plan))
    new_config = new_plan["configurations"][config_name]
    new_config["patch_size"] = safe_patch

    pretrain_info = new_plan.setdefault("pretrain_info", {})
    pretrain_info["cmr_multi_safe_patch"] = {
        "source_plans_name": source_plan["plans_name"],
        "source_patch_size": source_patch,
        "base_nnunet_patch_size": base_patch,
        "safe_patch_size": safe_patch,
        "stride_multiple": args.stride_multiple,
        "reason": (
            "Use the cine nnU-Net patch shape rounded up to ResEncL's "
            "5-stage stride multiple instead of the generic 160^3 MAE patch."
        ),
    }

    original_name = source_plan["plans_name"].split("____Patch__")[0]
    patch_tag = "x".join(str(v) for v in safe_patch)
    new_name = f"{original_name}____Patch__{patch_tag}"
    new_plan["plans_name"] = new_name
    output_path = dataset_dir / f"{new_name}.json"

    if output_path.exists() and not args.force:
        existing = load_json(output_path)
        existing_patch = existing["configurations"][config_name]["patch_size"]
        if [int(v) for v in existing_patch] != safe_patch:
            raise RuntimeError(
                f"{output_path} already exists with patch_size={existing_patch}; "
                "rerun with --force to overwrite it."
            )

    save_json(output_path, new_plan)
    print(f"source_plan={source_plan_path.name}")
    print(f"output_plan={output_path.name}")
    print(f"source_patch={source_patch}")
    print(f"base_patch={base_patch}")
    print(f"safe_patch={safe_patch}")


if __name__ == "__main__":
    main()
