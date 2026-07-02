#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a CARE ED nnU-Net plan whose architecture matches the MWM SAX baseline."
    )
    parser.add_argument("--care-preprocessed", required=True, type=Path)
    parser.add_argument("--mwm-preprocessed", required=True, type=Path)
    parser.add_argument("--care-dataset-id", default="601")
    parser.add_argument("--mwm-dataset-id", default="501")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--output-plans-name", default="nnUNetPlans_MWMSAXArch")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[16, 256, 256])
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def find_dataset(root: Path, dataset_id: str) -> Path:
    matches = sorted(root.glob(f"Dataset{dataset_id}_*"))
    if not matches:
        raise FileNotFoundError(f"Could not find Dataset{dataset_id}_* under {root}")
    return matches[0]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    care_dir = find_dataset(args.care_preprocessed, args.care_dataset_id)
    mwm_dir = find_dataset(args.mwm_preprocessed, args.mwm_dataset_id)

    care_plan_path = care_dir / "nnUNetPlans.json"
    mwm_plan_path = mwm_dir / "nnUNetPlans.json"
    care_plan = load_json(care_plan_path)
    mwm_plan = load_json(mwm_plan_path)

    care_config = care_plan["configurations"][args.configuration]
    mwm_config = mwm_plan["configurations"][args.configuration]

    output_plan = deepcopy(care_plan)
    output_plan["plans_name"] = args.output_plans_name
    output_config = output_plan["configurations"][args.configuration]
    output_config["architecture"] = deepcopy(mwm_config["architecture"])
    output_config["patch_size"] = list(args.patch_size)
    output_config["batch_size"] = int(args.batch_size)

    # Keep CARE preprocessing/data_identifier, labels, transpose, spacing and normalization.
    output_config["data_identifier"] = care_config["data_identifier"]

    output_path = care_dir / f"{args.output_plans_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_plan, f, indent=2)
        f.write("\n")

    print(f"Wrote {output_path}")
    print(f"CARE plan: {care_plan_path}")
    print(f"MWM SAX plan: {mwm_plan_path}")
    print(f"configuration={args.configuration} patch_size={args.patch_size} batch_size={args.batch_size}")


if __name__ == "__main__":
    main()
