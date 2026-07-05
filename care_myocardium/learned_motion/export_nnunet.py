from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .data import center_crop_or_pad, iter_cine_cases, normalize_pair
from .model import MotionUNet

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_cine_myops_to_nnunet import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DATASET_ROOT,
    load_json,
    remap_label,
)


FUSION_MODE_RAW_STACK = "raw_stack"
FUSION_MODE_FRAMEWISE_CONCAT = "framewise_concat"
FUSION_MODES = (FUSION_MODE_RAW_STACK, FUSION_MODE_FRAMEWISE_CONCAT)


def build_learned_motion_channel_names(
    num_frames: int,
    fusion_mode: str = FUSION_MODE_RAW_STACK,
) -> list[str]:
    if fusion_mode == FUSION_MODE_RAW_STACK:
        names = [f"cine_t{t:02d}" for t in range(num_frames)]
        for t in range(1, num_frames):
            names.extend([f"motion_t{t:02d}_dx", f"motion_t{t:02d}_dy"])
        return names

    if fusion_mode == FUSION_MODE_FRAMEWISE_CONCAT:
        names = ["cine_t00"]
        for t in range(1, num_frames):
            names.extend([f"cine_t{t:02d}", f"motion_t{t:02d}_dx", f"motion_t{t:02d}_dy"])
        return names

    raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}; expected one of {FUSION_MODES}")


def make_3d_image(data: np.ndarray, ref_img: nib.Nifti1Image, dtype: np.dtype | type) -> nib.Nifti1Image:
    header = ref_img.header.copy()
    zooms = header.get_zooms()[:3]
    out = nib.Nifti1Image(data.astype(dtype), ref_img.affine, header)
    out.header.set_data_shape(data.shape)
    out.header.set_zooms(zooms)
    out.header.set_data_dtype(dtype)
    return out


def center_crop_or_pad_volume(volume: np.ndarray, size: int | None, dtype: np.dtype | type | None = None) -> np.ndarray:
    arr = np.asarray(volume)
    out_dtype = dtype if dtype is not None else arr.dtype
    if size is None:
        return arr.astype(out_dtype, copy=False)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got shape {arr.shape}")
    h, w = arr.shape[:2]
    trailing = arr.shape[2:]
    out = np.zeros((int(size), int(size), *trailing), dtype=out_dtype)
    src_h = min(h, int(size))
    src_w = min(w, int(size))
    src_y0 = max((h - int(size)) // 2, 0)
    src_x0 = max((w - int(size)) // 2, 0)
    dst_y0 = max((int(size) - h) // 2, 0)
    dst_x0 = max((int(size) - w) // 2, 0)
    out[dst_y0 : dst_y0 + src_h, dst_x0 : dst_x0 + src_w, ...] = arr[
        src_y0 : src_y0 + src_h,
        src_x0 : src_x0 + src_w,
        ...,
    ].astype(out_dtype, copy=False)
    return out


def dataset_name_with_id(dataset_id: int, dataset_name: str) -> str:
    if dataset_name.startswith(f"Dataset{dataset_id:03d}_"):
        return dataset_name
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def load_motion_model(checkpoint: Path, device: torch.device) -> MotionUNet:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = MotionUNet(
        in_channels=2,
        base_channels=int(cfg.get("base_channels", 16)),
        max_flow=float(cfg.get("max_flow", 12.0)),
    ).to(device)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_case_frame_flows(
    model: MotionUNet,
    cine: np.ndarray,
    device: torch.device,
    num_frames: int,
    ed_frame_index: int = 0,
    image_size: int | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    h, w, z, total_frames = cine.shape
    n_frames = min(total_frames, num_frames)
    out_h = int(image_size) if image_size else h
    out_w = int(image_size) if image_size else w
    flows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for t in range(1, n_frames):
        dx = np.zeros((out_h, out_w, z), dtype=np.float32)
        dy = np.zeros((out_h, out_w, z), dtype=np.float32)
        for zi in range(z):
            fixed, moving = normalize_pair(cine[:, :, zi, ed_frame_index], cine[:, :, zi, t])
            if image_size:
                fixed = center_crop_or_pad(fixed, int(image_size))
                moving = center_crop_or_pad(moving, int(image_size))
            x = torch.from_numpy(np.stack([fixed, moving])[None]).to(device=device, dtype=torch.float32)
            flow = model(x)[0].detach().cpu().numpy()
            dx[:, :, zi] = flow[0]
            dy[:, :, zi] = flow[1]
        flows.append((t, dx, dy))
    return flows


@torch.no_grad()
def predict_case_flows(
    model: MotionUNet,
    cine: np.ndarray,
    device: torch.device,
    num_frames: int,
    ed_frame_index: int = 0,
    image_size: int | None = None,
) -> list[np.ndarray]:
    flows: list[np.ndarray] = []
    for _, dx, dy in predict_case_frame_flows(model, cine, device, num_frames, ed_frame_index, image_size):
        flows.extend([dx, dy])
    return flows


def iter_export_channels(
    cine_frames: list[np.ndarray],
    frame_flows: list[tuple[int, np.ndarray, np.ndarray]],
    fusion_mode: str,
) -> list[np.ndarray]:
    if fusion_mode == FUSION_MODE_RAW_STACK:
        channels = list(cine_frames)
        for _, dx, dy in frame_flows:
            channels.extend([dx, dy])
        return channels

    if fusion_mode == FUSION_MODE_FRAMEWISE_CONCAT:
        channels = [cine_frames[0]]
        for t, dx, dy in frame_flows:
            channels.extend([cine_frames[t], dx, dy])
        return channels

    raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}; expected one of {FUSION_MODES}")


def write_dataset_json(dataset_dir: Path, cfg: dict, num_training: int, channel_names: list[str]) -> None:
    labels = {name: int(value) for name, value in cfg["nnunet_labels"].items()}
    payload = {
        "channel_names": {str(i): name for i, name in enumerate(channel_names)},
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    with (dataset_dir / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export learned ED-referenced motion fields as nnU-Net channels.")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-id", type=int, default=610)
    p.add_argument("--dataset-name", default="CARE_CineMyoPS_LearnedMotionTexture")
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--ed-frame-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=192)
    p.add_argument("--fusion-mode", choices=FUSION_MODES, default=FUSION_MODE_RAW_STACK,
                   help="raw_stack keeps the legacy all-cine-then-all-motion layout; "
                        "framewise_concat exports ED followed by (frame, dx, dy) groups.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_json(args.config)
    remap = {int(k): int(v) for k, v in cfg["label_remap_source_to_nnunet"].items()}
    dataset_name = dataset_name_with_id(args.dataset_id, args.dataset_name)
    dataset_dir = args.dataset_root / "nnUNet_raw" / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    if dataset_dir.exists() and args.overwrite:
        shutil.rmtree(dataset_dir)
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_motion_model(args.checkpoint, device)
    channel_names = build_learned_motion_channel_names(args.num_frames, args.fusion_mode)
    cases = iter_cine_cases(args.data_root)
    manifest = []

    for center, case_id, cine_path in cases:
        gd_path = cine_path.with_name(f"{case_id}_gd.nii.gz")
        cine_img = nib.load(str(cine_path))
        gd_img = nib.load(str(gd_path))
        if cine_img.shape[:3] != gd_img.shape:
            raise ValueError(f"Shape mismatch for {case_id}: cine={cine_img.shape}, gd={gd_img.shape}")
        cine = np.asanyarray(cine_img.dataobj).astype(np.float32)
        n_frames = min(cine.shape[3], args.num_frames)
        if n_frames != args.num_frames:
            raise ValueError(f"{case_id} has {n_frames} frames, expected {args.num_frames}")

        cine_frames = [
            center_crop_or_pad_volume(cine[..., t], args.image_size, np.float32)
            for t in range(args.num_frames)
        ]
        frame_flows = predict_case_frame_flows(
            model,
            cine,
            device,
            args.num_frames,
            args.ed_frame_index,
            args.image_size,
        )
        export_channels = iter_export_channels(cine_frames, frame_flows, args.fusion_mode)
        if len(export_channels) != len(channel_names):
            raise RuntimeError(
                f"Exported {len(export_channels)} channel arrays but dataset.json has {len(channel_names)} names"
            )

        for offset, channel in enumerate(export_channels):
            nib.save(make_3d_image(channel, cine_img, np.float32), str(images_tr / f"{case_id}_{offset:04d}.nii.gz"))

        label_arr = remap_label(np.asanyarray(gd_img.dataobj), remap)
        label_arr = center_crop_or_pad_volume(label_arr, args.image_size, np.int16)
        label_nii = nib.Nifti1Image(label_arr, gd_img.affine, gd_img.header)
        label_nii.header.set_data_dtype(np.int16)
        nib.save(label_nii, str(labels_tr / f"{case_id}.nii.gz"))
        manifest.append({"center": center, "case_id": case_id, "source_cine": str(cine_path)})

    write_dataset_json(dataset_dir, cfg, len(manifest), channel_names)
    with (dataset_dir / "learned_motion_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_root": str(args.data_root),
                "motion_checkpoint": str(args.checkpoint),
                "ed_frame_index": args.ed_frame_index,
                "num_frames": args.num_frames,
                "image_size": args.image_size,
                "fusion_mode": args.fusion_mode,
                "channel_names": channel_names,
                "cases": manifest,
            },
            f,
            indent=2,
        )
    print(f"Exported {len(manifest)} cases with {len(channel_names)} channels to {dataset_dir}")


if __name__ == "__main__":
    main()
