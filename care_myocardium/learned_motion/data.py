from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MotionPairRecord:
    center: str
    case_id: str
    cine_path: Path
    slice_index: int
    frame_index: int


def resolve_cine_root(data_root: Path) -> Path:
    candidates = [
        data_root / "Myo_train" / "CineMyoPS_train",
        data_root / "CineMyoPS_train",
        data_root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and list(candidate.glob("*/*_Cine.nii.gz")):
            return candidate
    raise FileNotFoundError(f"Could not find CineMyoPS_train under {data_root}")


def iter_cine_cases(data_root: Path) -> list[tuple[str, str, Path]]:
    cine_root = resolve_cine_root(data_root)
    cases: list[tuple[str, str, Path]] = []
    for center_dir in sorted(p for p in cine_root.iterdir() if p.is_dir()):
        for cine_path in sorted(center_dir.glob("*_Cine.nii.gz")):
            case_id = cine_path.name.removesuffix("_Cine.nii.gz")
            cases.append((center_dir.name, case_id, cine_path))
    if not cases:
        raise FileNotFoundError(f"No *_Cine.nii.gz files found under {cine_root}")
    return cases


def normalize_pair(fixed: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair = np.stack([fixed, moving]).astype(np.float32, copy=False)
    mean = float(pair.mean())
    std = float(pair.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    pair = (pair - mean) / std
    return pair[0], pair[1]


def center_crop_or_pad(slice_2d: np.ndarray, size: int) -> np.ndarray:
    h, w = slice_2d.shape
    out = np.zeros((size, size), dtype=np.float32)
    src_h = min(h, size)
    src_w = min(w, size)
    src_y0 = max((h - size) // 2, 0)
    src_x0 = max((w - size) // 2, 0)
    dst_y0 = max((size - h) // 2, 0)
    dst_x0 = max((size - w) // 2, 0)
    out[dst_y0 : dst_y0 + src_h, dst_x0 : dst_x0 + src_w] = slice_2d[
        src_y0 : src_y0 + src_h,
        src_x0 : src_x0 + src_w,
    ]
    return out


class MotionPairDataset(Dataset):
    """ED-to-frame 2D registration pairs from 4D CARE cine NIfTI files."""

    def __init__(
        self,
        data_root: Path,
        num_frames: int = 30,
        ed_frame_index: int = 0,
        image_size: int | None = None,
        cache_in_memory: bool = False,
    ):
        self.data_root = Path(data_root)
        self.num_frames = int(num_frames)
        self.ed_frame_index = int(ed_frame_index)
        self.image_size = int(image_size) if image_size else None
        self.cache_in_memory = bool(cache_in_memory)
        self._cache: dict[Path, np.ndarray] = {}
        self.records: list[MotionPairRecord] = []

        for center, case_id, cine_path in iter_cine_cases(self.data_root):
            img = nib.load(str(cine_path))
            if len(img.shape) != 4:
                raise ValueError(f"Expected 4D cine image for {case_id}, got {img.shape}")
            z_slices = int(img.shape[2])
            n_frames = min(int(img.shape[3]), self.num_frames)
            if self.ed_frame_index < 0 or self.ed_frame_index >= n_frames:
                raise IndexError(f"ED frame {self.ed_frame_index} outside 0..{n_frames - 1} for {case_id}")
            for z in range(z_slices):
                for t in range(n_frames):
                    if t == self.ed_frame_index:
                        continue
                    self.records.append(MotionPairRecord(center, case_id, cine_path, z, t))

    def __len__(self) -> int:
        return len(self.records)

    def _load_cine(self, path: Path) -> np.ndarray:
        if self.cache_in_memory and path in self._cache:
            return self._cache[path]
        arr = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
        if self.cache_in_memory:
            self._cache[path] = arr
        return arr

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        cine = self._load_cine(rec.cine_path)
        fixed, moving = normalize_pair(
            cine[:, :, rec.slice_index, self.ed_frame_index],
            cine[:, :, rec.slice_index, rec.frame_index],
        )
        if self.image_size:
            fixed = center_crop_or_pad(fixed, self.image_size)
            moving = center_crop_or_pad(moving, self.image_size)
        return {
            "fixed": torch.from_numpy(fixed[None]),
            "moving": torch.from_numpy(moving[None]),
            "case_id": rec.case_id,
            "center": rec.center,
            "slice_index": rec.slice_index,
            "frame_index": rec.frame_index,
        }
