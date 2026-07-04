from __future__ import annotations

import torch
import torch.nn.functional as F


def _base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((x, y), dim=-1)
    return grid.unsqueeze(0).expand(batch, height, width, 2)


def warp_2d(moving: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp moving images with pixel displacement fields in ED reference space."""

    if moving.ndim != 4 or flow.ndim != 4:
        raise ValueError(f"Expected moving (B,C,H,W) and flow (B,2,H,W), got {moving.shape} and {flow.shape}")
    if flow.shape[1] != 2:
        raise ValueError(f"Expected two displacement channels, got {flow.shape[1]}")
    if moving.shape[0] != flow.shape[0] or moving.shape[-2:] != flow.shape[-2:]:
        raise ValueError(f"Moving image and flow shape mismatch: {moving.shape} vs {flow.shape}")

    b, _, h, w = moving.shape
    grid = _base_grid(b, h, w, moving.device, moving.dtype)
    flow_xy = flow.permute(0, 2, 3, 1).to(dtype=moving.dtype)
    sample = grid + flow_xy
    if w > 1:
        sample_x = 2.0 * sample[..., 0] / float(w - 1) - 1.0
    else:
        sample_x = torch.zeros_like(sample[..., 0])
    if h > 1:
        sample_y = 2.0 * sample[..., 1] / float(h - 1) - 1.0
    else:
        sample_y = torch.zeros_like(sample[..., 1])
    normalized = torch.stack((sample_x, sample_y), dim=-1)
    return F.grid_sample(moving, normalized, mode="bilinear", padding_mode="border", align_corners=True)


def smoothness_loss(flow: torch.Tensor) -> torch.Tensor:
    """First-order total variation smoothness for displacement fields."""

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected flow (B,2,H,W), got {flow.shape}")
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1]).mean()
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
    return dx + dy
