from __future__ import annotations

import math

import torch
from torch import nn


class CineTemporalMemoryStem(nn.Module):
    """SAM2-inspired temporal memory reader for 4D cine volumes.

    nnU-Net sees cine time as input channels. This module treats channel 0 as the
    supervised ED query frame and the remaining channels as a compact memory
    bank. At every voxel it reads memory values with temporal attention, then
    returns one enhanced ED volume for the downstream 3D segmentation network.
    """

    def __init__(
        self,
        num_frames: int,
        embed_dim: int = 8,
        query_frame_index: int = 0,
        residual_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if query_frame_index < 0:
            raise ValueError("query_frame_index must be >= 0")

        self.num_frames = int(num_frames)
        self.embed_dim = int(embed_dim)
        self.query_frame_index = int(query_frame_index)

        self.frame_encoder = nn.Sequential(
            nn.Conv3d(1, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(embed_dim, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )
        self.query_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv3d(embed_dim * 2, 1, kernel_size=1)
        self.phase_embedding = nn.Parameter(torch.zeros(num_frames, embed_dim, 1, 1, 1))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor [B, T, X, Y, Z], got shape {tuple(x.shape)}")

        batch_size, frames, *spatial_shape = x.shape
        if frames == 1:
            return x[:, :1]
        if frames > self.num_frames:
            raise ValueError(f"Input has {frames} frames, but this stem was built for {self.num_frames}")
        if self.query_frame_index >= frames:
            raise ValueError(
                f"query_frame_index={self.query_frame_index} outside available frames 0..{frames - 1}"
            )

        ed_frame = x[:, self.query_frame_index:self.query_frame_index + 1]
        frame_features = self.frame_encoder(x.reshape(batch_size * frames, 1, *spatial_shape))
        frame_features = frame_features.reshape(batch_size, frames, self.embed_dim, *spatial_shape)
        phase = self.phase_embedding[:frames].to(dtype=frame_features.dtype, device=frame_features.device)
        frame_features = frame_features + phase.unsqueeze(0)

        query = frame_features[:, self.query_frame_index]
        query_proj = self.query_proj(query).unsqueeze(1)

        flat_features = frame_features.reshape(batch_size * frames, self.embed_dim, *spatial_shape)
        keys = self.key_proj(flat_features).reshape(batch_size, frames, self.embed_dim, *spatial_shape)
        values = self.value_proj(flat_features).reshape(batch_size, frames, self.embed_dim, *spatial_shape)

        attention_logits = (query_proj * keys).sum(dim=2) / math.sqrt(self.embed_dim)
        attention = torch.softmax(attention_logits, dim=1)
        memory_context = (attention.unsqueeze(2) * values).sum(dim=1)

        enhancement = self.out_proj(torch.cat([query, memory_context], dim=1))
        return ed_frame + self.residual_scale.to(dtype=x.dtype, device=x.device) * enhancement


class CineMemorySegmentationNetwork(nn.Module):
    """Wrap a standard 3D nnU-Net with a cine memory stem."""

    def __init__(
        self,
        base_network: nn.Module,
        num_input_channels: int,
        memory_embed_dim: int = 8,
        query_frame_index: int = 0,
        residual_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.memory_stem = CineTemporalMemoryStem(
            num_frames=num_input_channels,
            embed_dim=memory_embed_dim,
            query_frame_index=query_frame_index,
            residual_scale_init=residual_scale_init,
        )
        self.base_network = base_network

    def forward(self, x: torch.Tensor):
        return self.base_network(self.memory_stem(x))

    def compute_conv_feature_map_size(self, input_size):
        if hasattr(self.base_network, "compute_conv_feature_map_size"):
            return self.base_network.compute_conv_feature_map_size(input_size)
        raise AttributeError("base_network does not implement compute_conv_feature_map_size")
