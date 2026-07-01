from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import nn

try:
    from cine_memory_network import CineMemorySegmentationNetwork
except ImportError:
    from .cine_memory_network import CineMemorySegmentationNetwork


class CineMemoryTrainer(nnUNetTrainer):
    """nnU-Net trainer that prepends a SAM2-inspired cine memory reader."""

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        memory_embed_dim = int(os.environ.get("CINE_MEMORY_EMBED_DIM", "8"))
        query_frame_index = int(os.environ.get("CINE_MEMORY_QUERY_FRAME_INDEX", "0"))
        residual_scale_init = float(os.environ.get("CINE_MEMORY_RESIDUAL_SCALE_INIT", "0.001"))

        base_network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            1,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return CineMemorySegmentationNetwork(
            base_network=base_network,
            num_input_channels=num_input_channels,
            memory_embed_dim=memory_embed_dim,
            query_frame_index=query_frame_index,
            residual_scale_init=residual_scale_init,
        )
