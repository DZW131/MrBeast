from __future__ import annotations

import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MotionTexture1000EpochTrainer(nnUNetTrainer):
    """Paper-length trainer for motion-texture segmentation experiments."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("MOTION_TEXTURE_EPOCHS", "1000"))
