from __future__ import annotations

import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class LearnedMotionSeg400EpochTrainer(nnUNetTrainer):
    """400-epoch Seg-Net route for learned motion + cine texture channels."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("LEARNED_MOTION_SEG_EPOCHS", "400"))
