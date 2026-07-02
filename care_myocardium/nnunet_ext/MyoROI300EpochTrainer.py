from __future__ import annotations

import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class MyoROI300EpochTrainer(nnUNetTrainer):
    """Short nnU-Net trainer for myocardium ROI refinement."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("MYO_ROI_EPOCHS", os.environ.get("ROI_EPOCHS", "300")))
