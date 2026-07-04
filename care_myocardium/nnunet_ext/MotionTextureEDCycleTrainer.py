from __future__ import annotations

import os

import torch

try:
    from EDCycleConsistencyTrainer import EDCycleConsistencyTrainer
except ImportError:
    from .EDCycleConsistencyTrainer import EDCycleConsistencyTrainer


class MotionTextureEDCycleTrainer(EDCycleConsistencyTrainer):
    """Motion-texture trainer with first/last ED cycle consistency."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("MOTION_TEXTURE_ED_CYCLE_EPOCHS", "300"))
        self.ed_cycle_weight = float(os.environ.get("MOTION_TEXTURE_ED_CYCLE_WEIGHT", "0.1"))
