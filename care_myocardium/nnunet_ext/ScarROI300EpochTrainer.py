from __future__ import annotations

import os

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class ScarROI300EpochTrainer(nnUNetTrainer):
    """Short nnU-Net trainer for second-stage scar ROI refinement."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = int(os.environ.get("SCAR_ROI_EPOCHS", "300"))
