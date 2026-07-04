from __future__ import annotations

import os

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from .scar_checkpointing import best_scar_from_history, scar_dice_from_pseudo_dice, should_update_best_scar
except ImportError:
    from scar_checkpointing import best_scar_from_history, scar_dice_from_pseudo_dice, should_update_best_scar


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
        self.scar_dice_index = int(os.environ.get("LEARNED_MOTION_SCAR_DICE_INDEX", "-1"))
        self._best_scar_dice: float | None = None

    def on_epoch_end(self):
        self._maybe_save_best_scar_checkpoint()
        super().on_epoch_end()

    def _maybe_save_best_scar_checkpoint(self) -> None:
        dice_history = self.logger.get_value("dice_per_class_or_region", step=None)
        if not dice_history:
            return

        current_scar_dice = scar_dice_from_pseudo_dice(dice_history[-1], self.scar_dice_index)
        previous_best = self._best_scar_dice
        if previous_best is None:
            previous_best = best_scar_from_history(dice_history[:-1], self.scar_dice_index)

        if should_update_best_scar(current_scar_dice, previous_best):
            self._best_scar_dice = current_scar_dice
            self.print_to_log_file(f"Best scar pseudo Dice: {np.round(self._best_scar_dice, decimals=4)}")
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_best_scar.pth"))
        else:
            self._best_scar_dice = previous_best
