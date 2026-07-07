from __future__ import annotations

import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context
from torch import autocast

try:
    from ed_cycle_consistency import build_last_ed_reference_view, ed_cycle_consistency_loss
except ImportError:
    from .ed_cycle_consistency import build_last_ed_reference_view, ed_cycle_consistency_loss


class EDCycleConsistencyTrainer(nnUNetTrainer):
    """nnU-Net trainer with first-ED <-> last-ED cine cycle consistency."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("ED_CYCLE_EPOCHS", "300"))
        self.ed_cycle_num_frames = int(os.environ.get("ED_CYCLE_NUM_FRAMES", "30"))
        self.ed_cycle_weight = float(os.environ.get("ED_CYCLE_WEIGHT", "0.2"))
        self.ed_cycle_ramp_epochs = int(os.environ.get("ED_CYCLE_RAMP_EPOCHS", "40"))
        self.ed_cycle_start_epoch = int(os.environ.get("ED_CYCLE_START_EPOCH", "0"))
        self.ed_cycle_confidence = float(os.environ.get("ED_CYCLE_CONFIDENCE", "0.6"))
        self.ed_cycle_scar_weight = float(os.environ.get("ED_CYCLE_SCAR_WEIGHT", "2.0"))

    def _current_cycle_weight(self) -> float:
        epoch = int(getattr(self, "current_epoch", 0))
        if epoch < self.ed_cycle_start_epoch:
            return 0.0
        if self.ed_cycle_ramp_epochs <= 0:
            return self.ed_cycle_weight
        ramp_pos = epoch - self.ed_cycle_start_epoch + 1
        return self.ed_cycle_weight * min(1.0, ramp_pos / float(self.ed_cycle_ramp_epochs))

    def _build_cycle_view(self, data: torch.Tensor) -> torch.Tensor:
        return build_last_ed_reference_view(data, self.ed_cycle_num_frames)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        cycle_weight = self._current_cycle_weight()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            first_ed_output = self.network(data)
            supervised_loss = self.loss(first_ed_output, target)
            if cycle_weight > 0:
                last_ed_data = self._build_cycle_view(data)
                last_ed_output = self.network(last_ed_data)
                cycle_loss = ed_cycle_consistency_loss(
                    first_ed_output,
                    last_ed_output,
                    confidence_threshold=self.ed_cycle_confidence,
                    scar_weight=self.ed_cycle_scar_weight,
                )
                total_loss = supervised_loss + cycle_weight * cycle_loss
            else:
                cycle_loss = supervised_loss.detach() * 0
                total_loss = supervised_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "supervised_loss": supervised_loss.detach().cpu().numpy(),
            "ed_cycle_loss": cycle_loss.detach().cpu().numpy(),
        }
