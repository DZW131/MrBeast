from __future__ import annotations

import math
from pathlib import Path
import unittest

from care_myocardium.nnunet_ext.scar_checkpointing import (
    best_scar_from_history,
    scar_dice_from_pseudo_dice,
    should_update_best_scar,
)


class ScarCheckpointingTests(unittest.TestCase):
    def test_scar_dice_defaults_to_last_foreground_class(self) -> None:
        pseudo_dice = [0.7037, 0.9132, 0.4425]

        self.assertAlmostEqual(scar_dice_from_pseudo_dice(pseudo_dice), 0.4425)

    def test_scar_dice_supports_explicit_index(self) -> None:
        pseudo_dice = [0.11, 0.22, 0.33]

        self.assertAlmostEqual(scar_dice_from_pseudo_dice(pseudo_dice, scar_index=1), 0.22)

    def test_best_scar_ignores_nan_history(self) -> None:
        history = [[0.1, 0.2, math.nan], [0.2, 0.3, 0.41], [0.3, 0.4, 0.39]]

        self.assertAlmostEqual(best_scar_from_history(history), 0.41)

    def test_should_update_only_for_finite_improvements(self) -> None:
        self.assertTrue(should_update_best_scar(0.42, None))
        self.assertTrue(should_update_best_scar(0.43, 0.42))
        self.assertFalse(should_update_best_scar(0.41, 0.42))
        self.assertFalse(should_update_best_scar(math.nan, 0.42))

    def test_learned_motion_trainer_writes_scar_specific_checkpoint(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        trainer = repo / "care_myocardium" / "nnunet_ext" / "LearnedMotionSeg400EpochTrainer.py"

        body = trainer.read_text(encoding="utf-8")

        self.assertIn("checkpoint_best_scar.pth", body)
        self.assertIn("Best scar pseudo Dice", body)


if __name__ == "__main__":
    unittest.main()
