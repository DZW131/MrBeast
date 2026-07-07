from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class LearnedMotionSummaryEDCycleRouteTests(unittest.TestCase):
    def test_learned_motion_summary_ed_cycle_route_is_launchable(self) -> None:
        trainer_path = REPO_ROOT / "care_myocardium" / "nnunet_ext" / "LearnedMotionSummaryEDCycleTrainer.py"
        script_path = REPO_ROOT / "care_myocardium" / "scripts" / "train_learned_motion_summary_ed_cycle_nnunet.sh"

        self.assertTrue(trainer_path.exists())
        self.assertTrue(script_path.exists())

        trainer_source = trainer_path.read_text(encoding="utf-8")
        script_source = script_path.read_text(encoding="utf-8")

        self.assertIn("class LearnedMotionSummaryEDCycleTrainer", trainer_source)
        self.assertIn('LEARNED_MOTION_ED_CYCLE_EPOCHS", "300"', trainer_source)
        self.assertIn('LEARNED_MOTION_ED_CYCLE_LR", "0.001"', trainer_source)
        self.assertIn('LEARNED_MOTION_ED_CYCLE_WEIGHT", "0.1"', trainer_source)
        self.assertIn("checkpoint_best_scar.pth", trainer_source)
        self.assertIn("LearnedMotionSummaryEDCycleTrainer", script_source)
        self.assertIn('CARE_DATASET_ID="${CARE_DATASET_ID:-612}"', script_source)
        self.assertIn('LEARNED_MOTION_ED_CYCLE_EPOCHS="${LEARNED_MOTION_ED_CYCLE_EPOCHS:-300}"', script_source)


if __name__ == "__main__":
    unittest.main()
