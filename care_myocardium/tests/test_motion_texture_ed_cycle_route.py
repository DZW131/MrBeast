from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class MotionTextureEDCycleRouteTests(unittest.TestCase):
    def test_motion_texture_ed_cycle_training_route_is_documented_and_launchable(self) -> None:
        trainer_path = REPO_ROOT / "care_myocardium" / "nnunet_ext" / "MotionTextureEDCycleTrainer.py"
        script_path = REPO_ROOT / "care_myocardium" / "scripts" / "train_motion_texture_ed_cycle_nnunet.sh"

        self.assertTrue(trainer_path.exists())
        self.assertTrue(script_path.exists())

        trainer_source = trainer_path.read_text(encoding="utf-8")
        script_source = script_path.read_text(encoding="utf-8")

        self.assertIn("class MotionTextureEDCycleTrainer", trainer_source)
        self.assertIn('MOTION_TEXTURE_ED_CYCLE_EPOCHS", "300"', trainer_source)
        self.assertIn('MOTION_TEXTURE_ED_CYCLE_WEIGHT", "0.1"', trainer_source)
        self.assertIn("MotionTextureEDCycleTrainer", script_source)
        self.assertIn('CARE_DATASET_ID="${CARE_DATASET_ID:-609}"', script_source)
        self.assertIn('MOTION_TEXTURE_ED_CYCLE_EPOCHS="${MOTION_TEXTURE_ED_CYCLE_EPOCHS:-300}"', script_source)


if __name__ == "__main__":
    unittest.main()
