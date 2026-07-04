from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from care_myocardium.learned_motion.data import MotionPairDataset
from care_myocardium.learned_motion.export_nnunet import build_learned_motion_channel_names
from care_myocardium.learned_motion.model import MotionUNet
from care_myocardium.learned_motion.spatial import smoothness_loss, warp_2d


def save_nii(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class LearnedMotionPipelineTests(unittest.TestCase):
    def test_motion_net_training_route_is_documented_and_launchable(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        train_module = repo / "care_myocardium" / "learned_motion" / "train_motion_net.py"
        train_script = repo / "care_myocardium" / "scripts" / "train_motion_net.sh"

        self.assertTrue(train_module.exists())
        self.assertTrue(train_script.exists())
        script = train_script.read_text(encoding="utf-8")
        self.assertIn("MOTION_NET_EPOCHS", script)
        self.assertIn("torchrun", script)
        self.assertIn("1000", script)

    def test_learned_motion_segnet_route_is_documented_and_launchable(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        export_script = repo / "care_myocardium" / "scripts" / "prepare_learned_motion_dataset.sh"
        train_script = repo / "care_myocardium" / "scripts" / "train_learned_motion_seg_nnunet.sh"
        trainer = repo / "care_myocardium" / "nnunet_ext" / "LearnedMotionSeg400EpochTrainer.py"

        self.assertTrue(export_script.exists())
        self.assertTrue(train_script.exists())
        self.assertTrue(trainer.exists())
        self.assertIn("LearnedMotionSeg400EpochTrainer", train_script.read_text(encoding="utf-8"))
        self.assertIn('LEARNED_MOTION_SEG_EPOCHS", "400"', trainer.read_text(encoding="utf-8"))

    def test_warp_with_zero_flow_returns_moving_image_and_smoothness_is_zero(self) -> None:
        moving = torch.randn(2, 1, 8, 9)
        flow = torch.zeros(2, 2, 8, 9)

        warped = warp_2d(moving, flow)

        self.assertTrue(torch.allclose(warped, moving, atol=1e-5))
        self.assertAlmostEqual(float(smoothness_loss(flow)), 0.0, places=6)

    def test_motion_unet_predicts_two_channel_displacement_field(self) -> None:
        net = MotionUNet(in_channels=2, base_channels=4)
        x = torch.randn(2, 2, 16, 16)

        flow = net(x)

        self.assertEqual(tuple(flow.shape), (2, 2, 16, 16))

    def test_motion_pair_dataset_samples_ed_to_each_non_ed_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CAREdatasets" / "Myo_train" / "CineMyoPS_train" / "center_alpha"
            cine = np.zeros((6, 7, 2, 4), dtype=np.float32)
            cine[..., 0] = 10
            cine[..., 1] = 12
            cine[..., 2] = 14
            cine[..., 3] = 16
            save_nii(root / "Case0001_Cine.nii.gz", cine)
            save_nii(root / "Case0001_gd.nii.gz", np.zeros((6, 7, 2), dtype=np.int16))

            dataset = MotionPairDataset(Path(tmp) / "CAREdatasets", num_frames=4)

            self.assertEqual(len(dataset), 6)
            sample = dataset[0]
            self.assertEqual(tuple(sample["fixed"].shape), (1, 6, 7))
            self.assertEqual(tuple(sample["moving"].shape), (1, 6, 7))
            self.assertEqual(sample["case_id"], "Case0001")
            self.assertIn(sample["frame_index"], {1, 2, 3})

    def test_motion_pair_dataset_can_center_crop_or_pad_to_fixed_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CAREdatasets" / "Myo_train" / "CineMyoPS_train" / "center_alpha"
            cine = np.ones((5, 7, 1, 3), dtype=np.float32)
            save_nii(root / "Case0001_Cine.nii.gz", cine)
            save_nii(root / "Case0001_gd.nii.gz", np.zeros((5, 7, 1), dtype=np.int16))

            dataset = MotionPairDataset(Path(tmp) / "CAREdatasets", num_frames=3, image_size=8)
            sample = dataset[0]

            self.assertEqual(tuple(sample["fixed"].shape), (1, 8, 8))
            self.assertEqual(tuple(sample["moving"].shape), (1, 8, 8))

    def test_learned_motion_channel_names_keep_per_frame_displacements(self) -> None:
        names = build_learned_motion_channel_names(num_frames=4)

        self.assertEqual(names, [
            "cine_t00",
            "cine_t01",
            "cine_t02",
            "cine_t03",
            "motion_t01_dx",
            "motion_t01_dy",
            "motion_t02_dx",
            "motion_t02_dy",
            "motion_t03_dx",
            "motion_t03_dy",
        ])


if __name__ == "__main__":
    unittest.main()
