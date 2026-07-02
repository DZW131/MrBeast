from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "care_myocardium" / "scripts" / "generate_scar_roi_dataset.py"
RESTORE = REPO_ROOT / "care_myocardium" / "scripts" / "restore_roi_predictions_to_full.py"


def save_nii(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class Stage3ROIPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ed_dir = self.root / "nnUNet_raw" / "Dataset601_CARE_CineMyoPS_ED"
        images = self.ed_dir / "imagesTr"
        labels = self.ed_dir / "labelsTr"
        pred_dir = self.root / "stage1_pred"
        myo_dir = self.root / "stage2_myo_full"
        for path in (images, labels, pred_dir, myo_dir):
            path.mkdir(parents=True, exist_ok=True)

        shape = (16, 16, 3)
        image = np.zeros(shape, dtype=np.float32)
        label = np.zeros(shape, dtype=np.int16)
        label[4:12, 4:12, 1] = 1
        label[7:9, 7:9, 1] = 3
        stage1 = np.zeros(shape, dtype=np.int16)
        stage1[6:10, 6:10, 1] = 3
        myo = np.zeros(shape, dtype=np.int16)
        myo[3:13, 3:13, 1] = 1

        save_nii(images / "Case0001_0000.nii.gz", image)
        save_nii(labels / "Case0001.nii.gz", label)
        save_nii(pred_dir / "Case0001.nii.gz", stage1)
        save_nii(myo_dir / "Case0001.nii.gz", myo)
        self.pred_dir = pred_dir
        self.myo_dir = myo_dir

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_scar_stage3_uses_stage2_myo_prior_channel(self) -> None:
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--target",
                "scar",
                "--dataset-root",
                str(self.root),
                "--stage1-pred-dir",
                str(self.pred_dir),
                "--stage2-myo-pred-dir",
                str(self.myo_dir),
                "--output-dataset-id",
                "605",
                "--output-dataset-name",
                "CARE_CineMyoPS_ScarMyoROI_ED",
                "--min-xy",
                "12",
                "--margin-xy",
                "1",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        out_dir = self.root / "nnUNet_raw" / "Dataset605_CARE_CineMyoPS_ScarMyoROI_ED"
        dataset_json = json.loads((out_dir / "dataset.json").read_text())
        self.assertEqual(dataset_json["channel_names"]["2"], "stage2_myo_prior")

        myo_channel = np.asanyarray(nib.load(str(out_dir / "imagesTr" / "Case0001_0002.nii.gz")).dataobj)
        label = np.asanyarray(nib.load(str(out_dir / "labelsTr" / "Case0001.nii.gz")).dataobj)
        self.assertEqual(int(myo_channel.sum()), 100)
        self.assertEqual(int(label.sum()), 4)

        manifest = json.loads((out_dir / "roi_manifest.json").read_text())
        case = manifest["cases"][0]
        self.assertIn("stage2_myo_prediction", case)
        self.assertEqual(case["stage2_myo_voxels_in_crop"], 100)

    def test_scar_stage3_can_use_dilated_stage1_proposal(self) -> None:
        subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--target",
                "scar",
                "--dataset-root",
                str(self.root),
                "--stage1-pred-dir",
                str(self.pred_dir),
                "--stage2-myo-pred-dir",
                str(self.myo_dir),
                "--target-prior-mode",
                "dilate_xy",
                "--prior-dilation-xy",
                "1",
                "--output-dataset-id",
                "606",
                "--output-dataset-name",
                "CARE_CineMyoPS_ScarMyoDilatedROI_ED",
                "--min-xy",
                "12",
                "--margin-xy",
                "1",
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        out_dir = self.root / "nnUNet_raw" / "Dataset606_CARE_CineMyoPS_ScarMyoDilatedROI_ED"
        dataset_json = json.loads((out_dir / "dataset.json").read_text())
        self.assertEqual(dataset_json["channel_names"]["1"], "stage1_scar_dilated_prior")
        proposal = np.asanyarray(nib.load(str(out_dir / "imagesTr" / "Case0001_0001.nii.gz")).dataobj)
        self.assertEqual(int(proposal.sum()), 36)

    def test_restore_roi_prediction_pastes_crop_to_full_image(self) -> None:
        crop_dataset = self.root / "nnUNet_raw" / "Dataset604_CARE_CineMyoPS_MyoROI_ED"
        crop_images = crop_dataset / "imagesTr"
        crop_images.mkdir(parents=True)
        save_nii(crop_images / "Case0001_0000.nii.gz", np.zeros((4, 5, 2), dtype=np.float32))
        manifest = {
            "cases": [
                {
                    "case_id": "Case0001",
                    "crop_box_xyz": [[3, 7], [4, 9], [1, 3]],
                    "source_shape": [16, 16, 3],
                }
            ]
        }
        (crop_dataset / "roi_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        pred_dir = self.root / "crop_predictions"
        crop_pred = np.zeros((4, 5, 2), dtype=np.int16)
        crop_pred[1:3, 2:4, 0] = 1
        save_nii(pred_dir / "Case0001.nii.gz", crop_pred)
        out_dir = self.root / "full_predictions"

        subprocess.run(
            [
                sys.executable,
                str(RESTORE),
                "--roi-dataset-dir",
                str(crop_dataset),
                "--crop-pred-dir",
                str(pred_dir),
                "--reference-dataset-dir",
                str(self.ed_dir),
                "--output-dir",
                str(out_dir),
                "--overwrite",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )

        full = np.asanyarray(nib.load(str(out_dir / "Case0001.nii.gz")).dataobj)
        self.assertEqual(full.shape, (16, 16, 3))
        self.assertEqual(int(full.sum()), 4)
        self.assertEqual(int(full[4:6, 6:8, 1].sum()), 4)


if __name__ == "__main__":
    unittest.main()
