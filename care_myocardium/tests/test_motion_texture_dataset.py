from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = REPO_ROOT / "care_myocardium" / "scripts" / "convert_cine_myops_motion_texture_to_nnunet.py"


def save_nii(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


class MotionTextureDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "CAREdatasets" / "Myo_train" / "CineMyoPS_train" / "center_alpha"
        self.source.mkdir(parents=True)

        cine = np.zeros((8, 8, 2, 4), dtype=np.float32)
        cine[2:6, 2:6, :, 0] = 10
        cine[2:6, 3:7, :, 1] = 14
        cine[3:7, 3:7, :, 2] = 4
        cine[1:5, 2:6, :, 3] = 18
        label = np.zeros((8, 8, 2), dtype=np.int16)
        label[2:4, 2:4, 0] = 200
        label[4:6, 2:4, 0] = 500
        label[3:5, 4:6, 0] = 2221

        save_nii(self.source / "Case0001_Cine.nii.gz", cine)
        save_nii(self.source / "Case0001_gd.nii.gz", label)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_converter(self, *extra_args: str, dataset_id: str = "608") -> Path:
        dataset_name = "CARE_CineMyoPS_MotionTexture"
        subprocess.run(
            [
                sys.executable,
                str(CONVERTER),
                "--data-root",
                str(self.root / "CAREdatasets"),
                "--dataset-root",
                str(self.root / "DATASET"),
                "--dataset-id",
                dataset_id,
                "--dataset-name",
                dataset_name,
                "--frame-index",
                "0",
                *extra_args,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return self.root / "DATASET" / "nnUNet_raw" / f"Dataset{int(dataset_id):03d}_{dataset_name}"

    def test_exports_all_cine_frames_plus_ed_referenced_motion_channels(self) -> None:
        dataset_dir = self.run_converter()

        dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(dataset_json["channel_names"]["0"], "cine_t00")
        self.assertEqual(dataset_json["channel_names"]["3"], "cine_t03")
        self.assertEqual(dataset_json["channel_names"]["4"], "temporal_std")
        self.assertEqual(dataset_json["channel_names"]["5"], "max_abs_diff_from_ed")
        self.assertEqual(dataset_json["channel_names"]["6"], "mean_abs_diff_from_ed")
        self.assertEqual(dataset_json["channel_names"]["7"], "signed_diff_at_max_abs_diff")
        self.assertEqual(dataset_json["channel_names"]["8"], "normalized_max_diff_frame")

        frame0 = np.asanyarray(nib.load(str(dataset_dir / "imagesTr" / "Case0001_0000.nii.gz")).dataobj)
        frame3 = np.asanyarray(nib.load(str(dataset_dir / "imagesTr" / "Case0001_0003.nii.gz")).dataobj)
        max_abs = np.asanyarray(nib.load(str(dataset_dir / "imagesTr" / "Case0001_0005.nii.gz")).dataobj)
        max_frame = np.asanyarray(nib.load(str(dataset_dir / "imagesTr" / "Case0001_0008.nii.gz")).dataobj)

        self.assertEqual(float(frame0[2, 2, 0]), 10.0)
        self.assertEqual(float(frame3[1, 2, 0]), 18.0)
        self.assertEqual(float(max_abs[2, 2, 0]), 10.0)
        self.assertAlmostEqual(float(max_frame[6, 3, 0]), 2.0 / 3.0, places=5)

        remapped = np.asanyarray(nib.load(str(dataset_dir / "labelsTr" / "Case0001.nii.gz")).dataobj)
        self.assertEqual(int(remapped[2, 2, 0]), 1)
        self.assertEqual(int(remapped[4, 2, 0]), 2)
        self.assertEqual(int(remapped[3, 4, 0]), 3)

        manifest = json.loads((dataset_dir / "care_motion_texture_manifest.json").read_text(encoding="utf-8"))
        case = manifest["cases"][0]
        self.assertEqual(case["num_cine_frames"], 4)
        self.assertEqual(case["num_channels"], 9)
        self.assertEqual(manifest["motion_feature_names"], [
            "temporal_std",
            "max_abs_diff_from_ed",
            "mean_abs_diff_from_ed",
            "signed_diff_at_max_abs_diff",
            "normalized_max_diff_frame",
        ])

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV/cv2 is not installed")
    def test_can_append_farneback_motion_summary_channels(self) -> None:
        dataset_dir = self.run_converter(
            "--flow-mode",
            "farneback_agg",
            "--flow-frame-stride",
            "1",
            dataset_id="609",
        )

        dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(dataset_json["channel_names"]["9"], "farneback_mean_dx_to_ed")
        self.assertEqual(dataset_json["channel_names"]["10"], "farneback_mean_dy_to_ed")
        self.assertEqual(dataset_json["channel_names"]["11"], "farneback_mean_magnitude_to_ed")
        self.assertEqual(dataset_json["channel_names"]["12"], "farneback_max_magnitude_to_ed")

        flow_mag = np.asanyarray(nib.load(str(dataset_dir / "imagesTr" / "Case0001_0011.nii.gz")).dataobj)
        self.assertEqual(flow_mag.shape, (8, 8, 2))

        manifest = json.loads((dataset_dir / "care_motion_texture_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["flow_mode"], "farneback_agg")
        self.assertEqual(manifest["cases"][0]["num_channels"], 13)


if __name__ == "__main__":
    unittest.main()
