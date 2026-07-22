from __future__ import annotations

import unittest

import numpy as np

from care_myocardium.scripts.package_care_myocardium_submission import center_restore_to_shape


class SubmissionPackagingTests(unittest.TestCase):
    def test_center_restore_embeds_cropped_prediction(self) -> None:
        arr = np.ones((4, 4, 2), dtype=np.int16)

        restored = center_restore_to_shape(arr, (6, 6, 2))

        self.assertEqual(restored.shape, (6, 6, 2))
        self.assertEqual(int(restored.sum()), 32)
        self.assertTrue(np.all(restored[1:5, 1:5, :] == 1))
        self.assertTrue(np.all(restored[0, :, :] == 0))

    def test_center_restore_crops_padded_prediction(self) -> None:
        arr = np.zeros((6, 6, 2), dtype=np.int16)
        arr[1:5, 1:5, :] = 2

        restored = center_restore_to_shape(arr, (4, 4, 2))

        self.assertEqual(restored.shape, (4, 4, 2))
        self.assertTrue(np.all(restored == 2))


if __name__ == "__main__":
    unittest.main()
