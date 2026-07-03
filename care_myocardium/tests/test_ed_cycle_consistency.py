from __future__ import annotations

import unittest

import torch

from care_myocardium.nnunet_ext.ed_cycle_consistency import (
    build_last_ed_reference_view,
    ed_cycle_consistency_loss,
)


class EDCycleConsistencyTests(unittest.TestCase):
    def test_last_ed_reference_view_reverses_cine_and_recomputes_motion_features(self) -> None:
        data = torch.zeros((1, 9, 1, 1, 1), dtype=torch.float32)
        data[:, 0, ...] = 10
        data[:, 1, ...] = 12
        data[:, 2, ...] = 4
        data[:, 3, ...] = 18

        reversed_view = build_last_ed_reference_view(data, num_frames=4)

        self.assertEqual(reversed_view.shape, data.shape)
        self.assertEqual(float(reversed_view[0, 0, 0, 0, 0]), 18.0)
        self.assertEqual(float(reversed_view[0, 1, 0, 0, 0]), 4.0)
        self.assertEqual(float(reversed_view[0, 2, 0, 0, 0]), 12.0)
        self.assertEqual(float(reversed_view[0, 3, 0, 0, 0]), 10.0)

        # Relative to the new ED reference of 18, the largest absolute change is
        # frame 1 with signed difference -14 and normalized index 1 / 3.
        self.assertAlmostEqual(float(reversed_view[0, 4, 0, 0, 0]), float(torch.tensor([18, 4, 12, 10]).float().std(unbiased=False)))
        self.assertEqual(float(reversed_view[0, 5, 0, 0, 0]), 14.0)
        self.assertEqual(float(reversed_view[0, 6, 0, 0, 0]), 7.0)
        self.assertEqual(float(reversed_view[0, 7, 0, 0, 0]), -14.0)
        self.assertAlmostEqual(float(reversed_view[0, 8, 0, 0, 0]), 1.0 / 3.0, places=5)

    def test_ed_cycle_loss_is_near_zero_for_matching_predictions(self) -> None:
        logits = torch.zeros((1, 4, 2, 2, 2), dtype=torch.float32)
        logits[:, 1, ...] = 8.0

        loss = ed_cycle_consistency_loss(logits, logits.clone(), confidence_threshold=0.5)

        self.assertLess(float(loss), 1e-3)

    def test_ed_cycle_loss_penalizes_first_and_last_ed_disagreement(self) -> None:
        first_logits = torch.zeros((1, 4, 2, 2, 2), dtype=torch.float32)
        last_logits = torch.zeros_like(first_logits)
        first_logits[:, 1, ...] = 8.0
        last_logits[:, 3, ...] = 8.0

        loss = ed_cycle_consistency_loss(first_logits, last_logits, confidence_threshold=0.5)

        self.assertGreater(float(loss), 5.0)


if __name__ == "__main__":
    unittest.main()
