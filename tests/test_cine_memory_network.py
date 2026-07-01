import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_myocardium.nnunet_ext.cine_memory_network import CineTemporalMemoryStem


class CineTemporalMemoryStemTests(unittest.TestCase):
    def test_memory_frames_influence_enhanced_ed_output(self):
        torch.manual_seed(0)
        stem = CineTemporalMemoryStem(
            num_frames=4,
            embed_dim=6,
            query_frame_index=0,
            residual_scale_init=1.0,
        )
        stem.eval()

        x = torch.zeros(2, 4, 5, 6, 4)
        x[:, 0] = 1.0
        y_without_motion = stem(x)

        x[:, 1] = 3.0
        y_with_motion = stem(x)

        self.assertEqual(tuple(y_with_motion.shape), (2, 1, 5, 6, 4))
        self.assertGreater(
            (y_with_motion - y_without_motion).abs().sum().item(),
            0.0,
        )

    def test_single_frame_input_bypasses_memory(self):
        stem = CineTemporalMemoryStem(num_frames=1, embed_dim=4, query_frame_index=0)
        x = torch.randn(1, 1, 3, 4, 5)

        y = stem(x)

        self.assertTrue(torch.equal(x, y))


if __name__ == "__main__":
    unittest.main()
