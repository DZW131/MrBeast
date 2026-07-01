import unittest
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from care_myocardium.nnunet_ext.cine_memory_network import (
    CineMemorySegmentationNetwork,
    CineTemporalMemoryStem,
)


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

    def test_segmentation_wrapper_exposes_nnunet_backbone_contract(self):
        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.deep_supervision = True

        class BaseNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Identity()
                self.decoder = Decoder()

            def forward(self, x):
                return x

            def compute_conv_feature_map_size(self, input_size):
                return tuple(input_size)

        base = BaseNetwork()
        network = CineMemorySegmentationNetwork(base, num_input_channels=3, memory_embed_dim=2)

        network.decoder.deep_supervision = False

        self.assertIs(network.decoder, base.decoder)
        self.assertIs(network.encoder, base.encoder)
        self.assertFalse(base.decoder.deep_supervision)
        self.assertEqual(network.compute_conv_feature_map_size((4, 5, 6)), (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
