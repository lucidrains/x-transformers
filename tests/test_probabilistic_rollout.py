import unittest

import torch
from torch import nn

from x_transformers import Decoder
from x_transformers.continuous import ContinuousAutoregressiveWrapper, ContinuousTransformerWrapper


class RecordingGaussianNLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.prediction_shapes = None

    def forward(self, predicted, target):
        mean, variance = predicted
        self.prediction_shapes = (mean.shape, variance.shape, target.shape)
        if mean.shape != target.shape or variance.shape != target.shape:
            raise AssertionError("rollout must preserve batch and concatenate time")
        return nn.functional.gaussian_nll_loss(mean, target, variance, reduction="none")


class TestProbabilisticRollout(unittest.TestCase):
    def test_gaussian_rollout_preserves_batch_and_time(self):
        for batch, steps in ((2, 2), (3, 3)):
            with self.subTest(batch=batch, steps=steps):
                torch.manual_seed(42)
                net = ContinuousTransformerWrapper(
                    max_seq_len=9, dim_in=2, dim_out=2, probabilistic=True,
                    attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
                )
                loss_fn = RecordingGaussianNLL()
                wrapper = ContinuousAutoregressiveWrapper(net, loss_fn=loss_fn)
                loss = wrapper(torch.randn(batch, 9, 2), rollout_steps=steps)
                expected_shape = torch.Size([batch, steps, 2])
                self.assertEqual(loss_fn.prediction_shapes, (expected_shape,) * 3)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_deterministic_rollout_shape_is_preserved(self):
        net = ContinuousTransformerWrapper(
            max_seq_len=9, dim_in=2, dim_out=2,
            attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
        )
        wrapper = ContinuousAutoregressiveWrapper(net)
        loss = wrapper(torch.randn(2, 9, 2), rollout_steps=2)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
