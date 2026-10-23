import unittest

import torch
import torch.nn.functional as F

from x_transformers import Decoder
from x_transformers.continuous import (
    ContinuousAutoregressiveWrapper, ContinuousTransformerWrapper, batch_masked_mean,
)


class TestContinuousBatchMean(unittest.TestCase):
    def test_feature_replication_does_not_rescale_mean(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        data = torch.tensor([[[1.], [3.], [100.]], [[5.], [100.], [100.]]])
        expected = torch.tensor([2., 5.])
        for dim in (1, 2, 7):
            with self.subTest(dim=dim):
                torch.testing.assert_close(batch_masked_mean(data.expand(-1, -1, dim), mask), expected)

    def test_gradient_and_empty_example(self):
        values = torch.arange(24.).reshape(2, 3, 4).requires_grad_()
        mask = torch.tensor([[True, False, True], [False, False, False]])
        actual = batch_masked_mean(values, mask)
        torch.testing.assert_close(actual, torch.stack((values[0, [0, 2]].mean(), values[1].sum() * 0)))
        grad = torch.autograd.grad(actual.sum(), values)[0]
        expected = torch.zeros_like(values)
        expected[0, [0, 2]] = 1 / 8
        torch.testing.assert_close(grad, expected)

    def test_native_wrapper_matches_per_example_element_mean(self):
        torch.manual_seed(15)
        net = ContinuousTransformerWrapper(
            max_seq_len=5, dim_in=3, dim_out=3,
            attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
        )
        wrapper = ContinuousAutoregressiveWrapper(net, equal_loss_weight_batch=True)
        x = torch.randn(2, 5, 3)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        predicted = net(x[:, :-1], mask=mask)
        pointwise = F.mse_loss(predicted, x[:, 1:], reduction="none")
        expected = torch.stack([pointwise[i, mask[i]].mean() for i in range(2)]).mean()
        torch.testing.assert_close(wrapper(x, mask=mask), expected)


if __name__ == "__main__":
    unittest.main()
