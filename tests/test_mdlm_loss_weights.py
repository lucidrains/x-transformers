import unittest

import torch

from x_transformers import Encoder, TransformerWrapper
from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper


def make_wrapper(schedule):
    net = TransformerWrapper(
        num_tokens=7, max_seq_len=5,
        attn_layers=Encoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )
    return NonAutoregressiveWrapper(
        net, mask_id=6, schedule=schedule,
        no_replace_prob=0., random_token_prob=0.,
    )


class TestMDLMLossWeights(unittest.TestCase):
    def test_schedule_weights_are_negative_log_likelihood_weights(self):
        times = torch.tensor([0.1, 0.25, 0.7, 0.9])
        for schedule in ("linear", "cosine"):
            with self.subTest(schedule=schedule):
                model = make_wrapper(schedule)
                if schedule == "linear":
                    expected = times.reciprocal()
                else:
                    expected = (torch.pi / 2) * torch.sin(times * torch.pi / 2) / (
                        1 - torch.cos(times * torch.pi / 2)
                    )
                torch.testing.assert_close(model.loss_weight_fn(times), expected)
                self.assertTrue((model.loss_weight_fn(times) > 0).all())

    def test_native_generator_objective_is_positive_and_differentiable(self):
        torch.manual_seed(120)
        model = make_wrapper("linear")
        tokens = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        loss = model(tokens, only_train_generator=True).loss
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
