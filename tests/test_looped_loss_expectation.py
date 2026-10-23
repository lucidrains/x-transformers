import unittest

import torch
import torch.nn.functional as F

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


def make_wrapper():
    torch.manual_seed(27)
    net = TransformerWrapper(
        num_tokens=7, max_seq_len=6, looped=True, max_looped_steps=4,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4, pre_and_post_norm=True),
    )
    return AutoregressiveWrapper(net, looped_exit_loss_weight=0.)


def reference_loss(wrapper, tokens, steps):
    inputs = tokens.masked_fill(tokens == -100, 0)
    _, cache = wrapper.net(
        inputs, return_intermediates=True, looped_pred_all_logits=True, looped_steps=steps,
    )
    targets = tokens[:, 1:]
    survival = torch.ones_like(targets, dtype=torch.float)
    expectation = torch.zeros_like(survival)
    for index, (logits, exit_logit) in enumerate(zip(cache.all_pred_logits, cache.exit_logits)):
        ce = F.cross_entropy(
            logits[:, :-1].transpose(1, 2), targets, ignore_index=-100, reduction="none"
        )
        exit_probability = exit_logit[:, :-1, 0].sigmoid().detach()
        probability = survival if index == steps - 1 else survival * exit_probability
        expectation = expectation + probability * ce
        survival = survival * (1 - exit_probability)
    return expectation[targets != -100].mean()


class TestLoopedLossExpectation(unittest.TestCase):
    def test_matches_stick_breaking_expected_token_loss(self):
        wrapper = make_wrapper()
        tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 1, -100, -100]])
        for steps in (2, 3, 4):
            with self.subTest(steps=steps):
                torch.testing.assert_close(wrapper(tokens, looped_steps=steps), reference_loss(wrapper, tokens, steps))

    def test_parameter_gradient_matches_expected_token_loss(self):
        wrapper = make_wrapper()
        tokens = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 1, -100, -100]])
        param = wrapper.net.to_logits.weight
        actual = torch.autograd.grad(wrapper(tokens, looped_steps=3), param)[0]
        expected = torch.autograd.grad(reference_loss(wrapper, tokens, 3), param)[0]
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
