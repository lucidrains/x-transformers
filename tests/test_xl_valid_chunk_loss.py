import unittest

import torch
import torch.nn.functional as F

from x_transformers import Decoder, TransformerWrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper


def make_model():
    torch.manual_seed(95)
    return TransformerWrapper(
        num_tokens=6, max_seq_len=3, max_mem_len=0,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )


def reference_loss(net, tokens):
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    numerator = torch.zeros(tokens.shape[0])
    denominator = (targets != 0).sum(-1).clamp_min(1)
    mems = None
    for chunk, labels in zip(inputs.split(3, -1), targets.split(3, -1)):
        logits, intermediates = net(chunk, mems=mems, return_mems=True, return_intermediates=True)
        mems = intermediates.mems
        numerator = numerator + F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=0, reduction="none"
        ).sum(-1)
    return (numerator / denominator).mean()


class TestXLValidChunkLoss(unittest.TestCase):
    def test_padding_and_empty_chunks_preserve_per_sequence_objective(self):
        model = make_model()
        wrapper = XLAutoregressiveWrapper(model, ignore_index=0)
        for tokens in (
            torch.tensor([[1, 2, 3, 4, 5, 0, 0], [2, 3, 4, 0, 0, 0, 0]]),
            torch.tensor([[1, 2, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]),
        ):
            with self.subTest(tokens=tokens.tolist()):
                torch.testing.assert_close(wrapper(tokens), reference_loss(model, tokens))

    def test_gradients_match_valid_target_average(self):
        model = make_model()
        wrapper = XLAutoregressiveWrapper(model, ignore_index=0)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [2, 3, 4, 0, 0, 0, 0]])
        param = model.to_logits.weight
        actual = torch.autograd.grad(wrapper(tokens), param)[0]
        expected = torch.autograd.grad(reference_loss(model, tokens), param)[0]
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
