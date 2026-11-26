import pytest
import torch

from x_transformers import Decoder, TransformerWrapper
from x_transformers.discover_wrapper import DiscoverDecoder


def make_decoder(continuation):
    net = TransformerWrapper(
        num_tokens = 8,
        max_seq_len = 8,
        use_abs_pos_emb = False,
        attn_layers = Decoder(dim = 8, depth = 1, heads = 2)
    )
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.token_emb.emb.weight.copy_(torch.eye(8))
        tokens = [6, *continuation, continuation[-1]]
        for current, following in zip(tokens, tokens[1:]):
            net.to_logits.weight[following, current] = 1.

    return DiscoverDecoder(
        decoder = net,
        num_fillers = 6,
        bos_id = 6,
        pad_id = 7,
        max_seq_len = 6
    )


@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('max_len', (None, 4))
@pytest.mark.parametrize('continuation', ((0,), (1, 0), (1, 2, 0), (1,)))
def test_discover_decode_pads_early_eos(batch_size, max_len, continuation):
    decoder = make_decoder(continuation)
    output = decoder.decode(
        torch.zeros(batch_size, 8),
        eos_id = 0,
        max_len = max_len
    )

    total_len = 6 if max_len is None else max_len
    if continuation[-1] == 0:
        expected_row = [*continuation, *([7] * (total_len - len(continuation)))]
    else:
        expected_row = [1] * total_len

    assert torch.equal(output, torch.tensor([expected_row] * batch_size))


def test_discover_decode_preserves_explicit_lengths():
    decoder = make_decoder((1,))
    output = decoder.decode(torch.zeros(2, 8), lengths = torch.tensor([2, 4]))

    assert torch.equal(output, torch.tensor([[1, 1, 7, 7], [1, 1, 1, 1]]))
