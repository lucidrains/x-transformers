from copy import deepcopy

import pytest
import torch

from x_transformers import ContinuousTransformerWrapper, Encoder


@pytest.mark.parametrize('length_input', ('lens', 'mask', 'none'))
@pytest.mark.parametrize('memory_tokens', (0, 2))
@pytest.mark.parametrize('return_embeddings', (False, True))
def test_continuous_pool_uses_valid_sequence_lengths(length_input, memory_tokens, return_embeddings):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3,
        dim_out = 5,
        max_seq_len = 8,
        num_memory_tokens = memory_tokens,
        average_pool_embed = True,
        attn_layers = Encoder(dim = 8, depth = 1, heads = 2, attn_dim_head = 4)
    ).eval()
    unpooled = deepcopy(model)
    unpooled.average_pool_embed = False
    inputs = torch.randn(2, 5, 3, requires_grad = True)
    lengths = torch.tensor([2, 4]) if length_input != 'none' else torch.tensor([5, 5])
    mask = torch.arange(5)[None, :] < lengths[:, None]
    kwargs = {length_input: lengths if length_input == 'lens' else mask} if length_input != 'none' else {}

    actual = model(inputs, return_embeddings = return_embeddings, **kwargs)
    expected = torch.stack([
        unpooled(inputs[row:row + 1, :length], return_embeddings = return_embeddings).mean(dim = 1)[0]
        for row, length in enumerate(lengths)
    ])
    torch.testing.assert_close(actual, expected, rtol = 1e-4, atol = 1e-5)

    actual_grad, = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grad, = torch.autograd.grad(expected.square().sum(), inputs)
    torch.testing.assert_close(actual_grad, expected_grad, rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))


def test_continuous_pool_empty_length_returns_zero():
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3,
        dim_out = 5,
        max_seq_len = 8,
        average_pool_embed = True,
        attn_layers = Encoder(dim = 8, depth = 1, heads = 2, attn_dim_head = 4)
    )
    inputs = torch.randn(2, 5, 3, requires_grad = True)
    result = model(inputs, lens = torch.tensor([0, 4]))
    torch.testing.assert_close(result[0], torch.zeros_like(result[0]))
    result.sum().backward()
    torch.testing.assert_close(inputs.grad[0], torch.zeros_like(inputs.grad[0]))
