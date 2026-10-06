import math

import pytest
import torch
from x_transformers import Decoder, TransformerWrapper
from x_transformers.entropy_based_tokenizer import (
    EntropyBasedTokenizer,
    calc_entropy_from_logits,
)


def scalar_entropy(values):
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(prob * math.log(prob) for prob in probabilities if prob > 0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("scale", [1.0, 100.0, "maximum"])
def test_entropy_of_finite_logits_matches_scalar_log_sum_exp_oracle(dtype, scale):
    if scale == "maximum":
        maximum = torch.finfo(dtype).max
        logits = torch.tensor([[maximum, maximum, -maximum, -maximum]], dtype=dtype)
    else:
        logits = (
            torch.tensor([[2.0, -1.0, 0.0, -3.0], [1.0, 1.0, -2.0, -2.0]], dtype=dtype)
            * scale
        )
    expected = torch.tensor(
        [scalar_entropy(row) for row in logits.tolist()], dtype=torch.float64
    )
    actual = calc_entropy_from_logits(logits)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("accumulate", [False, True])
@pytest.mark.parametrize("variable_lengths", [False, True])
@pytest.mark.parametrize("maximum_token_size", [None, 2])
def test_complete_entropy_tokenizer_preserves_confident_model_boundaries(
    dtype, accumulate, variable_lengths, maximum_token_size
):
    torch.manual_seed(157)
    decoder = TransformerWrapper(
        num_tokens=7,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    ).to(dtype=dtype)
    # Two equal dominant logits or five equal dominant logits both have entropy
    # above the threshold. Confident tails stress underflow with finite logits.
    with torch.no_grad():
        decoder.to_logits.weight.zero_()
        decoder.to_logits.weight[:2, 0] = 8192
    tokenizer = EntropyBasedTokenizer(
        decoder,
        entropy_threshold=0.4,
        accumulate_entropy=accumulate,
        max_token_size=maximum_token_size,
    )
    seq = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 1]])
    lens = torch.tensor([6, 4]) if variable_lengths else None
    lengths = [6, 4] if variable_lengths else [6, 6]
    logits = decoder(seq)
    assert torch.isfinite(logits).all()
    for row, length in zip(logits.tolist(), lengths):
        assert all(scalar_entropy(values) >= 0.4 for values in row[:length])
    expected_lengths = torch.zeros((2, 6), dtype=torch.long)
    for row, length in enumerate(lengths):
        expected_lengths[row, :length] = 1
    actual = tokenizer(seq, lens=lens)
    assert torch.equal(actual, expected_lengths)
    segmented = tokenizer(seq, lens=lens, return_segmented_seq=True)
    for original, segments, length in zip(seq, segmented, lengths):
        assert len(segments) == length
        assert all(segment.numel() == 1 for segment in segments)
        assert torch.equal(torch.cat(segments), original[:length])
