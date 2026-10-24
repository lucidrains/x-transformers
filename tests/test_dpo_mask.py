import torch
from x_transformers.dpo import masked_mean


def test_a_pad_is_not_scored_as_the_answer():
    # prompt, answer, pad. The answer logprob is -1 and the pad logprob is -4.
    log_probs = torch.tensor([[-1.0, -4.0]])
    mask = torch.tensor([[False, True, False]])
    got = masked_mean(log_probs, mask)
    assert got.tolist() == [-1.0]

    # A mask that is already aligned with the logprobs stays put.
    aligned = torch.tensor([[True, False]])
    assert masked_mean(log_probs, aligned).tolist() == [-1.0]
