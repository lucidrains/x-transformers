import pytest
import torch
from torch import nn

from x_transformers.xl_autoregressive_wrapper import LowRankLinear


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('bias', [True, False])
def test_low_rank_parameters_match_weight_dtype(dtype, bias):
    linear = nn.Linear(4, 6, bias = bias, dtype = dtype)
    low_rank = LowRankLinear(linear, low_rank_dim = 2)

    assert all(param.dtype == dtype for param in low_rank.parameters())
    output = low_rank(torch.ones(2, 4, dtype = dtype))
    assert output.shape == (2, 6)
    assert output.dtype == dtype
    output.sum().backward()
    assert all(param.grad is not None for param in low_rank.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason = 'CUDA is unavailable')
def test_low_rank_parameters_match_weight_device():
    linear = nn.Linear(4, 6, device = 'cuda')
    low_rank = LowRankLinear(linear, low_rank_dim = 2)

    assert all(param.device == linear.weight.device for param in low_rank.parameters())
    output = low_rank(torch.ones(2, 4, device = linear.weight.device))
    output.sum().backward()
    assert output.device == linear.weight.device
