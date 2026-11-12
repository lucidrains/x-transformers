import pytest
import torch
import torch.nn.functional as F
from torch import nn

from x_transformers import Decoder, TransformerWrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper, newtonschulz5


def normalized_reference(gradient, steps=5):
    transposed = gradient.shape[-2] > gradient.shape[-1]
    values = gradient.transpose(-1, -2) if transposed else gradient
    shape = values.shape
    values = values.reshape(-1, *shape[-2:])
    # Compute the mathematical normalization in double, then retain the native iteration dtype.
    precise = values.double()
    values = (precise / torch.linalg.vector_norm(precise, dim=(-1, -2), keepdim=True).clamp(min=1e-7)).to(gradient.dtype)
    for _ in range(steps):
        gram = values @ values.transpose(-1, -2)
        values = 3.4445 * values + (-4.7750 * gram + 2.0315 * gram @ gram) @ values
    values = values.reshape(shape)
    return values.transpose(-1, -2) if transposed else values


@pytest.mark.parametrize("shape", [(2, 8, 16), (2, 16, 8), (2, 3, 8, 8)])
@pytest.mark.parametrize("steps", [0, 5])
@pytest.mark.parametrize("scale", [1, 20000])
def test_finite_half_gradients_keep_nonzero_muon_updates(shape, steps, scale):
    torch.manual_seed(16)
    gradient = (torch.randn(shape).clamp(-2, 2) * scale).half()
    original = gradient.clone()
    assert torch.isfinite(gradient).all()
    if scale == 20000:
        assert gradient.norm(dim=(-1, -2)).isinf().all()

    actual, transformed = newtonschulz5(gradient, steps=steps)
    expected = normalized_reference(gradient, steps)

    assert transformed
    assert actual.dtype == torch.float16
    assert actual.shape == gradient.shape
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(gradient, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_other_dtypes_preserve_existing_normalization(dtype):
    torch.manual_seed(16)
    gradient = torch.randn(2, 8, 16, dtype=dtype)
    values = gradient / gradient.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-7)
    for _ in range(5):
        gram = values @ values.transpose(-1, -2)
        values = 3.4445 * values + (-4.7750 * gram + 2.0315 * gram @ gram) @ values
    actual, transformed = newtonschulz5(gradient)
    assert transformed
    torch.testing.assert_close(actual, values, rtol=0, atol=0)


def test_zero_half_gradient_stays_zero():
    actual, transformed = newtonschulz5(torch.zeros(2, 8, 16, dtype=torch.float16))
    assert transformed
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("shape", [(8,), (2, 8)])
def test_nonmatrix_batch_uses_original_sgd_gradient(shape):
    gradient = torch.ones(shape, dtype=torch.float16)
    actual, transformed = newtonschulz5(gradient)
    assert not transformed
    assert actual is gradient


class ScaledNextTokenLoss(nn.Module):
    def __init__(self, head, labels):
        super().__init__()
        self.head = head
        self.register_buffer("labels", labels)

    def forward(self, intermediates, mask):
        logits = self.head(intermediates.last_layer_hiddens).float()
        return F.cross_entropy(logits.transpose(1, 2), self.labels, reduction="none") * 60000


@pytest.mark.parametrize("create_graph", [False, True])
def test_native_half_ttt_keeps_the_scaled_gradient_update(create_graph):
    torch.manual_seed(16)
    path = "attn_layers.layers.0.1.to_v"
    model = TransformerWrapper(
        num_tokens=16, max_seq_len=4, max_mem_len=4,
        attn_layers=Decoder(dim=32, depth=1, heads=2, attn_dim_head=16, use_rmsnorm=True),
    ).half()
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    labels = tokens + 1
    objective = ScaledNextTokenLoss(model.to_logits, labels)
    wrapper = XLAutoregressiveWrapper(
        model, tbptt_steps=100, ttt_module_paths=(path,), ttt_use_muon=True,
        ttt_muon_lr=0.1, ttt_wd=0.0, ttt_custom_loss_module=objective,
    )
    wrapper.init_ttt(2)
    logits, intermediates = wrapper.net(tokens, return_intermediates=True)
    parameter = wrapper.ttt_wrappers[path].batch_params["weight"]
    loss = objective(intermediates, labels != -100).mean(-1).sum()
    gradient, = torch.autograd.grad(loss, (parameter,), retain_graph=True)
    assert torch.isfinite(gradient).all()
    assert gradient.norm(dim=(-1, -2)).isinf().all()
    expected = parameter.detach() - 0.1 * normalized_reference(gradient)

    wrapper.update_ttt(logits, labels, create_graph=create_graph, intermediates=intermediates)

    actual = wrapper.ttt_wrappers[path].batch_params["weight"]
    assert torch.count_nonzero(actual.detach() - parameter.detach()) > 0
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=2e-2)
    after = wrapper.net(tokens)
    assert torch.isfinite(after).all()
    F.cross_entropy(after.float().transpose(1, 2), labels).backward()
    for value in wrapper.parameters():
        if value.grad is not None:
            assert torch.isfinite(value.grad).all()
