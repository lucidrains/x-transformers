import pytest
import torch
from x_transformers import Decoder
from x_transformers.continuous import (
    ContinuousAutoregressiveWrapper,
    ContinuousTransformerWrapper,
)


@pytest.mark.parametrize("loss_kind", ["mse", "l1", "gaussian"])
@pytest.mark.parametrize("mask_kind", ["lens", "full_mask", "all_valid", "none"])
def test_continuous_loss_masks_next_token_targets_and_padding_gradients(
    loss_kind, mask_kind
):
    torch.manual_seed(67)
    net = ContinuousTransformerWrapper(
        max_seq_len=7,
        dim_in=2,
        dim_out=2,
        probabilistic=loss_kind == "gaussian",
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )
    wrapper = ContinuousAutoregressiveWrapper(net, use_l1_loss=loss_kind == "l1")
    inputs = torch.randn(2, 7, 2, requires_grad=True)
    lens = torch.tensor([4, 5])
    mask = torch.arange(7)[None, :] < lens[:, None]
    if mask_kind == "all_valid":
        mask = torch.ones_like(mask)
    if mask_kind == "lens":
        kwargs = {"lens": lens}
    elif mask_kind == "none":
        kwargs = {}
        mask = None
    else:
        kwargs = {"mask": mask}
    actual = wrapper(inputs, **kwargs)
    input_mask = mask[:, :-1] if mask is not None else None
    target_mask = mask[:, 1:] if mask is not None else None
    predicted = net(inputs[:, :-1], mask=input_mask)
    targets = inputs[:, 1:]
    if loss_kind == "mse":
        elementwise = (predicted - targets).square()
    elif loss_kind == "l1":
        elementwise = (predicted - targets).abs()
    else:
        mean, variance = predicted
        elementwise = 0.5 * (variance.log() + (mean - targets).square() / variance)
    expected = (
        elementwise[target_mask].mean()
        if target_mask is not None
        else elementwise.mean()
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, inputs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    if mask_kind in ("lens", "full_mask"):
        assert torch.equal(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))
        changed = inputs.detach().clone()
        changed[~mask] += 9
        torch.testing.assert_close(wrapper(changed, **kwargs), actual.detach())
    actual.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )
