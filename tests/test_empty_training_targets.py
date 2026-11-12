from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


def make_wrapper(looped, mixture, continuous):
    model = TransformerWrapper(
        num_tokens=16, max_seq_len=8, looped=looped,
        mixture_of_softmax=mixture, add_continuous_pred_head=continuous,
        attn_layers=Decoder(dim=16, depth=1, heads=2, attn_dim_head=8, pre_and_post_norm=looped),
    )
    return AutoregressiveWrapper(model, next_embed_loss_weight=0.2)


@pytest.mark.parametrize("looped", [False, True])
@pytest.mark.parametrize("mixture,continuous", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("case", ["ignored", "lens", "single"])
def test_no_supervised_targets_has_zero_differentiable_loss(looped, mixture, continuous, case):
    torch.manual_seed(16)
    wrapper = make_wrapper(looped, mixture, continuous)
    inputs = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    kwargs = {}
    if case == "ignored":
        inputs[:, 1:] = -100
    elif case == "lens":
        kwargs["lens"] = torch.tensor([1, 1])
    else:
        inputs = inputs[:, :1]

    loss = wrapper(inputs, **kwargs)

    assert loss.requires_grad
    assert torch.isfinite(loss)
    assert loss.item() == 0
    loss.backward()
    gradients = [p.grad for p in wrapper.parameters() if p.grad is not None]
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("mixture,continuous", [(False, False), (False, True), (True, False)])
def test_nonempty_targets_keep_torch_loss_and_gradients(mixture, continuous):
    torch.manual_seed(16)
    wrapper = make_wrapper(False, mixture, continuous)
    reference = deepcopy(wrapper.net)
    inputs = torch.tensor([[1, 2, 3, 4, -100], [6, 7, 8, -100, -100]])
    target = inputs[:, 1:]
    mask = target != -100
    output = reference(inputs.masked_fill(inputs == -100, 0), return_next_embed_pred=continuous)
    if continuous:
        logits, (predicted_embeds, initial_embeds) = output
    else:
        logits = output
    loss_fn = F.nll_loss if mixture else F.cross_entropy
    expected = loss_fn(logits[:, :-1].transpose(1, 2), target, ignore_index=-100)
    if continuous:
        expected = expected + 0.2 * F.l1_loss(
            predicted_embeds[:, :-1][mask], initial_embeds[:, 1:].detach()[mask]
        )

    actual = wrapper(inputs)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    for parameter, reference_parameter in zip(wrapper.net.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)


def test_half_precision_continuous_loss_keeps_stable_nonempty_mean():
    torch.manual_seed(16)
    model = TransformerWrapper(
        num_tokens=16, max_seq_len=65, add_continuous_pred_head=True,
        attn_layers=Decoder(dim=32, depth=1, heads=2, attn_dim_head=16),
    ).half()
    wrapper = AutoregressiveWrapper(model, next_embed_loss_weight=0.2)
    reference = deepcopy(model)
    inputs = torch.randint(0, 16, (128, 65))
    logits, (predicted, initial) = reference(inputs, return_next_embed_pred=True)
    element_losses = F.l1_loss(predicted[:, :-1], initial[:, 1:].detach(), reduction="none")
    assert element_losses.sum().isinf()
    expected = F.cross_entropy(logits[:, :-1].transpose(1, 2), inputs[:, 1:]) + 0.2 * element_losses.mean()

    actual = wrapper(inputs)

    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    actual.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
