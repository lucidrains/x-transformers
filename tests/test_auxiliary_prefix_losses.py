from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


def reference_loss(net, tokens, prefix, *, looped, continuous):
    inputs = tokens.masked_fill(tokens == -100, 0)
    target = tokens[:, 1:]
    valid = target != -100
    prefix_len = 0 if prefix is None else prefix.shape[1]
    output, cache = net(
        inputs, prepend_embeds = prefix, return_intermediates = True,
        return_next_embed_pred = continuous, looped_pred_all_logits = True,
        **({'looped_steps': 3} if looped else {})
    )
    logits, (predicted, initial) = output if continuous else (output, (None, None))

    if looped:
        per_loop_losses = torch.stack([
            F.cross_entropy(value[:, prefix_len:-1].transpose(1, 2), target, reduction = 'none')
            for value in cache.all_pred_logits
        ], dim = 1)
        exits = torch.stack([value[:, prefix_len:-1, 0] for value in cache.exit_logits], dim = 1)
        hazards = exits.detach().sigmoid()
        survival = torch.ones_like(hazards[:, 0])
        weighted = []
        for step in range(hazards.shape[1]):
            weight = survival if step == hazards.shape[1] - 1 else survival * hazards[:, step]
            weighted.append(per_loop_losses[:, step] * weight)
            survival = survival * (1 - hazards[:, step])
        loss = torch.stack(weighted, dim = 1)[valid[:, None, :].expand_as(per_loop_losses)].mean()
        exit_targets = (50 * (per_loop_losses[:, 1:] - per_loop_losses[:, :-1] + 0.05)).sigmoid().detach()
        exit_losses = F.binary_cross_entropy_with_logits(exits[:, :-1], exit_targets, reduction = 'none')
        loss = loss + exit_losses[valid[:, None, :].expand_as(exit_losses)].mean()
    else:
        loss = F.cross_entropy(logits[:, prefix_len:-1].transpose(1, 2), target)

    if continuous:
        differences = (predicted[:, prefix_len:-1] - initial[:, prefix_len + 1:].detach()).abs()
        loss = loss + 0.1 * differences[valid].mean()
    return loss


@pytest.mark.parametrize('looped', (False, True))
@pytest.mark.parametrize('continuous', (False, True))
@pytest.mark.parametrize('prefix_len', (0, 2))
@pytest.mark.parametrize('padding', (False, True))
def test_auxiliary_losses_align_with_text_after_prefix(looped, continuous, prefix_len, padding):
    torch.manual_seed(0)
    net = TransformerWrapper(
        num_tokens = 11, max_seq_len = 16, add_continuous_pred_head = continuous,
        looped = looped, max_looped_steps = 3,
        attn_layers = Decoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 8, pre_and_post_norm = True)
    )
    expected_net = deepcopy(net)
    wrapper = AutoregressiveWrapper(net)
    tokens = torch.randint(0, 11, (2, 5))
    if padding:
        tokens[0, -1] = -100
    prefix = torch.randn(2, prefix_len, 16, requires_grad = True) if prefix_len else None
    expected_prefix = prefix.detach().clone().requires_grad_() if prefix_len else None

    actual = wrapper(tokens, prepend_embeds = prefix, **({'looped_steps': 3} if looped else {}))
    expected = reference_loss(expected_net, tokens, expected_prefix, looped = looped, continuous = continuous)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for parameter, expected_parameter in zip(net.parameters(), expected_net.parameters()):
        if expected_parameter.grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol = 1e-4, atol = 1e-5)
    if prefix_len:
        assert prefix.grad.abs().sum() > 0
        torch.testing.assert_close(prefix.grad, expected_prefix.grad, rtol = 1e-4, atol = 1e-5)
