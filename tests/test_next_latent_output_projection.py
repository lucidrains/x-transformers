from copy import deepcopy

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from x_transformers import Decoder, TransformerWrapper
from x_transformers.next_latent_wrapper import NextLatentWrapper


class NonlinearProjection(nn.Linear):
    def forward(self, inputs):
        return super().forward(inputs).tanh()


def make_wrapper(head_kind, rollouts, *, dim = 16, num_tokens = 13):
    custom_head = None
    if head_kind == 'mlp':
        custom_head = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, num_tokens))
    elif head_kind == 'biased':
        custom_head = nn.Linear(dim, num_tokens)
    elif head_kind == 'subclass':
        custom_head = NonlinearProjection(dim, num_tokens)
    net = TransformerWrapper(
        num_tokens = num_tokens, max_seq_len = 12,
        tie_embedding = head_kind == 'tied', tie_embedding_scale = 0.37,
        to_logits = custom_head,
        attn_layers = Decoder(dim = dim, depth = 1, heads = 2, attn_dim_head = 8)
    )
    return NextLatentWrapper(
        net, dim = dim, num_rollouts = rollouts,
        next_latent_loss_weight = 0., kl_loss_weight = 1.,
        dynamic_rollout_loss_weight = False
    )


def reference_kl(wrapper, tokens, head_kind):
    inputs = tokens.masked_fill(tokens == -100, 0)
    hidden, _ = wrapper.net(inputs, return_embeddings_and_intermediates = True)
    embeddings = wrapper.net.token_emb(inputs).detach()
    if head_kind == 'tied':
        frozen_head = nn.Linear(hidden.shape[-1], wrapper.net.num_tokens, bias = False)
        with torch.no_grad():
            frozen_head.weight.copy_(wrapper.net.token_emb.emb.weight * wrapper.net.tie_embedding_scale)
    else:
        frozen_head = deepcopy(wrapper.net.to_logits)
    frozen_head.requires_grad_(False)

    latent = hidden[:, :-1]
    loss = hidden.new_zeros(())
    for step in range(1, wrapper.num_rollouts + 1):
        future_embeds = F.pad(embeddings[:, step:], (0, 0, 0, step - 1))
        latent = latent + wrapper.dynamics_model.net(torch.cat((latent, future_embeds), dim = -1))
        prediction = latent[:, :tokens.shape[1] - step]
        target = hidden[:, step:].detach()
        probability = frozen_head(target).softmax(dim = -1)
        log_probability = frozen_head(prediction).log_softmax(dim = -1)
        divergence = F.kl_div(log_probability, probability, reduction = 'none').sum(dim = -1)
        valid = tokens[:, step:] != -100
        loss = loss + divergence[valid].mean() / wrapper.num_rollouts
    return loss


@pytest.mark.parametrize('head_kind', ('linear', 'tied', 'mlp', 'biased', 'subclass'))
@pytest.mark.parametrize('rollouts', (1, 3))
@pytest.mark.parametrize('padding', (False, True))
def test_next_latent_kl_preserves_hidden_gradients_and_freezes_projection(head_kind, rollouts, padding):
    torch.manual_seed(11)
    wrapper = make_wrapper(head_kind, rollouts)
    tokens = torch.randint(0, 13, (2, 7))
    if padding:
        tokens[0, -2:] = -100
    total, losses = wrapper(tokens, return_loss_breakdown = True)
    expected = reference_kl(wrapper, tokens, head_kind)
    torch.testing.assert_close(losses.kl, expected, rtol = 1e-4, atol = 1e-6)
    assert losses.kl > 0
    parameters = tuple(wrapper.parameters())
    actual_gradients = torch.autograd.grad(losses.kl, parameters, allow_unused = True, retain_graph = True)
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused = True)
    for actual, reference in zip(actual_gradients, expected_gradients):
        if reference is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, reference, rtol = 1e-4, atol = 2e-6)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)


def test_next_latent_linear_projection_does_not_allocate_a_second_vocabulary_weight():
    torch.manual_seed(11)
    wrapper = make_wrapper('linear', 1, dim = 32, num_tokens = 32768)
    tokens = torch.randint(0, 32768, (1, 5))
    wrapper(tokens)
    weight_bytes = wrapper.net.to_logits.weight.numel() * wrapper.net.to_logits.weight.element_size()
    with torch.profiler.profile(activities = [torch.profiler.ProfilerActivity.CPU], profile_memory = True) as profile:
        wrapper(tokens)
    largest_allocation = max(event.cpu_memory_usage for event in profile.events())
    # The logits and KL workspaces for this short input are much smaller than a vocabulary weight.
    assert largest_allocation < weight_bytes / 2
