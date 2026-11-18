from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


def make_model(components=2, looped=False, **kwargs):
    return TransformerWrapper(
        num_tokens=16, max_seq_len=8, looped=looped,
        mixture_of_softmax=components > 0, mixture_of_softmax_k=max(components, 1),
        attn_layers=Decoder(dim=16, depth=1, heads=2, attn_dim_head=8, pre_and_post_norm=looped),
        **kwargs,
    )


def reference_projection(model, hidden):
    if model.to_mixture is None:
        logits = F.linear(hidden, model.to_logits.weight)
        return logits + F.logsigmoid(logits) if model.sigsoftmax_logits else logits
    count = model.combine_mixture.out_features
    components = F.linear(hidden, model.to_mixture[0].weight).reshape(*hidden.shape[:-1], count, -1)
    logits = F.linear(components, model.to_logits.weight)
    if model.sigsoftmax_logits:
        logits = logits + F.logsigmoid(logits)
    weights = F.linear(hidden, model.combine_mixture.weight).softmax(-1)
    return (logits.softmax(-1) * weights.unsqueeze(-1)).sum(-2).clamp(min=1e-20).log()


@pytest.mark.parametrize("components", [0, 2, 4])
@pytest.mark.parametrize("mode", ["embeddings", "both", "intermediates"])
def test_embedding_returns_preserve_decoder_hidden_states(components, mode):
    torch.manual_seed(16)
    model = make_model(components)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    observed = []
    handle = model.attn_layers.register_forward_hook(lambda module, args, output: observed.append(output[0]))
    try:
        if mode == "embeddings":
            embeddings = model(tokens, return_embeddings=True)
        elif mode == "both":
            logits, embeddings = model(tokens, return_logits_and_embeddings=True)
            torch.testing.assert_close(logits, reference_projection(model, observed[-1]))
        else:
            embeddings, intermediates = model(tokens, return_embeddings_and_intermediates=True)
            torch.testing.assert_close(embeddings, intermediates.last_layer_hiddens)
    finally:
        handle.remove()

    assert embeddings.shape == (2, 4, 16)
    torch.testing.assert_close(embeddings, observed[-1])
    embeddings.square().mean().backward()
    assert model.token_emb.emb.weight.grad is not None


@pytest.mark.parametrize("components", [0, 2])
def test_embedding_only_with_logit_options_and_prefix(components):
    torch.manual_seed(16)
    model = make_model(components, sigsoftmax_logits=True, squeeze_out_last_dim=True)
    tokens = torch.tensor([[1, 2, 3]])
    prefix = torch.randn(1, 2, 16)
    (logits, hidden), cache = model(
        tokens, prepend_embeds=prefix, return_logits_and_embeddings=True, return_intermediates=True
    )
    embeddings = model(tokens, prepend_embeds=prefix, return_embeddings=True, excise_prepend_embeds=True)
    assert embeddings.shape == (1, 3, 16)
    torch.testing.assert_close(embeddings, cache.last_layer_hiddens[:, 2:])
    torch.testing.assert_close(logits, reference_projection(model, cache.last_layer_hiddens))
    torch.testing.assert_close(hidden, cache.last_layer_hiddens)


@pytest.mark.parametrize("components", [0, 2, 4])
@pytest.mark.parametrize("looped", [False, True])
def test_continuous_predictions_use_hidden_states(components, looped):
    torch.manual_seed(16)
    model = make_model(components, looped, add_continuous_pred_head=True)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    (_, (predicted, _)), cache = model(tokens, return_next_embed_pred=True, return_intermediates=True)
    assert predicted.shape == (2, 4, 16)
    torch.testing.assert_close(predicted, model.to_next_embed_pred(cache.last_layer_hiddens))

    loss = AutoregressiveWrapper(model)(tokens)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.to_next_embed_pred[0].weight.grad is not None


@pytest.mark.parametrize("components", [0, 2, 4])
@pytest.mark.parametrize("sigsoftmax", [False, True])
def test_loop_predictions_apply_the_complete_output_head(components, sigsoftmax):
    torch.manual_seed(16)
    model = make_model(components, looped=True, sigsoftmax_logits=sigsoftmax)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    hidden_states = []
    handle = model.attn_layers.register_forward_hook(lambda module, args, output: hidden_states.append(output[0]))
    try:
        logits, cache = model(tokens, looped_steps=3, return_intermediates=True)
    finally:
        handle.remove()

    assert len(cache.all_pred_logits) == 3
    for prediction, hidden in zip(cache.all_pred_logits, hidden_states):
        torch.testing.assert_close(prediction, reference_projection(model, hidden))
        if components:
            torch.testing.assert_close(prediction.exp().sum(-1), torch.ones_like(prediction[..., 0]))
    torch.testing.assert_close(logits, cache.all_pred_logits[-1])


@pytest.mark.parametrize("components", [2, 4])
def test_looped_mixture_training_matches_explicit_exit_weighted_nll(components):
    torch.manual_seed(16)
    model = make_model(components, looped=True)
    reference = deepcopy(model)
    wrapper = AutoregressiveWrapper(model)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    hidden_states = []
    handle = reference.attn_layers.register_forward_hook(lambda module, args, output: hidden_states.append(output[0]))
    try:
        _, cache = reference(tokens, looped_steps=3, return_intermediates=True)
    finally:
        handle.remove()
    targets = tokens[:, 1:]
    losses = torch.stack([
        F.nll_loss(reference_projection(reference, hidden)[:, :-1].transpose(1, 2), targets, reduction="none")
        for hidden in hidden_states
    ], dim=1)
    gates = torch.stack(cache.exit_logits, dim=1)[:, :, :-1, 0]
    exit_targets = torch.sigmoid(
        wrapper.looped_loss_slope * (losses[:, 1:] - losses[:, :-1] + wrapper.looped_loss_threshold_exit)
    ).detach()
    exit_loss = F.binary_cross_entropy_with_logits(gates[:, :-1], exit_targets)
    continuation = torch.ones_like(gates[:, 0])
    expected_weighted = torch.zeros_like(continuation)
    for step in range(3):
        gate = gates[:, step].sigmoid().detach() if step < 2 else torch.ones_like(continuation)
        expected_weighted = expected_weighted + continuation * gate * losses[:, step]
        continuation = continuation * (1 - gate)
    # The wrapper averages the weighted contributions across all loop/token entries.
    expected = expected_weighted.mean() / 3 + wrapper.looped_exit_loss_weight * exit_loss
    actual = wrapper(tokens, looped_steps=3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=1e-6, rtol=1e-5)
