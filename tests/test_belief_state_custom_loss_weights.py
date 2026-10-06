import pytest
import torch
from torch import nn
from x_transformers import Decoder, TransformerWrapper
from x_transformers.belief_state_wrapper import BeliefStateWrapper


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("backward_weight", [1.0, 0.5])
@pytest.mark.parametrize(
    "weight_kind", ["none", "ones", "one_dimensional", "two_dimensional"]
)
def test_custom_pair_weights_are_applied_with_default_directional_weight(
    batch, backward_weight, weight_kind
):
    torch.manual_seed(109)
    decoder = TransformerWrapper(
        num_tokens=13,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )
    model = BeliefStateWrapper(decoder, backward_ar_loss_weight=backward_weight)
    seq = torch.tensor([[1, 2, 3, 4, 5], [2, 4, 5, 6, 7]])[:batch]
    calls = []

    def callback(pairs):
        calls.append(pairs.detach().clone())
        left, right = pairs.unbind(-1)
        if weight_kind == "ones":
            return torch.ones(pairs.shape[0])
        weights = (left.float() + 1) / (right.float() + 1)
        if weight_kind == "two_dimensional":
            weights = torch.stack((weights, weights * 0.3 + 0.2), -1)
        return weights

    actual = model(
        seq, loss_weight_by_fb_indices=None if weight_kind == "none" else callback
    )
    assert len(calls) == (0 if weight_kind == "none" else 1)
    # Enumerate valid prefix/suffix pairs directly instead of using the native
    # cartesian-product/filter routine, and compute both label likelihoods.
    pairs = [
        (left, right) for left in range(5) for right in range(1, 6) if right - left >= 2
    ]
    left = torch.tensor([pair[0] for pair in pairs])
    right = torch.tensor([pair[1] for pair in pairs])
    fwd = model.forward_decoder(seq, return_embeddings=True)
    suffix = model.suffix_token[None, None, :].expand(batch, 1, -1)
    bwd = model.backward_decoder(
        seq.flip(1), prepend_embeds=suffix, return_embeddings=True
    ).flip(1)
    logits = model.text_head(torch.cat((fwd[:, left], bwd[:, right]), -1))
    fwd_logits, bwd_logits = logits.chunk(2, -1)
    fwd_loss = nn.functional.cross_entropy(
        fwd_logits.transpose(1, 2), seq[:, left + 1], reduction="none"
    )
    bwd_loss = nn.functional.cross_entropy(
        bwd_logits.transpose(1, 2), seq[:, right - 1], reduction="none"
    )
    losses = torch.stack((fwd_loss, bwd_loss * backward_weight), 1)
    if weight_kind in ("one_dimensional", "two_dimensional"):
        weights = (left.float() + 1) / (right.float() + 1)
        if weight_kind == "one_dimensional":
            losses = losses * weights
        else:
            weights = torch.stack((weights, weights * 0.3 + 0.2))
            losses = losses * weights
    expected = losses.mean()
    torch.testing.assert_close(actual, expected)
    weight = model.text_head[-1].weight
    actual_grad = torch.autograd.grad(actual, weight, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    actual.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
