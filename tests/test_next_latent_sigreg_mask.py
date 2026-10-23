import pytest
import torch
from x_transformers import Decoder, TransformerWrapper
from x_transformers.next_latent_wrapper import NextLatentWrapper


@pytest.mark.parametrize("dynamics", ["residual", "gru"])
@pytest.mark.parametrize("num_rollouts", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_sigreg_regularizes_prediction_states_with_matching_target_mask(
    dynamics, num_rollouts, padding, enabled
):
    torch.manual_seed(97)
    net = TransformerWrapper(
        num_tokens=13,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )
    wrapper = NextLatentWrapper(
        net,
        dim=8,
        num_rollouts=num_rollouts,
        dynamics_type=dynamics,
        sigreg_loss_weight=0.3 if enabled else 0.0,
        sigreg_loss_kwargs={"num_slices": 7, "domain": (-2, 2), "num_knots": 5},
    )
    seq = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 4, 5, 6, 7, 8]])
    if padding:
        seq[0, -2:] = -100
        seq[1, -1] = -100
    torch.manual_seed(517)
    total, breakdown = wrapper(seq, return_loss_breakdown=True)
    safe = torch.where(seq == -100, 0, seq)
    hiddens = net(safe, return_embeddings=True)
    selected = hiddens[:, :-1][seq[:, 1:] != -100]
    if enabled:
        # Independent real/imaginary characteristic-function oracle, using the
        # same reproducible random projections and explicit trapezoid weights.
        torch.manual_seed(517)
        projections = torch.randn(7, 8)
        projections = projections / projections.norm(dim=-1, keepdim=True)
        knots = torch.linspace(-2, 2, 5)
        angles = (selected @ projections.t())[:, :, None] * knots
        real = angles.cos().mean(0)
        imag = angles.sin().mean(0)
        reference = (-0.5 * knots.square()).exp()
        integrand = ((real - reference).square() + imag.square()) * reference
        expected = (
            ((integrand[:, 1:] + integrand[:, :-1]) * 0.5 * knots.diff()).sum(-1).mean()
        )
        torch.testing.assert_close(breakdown.sigreg, expected)
        weight = net.token_emb.emb.weight
        actual_grad = torch.autograd.grad(breakdown.sigreg, weight, retain_graph=True)[
            0
        ]
        expected_grad = torch.autograd.grad(expected, weight)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
    else:
        torch.testing.assert_close(breakdown.sigreg, torch.tensor(0.0))
    total.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None
    )
