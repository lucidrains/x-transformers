import pytest
import torch
from x_transformers.continuous_autoencoder import ContinuousTransformerAutoencoder


@pytest.mark.parametrize("loss_type", ["l1", "l2"])
@pytest.mark.parametrize("bottleneck", ["deterministic", "variational"])
@pytest.mark.parametrize("variable_lengths", [False, True])
@pytest.mark.parametrize("unreduced", [False, True])
@pytest.mark.parametrize("seq_len", [1, 6])
def test_reconstruction_predicts_each_target_from_its_strict_prefix(
    loss_type, bottleneck, variable_lengths, unreduced, seq_len
):
    torch.manual_seed(191)
    model = ContinuousTransformerAutoencoder(
        dim=8,
        dim_input=3,
        dim_latent=2,
        enc_depth=1,
        dec_depth=1,
        max_seq_len=8,
        bottleneck_type=bottleneck,
        loss_type=loss_type,
        heads=2,
        attn_dim_head=4,
        latents_dropout_prob=0.0,
    )
    inputs = torch.randn(2, seq_len, 3, requires_grad=True)
    lens = torch.tensor([seq_len, max(1, seq_len - 2)]) if variable_lengths else None
    torch.manual_seed(223)
    actual, (actual_recon, actual_aux) = model(
        inputs,
        lens=lens,
        return_all_losses=True,
        return_unreduced_loss=unreduced,
    )
    # An independent autoregressive oracle makes one complete native decoder
    # call per target, exposing only its strict prefix and the latent token.
    torch.manual_seed(223)
    latents, expected_aux = model.encode(
        inputs, lens=lens, return_aux_loss=True, return_unreduced_loss=unreduced
    )
    prepend = model.from_latent_to_prepend_token(latents)
    predictions = []
    for target_pos in range(seq_len):
        prefix_predictions = model.decoder(
            inputs[:, :target_pos],
            prepend_embeds=prepend,
            seq_start_pos=torch.zeros(2, dtype=torch.long),
        )
        predictions.append(prefix_predictions[:, -1])
    predictions = torch.stack(predictions, 1)
    errors = predictions - inputs
    errors = errors.abs() if loss_type == "l1" else errors.square()
    if lens is None:
        expected_recon = errors.mean((1, 2)) if unreduced else errors.mean()
    else:
        per_row = torch.stack(
            [row[:length].mean() for row, length in zip(errors, lens)]
        )
        expected_recon = per_row if unreduced else (per_row * lens).sum() / lens.sum()
    expected = expected_recon if expected_aux is None else expected_recon + expected_aux
    torch.testing.assert_close(actual_recon, expected_recon)
    if expected_aux is None:
        assert actual_aux is None
    else:
        torch.testing.assert_close(actual_aux, expected_aux)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), inputs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    actual.sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_native_decoder_prediction_cannot_read_current_or_future_targets():
    model = ContinuousTransformerAutoencoder(
        dim=8,
        dim_input=3,
        dim_latent=2,
        enc_depth=1,
        dec_depth=1,
        max_seq_len=8,
        heads=2,
        attn_dim_head=4,
        latents_dropout_prob=0.0,
    ).eval()
    seq = torch.randn(2, 6, 3)
    prepend = model.from_latent_to_prepend_token(torch.randn(2, 2))
    original = model.decoder(seq[:, :-1], prepend_embeds=prepend)
    changed = seq.clone()
    changed[:, 2:] += 10
    modified = model.decoder(changed[:, :-1], prepend_embeds=prepend)
    torch.testing.assert_close(original[:, :3], modified[:, :3])
