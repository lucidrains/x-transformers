import pytest
import torch
from x_transformers.gpt_vae import GPTVAE


def make_model(pad_id):
    torch.manual_seed(77)
    return GPTVAE(
        num_tokens=11,
        dim=8,
        dim_latent=3,
        depth=1,
        enc_depth=1,
        max_seq_len=8,
        heads=2,
        attn_dim_head=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        latents_dropout_prob=0.0,
        pad_id=pad_id,
    )


def padded_sequence(pad_id):
    return torch.tensor(
        [[1, 2, 3, 4, pad_id, pad_id], [2, 4, 5, pad_id, pad_id, pad_id]]
    )


def reference_latents(model, seq):
    # Use the complete native encoder with known valid embedding IDs. The
    # production encode_to_latents helper is deliberately not used here.
    mask = seq != model.pad_id
    safe_seq = torch.where(mask, seq, torch.zeros_like(seq))
    pooled = model.encoder(safe_seq, mask=mask)
    mean, log_variance = model.to_latent_mean_log_variance(pooled)
    sampled = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
    return sampled, mean, log_variance


@pytest.mark.parametrize("pad_id", [-1, 0, 20])
@pytest.mark.parametrize("no_padding", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_encode_ignores_padding_before_embedding_and_matches_gaussian_oracle(
    pad_id, no_padding, training
):
    model = make_model(pad_id).train(training)
    seq = padded_sequence(pad_id)
    if no_padding:
        seq = torch.tensor([[1, 2, 3, 4, 5, 6], [2, 4, 5, 6, 7, 8]])
    torch.manual_seed(113)
    actual, (actual_mean, actual_log_var) = model.encode_to_latents(
        seq, return_mean_log_var=True
    )
    torch.manual_seed(113)
    expected, expected_mean, expected_log_var = reference_latents(model, seq)
    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_log_var, expected_log_var)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("pad_id", [-1, 0, 20])
@pytest.mark.parametrize("separate_latent_seq", [False, True])
def test_full_gpt_vae_training_handles_padding_and_matches_loss_oracle(
    pad_id, separate_latent_seq
):
    model = make_model(pad_id).train()
    seq = padded_sequence(pad_id)
    latent_seq = seq.flip(0) if separate_latent_seq else seq
    kwargs = {"seq_for_latents": latent_seq} if separate_latent_seq else {}
    torch.manual_seed(997)
    total, (actual_ar, actual_kl) = model(seq, return_all_losses=True, **kwargs)
    torch.manual_seed(997)
    latents, mean, log_var = reference_latents(model, latent_seq)
    expected_kl = (0.5 * (log_var.exp() + mean.square() - log_var - 1)).sum(-1).mean()
    expected_ar = model.ar_wrapped_decoder(
        seq,
        prepend_embeds=model.from_latent_to_prepend_token(latents),
        seq_start_pos=torch.zeros(seq.shape[0], dtype=torch.long),
    )
    torch.testing.assert_close(actual_ar, expected_ar)
    torch.testing.assert_close(actual_kl, expected_kl)
    torch.testing.assert_close(total, expected_ar + expected_kl)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
