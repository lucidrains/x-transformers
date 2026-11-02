import pytest
import torch
import torch.nn.functional as F

from x_transformers.free_transformer import FreeTransformer


def make_model(pad_id, head_depth, per_token_latents):
    return FreeTransformer(
        num_tokens = 8,
        dim = 8,
        max_seq_len = 8,
        heads = 2,
        attn_dim_head = 8,
        dec_head_depth = head_depth,
        dec_tail_depth = 1,
        latent_bits = 2,
        pad_id = pad_id,
        per_token_latents = per_token_latents
    )


@pytest.mark.parametrize('pad_id', (-1, 0, 7, 8))
@pytest.mark.parametrize('head_depth', (0, 1))
@pytest.mark.parametrize('per_token_latents', (False, True))
@pytest.mark.parametrize('separate_latents', (False, True))
def test_free_transformer_masks_padding_before_embedding(
    pad_id,
    head_depth,
    per_token_latents,
    separate_latents
):
    torch.manual_seed(0)
    model = make_model(pad_id, head_depth, per_token_latents)
    seq = torch.tensor([[1, 2, 3, pad_id, pad_id], [3, 4, 5, 6, pad_id]])
    seq_for_latents = None
    if separate_latents:
        seq = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        seq_for_latents = torch.tensor([[1, 2, pad_id, pad_id], [3, 4, 5, pad_id]])

    embedding_inputs, logits, encoder_masks = [], [], []
    handles = [
        model.token_emb.register_forward_pre_hook(
            lambda module, args: embedding_inputs.append(args[0].detach().clone())
        ),
        model.token_unembed.register_forward_hook(
            lambda module, args, output: logits.append(output.detach())
        ),
        model.encoder.register_forward_pre_hook(
            lambda module, args, kwargs: encoder_masks.append(kwargs['context_mask']),
            with_kwargs = True
        )
    ]
    try:
        loss, (ar_loss, aux_loss) = model(
            seq,
            seq_for_latents,
            return_all_losses = True
        )
    finally:
        for handle in handles:
            handle.remove()

    assert embedding_inputs
    assert all(((tokens >= 0) & (tokens < 8)).all() for tokens in embedding_inputs)
    if 0 <= pad_id < 8:
        assert torch.equal(embedding_inputs[0], seq[:, :-1])
        if seq_for_latents is not None:
            assert torch.equal(embedding_inputs[1], seq_for_latents)
    expected_mask = (seq[:, :-1] if seq_for_latents is None else seq_for_latents) != pad_id
    assert torch.equal(encoder_masks[0], expected_mask)
    expected_loss = F.cross_entropy(
        logits[0].transpose(1, 2),
        seq[:, 1:],
        ignore_index = pad_id
    )
    torch.testing.assert_close(ar_loss, expected_loss)
    assert torch.isfinite(loss)
    assert torch.isfinite(aux_loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize('separate_latents', (False, True))
def test_free_transformer_still_rejects_nonpadding_invalid_ids(separate_latents):
    model = make_model(-1, 1, True)
    seq = torch.tensor([[1, 8, 2]]) if not separate_latents else torch.tensor([[1, 2, 3]])
    seq_for_latents = torch.tensor([[1, 8]]) if separate_latents else None

    with pytest.raises(IndexError):
        model(seq, seq_for_latents)
