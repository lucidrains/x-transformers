import pytest
import torch

from x_transformers import Decoder, XValAutoregressiveWrapper, XValTransformerWrapper


@pytest.mark.parametrize('ignore_index', (-100, 99))
@pytest.mark.parametrize('pad_value', (0, 2))
@pytest.mark.parametrize('with_mask', (False, True))
def test_xval_ignored_input_ids_preserve_unpadded_loss(ignore_index, pad_value, with_mask):
    torch.manual_seed(0)
    wrapper = XValAutoregressiveWrapper(
        XValTransformerWrapper(
            num_tokens = 4,
            numerical_token_id = 3,
            max_seq_len = 16,
            attn_layers = Decoder(dim = 8, depth = 1, heads = 2)
        ),
        ignore_index = ignore_index,
        pad_value = pad_value
    ).eval()
    tokens = torch.tensor([[1, 3, 2, 3, ignore_index, ignore_index]])
    numbers = torch.tensor([[float('nan'), 0.5, float('nan'), -1., float('nan'), float('nan')]])
    mask = tokens != ignore_index
    kwargs = dict(mask = mask) if with_mask else dict()

    seen_inputs = []
    hook = wrapper.net.token_emb.register_forward_pre_hook(
        lambda module, args: seen_inputs.append(args[0].detach().clone())
    )
    loss = wrapper(tokens, numbers, **kwargs)
    hook.remove()
    unpadded_loss = wrapper(tokens[:, :4], numbers[:, :4])

    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, unpadded_loss)
    assert seen_inputs[0][0, -1].item() == pad_value
    assert tokens[0, -1].item() == ignore_index

    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None)
