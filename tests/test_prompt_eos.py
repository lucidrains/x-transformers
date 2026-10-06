import pytest
import torch
from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


@pytest.mark.parametrize("cache_kv", [False, True])
@pytest.mark.parametrize("prompt", [torch.tensor([1, 2, 1]), torch.tensor([1, 1, 1])])
def test_generate_ignores_eos_inside_prompt(cache_kv, prompt):
    net = TransformerWrapper(
        num_tokens=4,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=1, attn_dim_head=8),
    )
    with torch.no_grad():
        net.to_logits.weight.zero_()
    model = AutoregressiveWrapper(net)
    generated = model.generate(
        prompt, seq_len=3, eos_token=2, temperature=0, cache_kv=cache_kv
    )
    torch.testing.assert_close(generated, torch.zeros(3, dtype=torch.long))


@pytest.mark.parametrize("cache_kv", [False, True])
def test_generate_still_stops_on_generated_eos(cache_kv):
    net = TransformerWrapper(
        num_tokens=4,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=1, attn_dim_head=8),
    )
    with torch.no_grad():
        net.to_logits.weight.zero_()
    model = AutoregressiveWrapper(net)
    generated = model.generate(
        torch.tensor([1, 2, 1]),
        seq_len=3,
        eos_token=0,
        temperature=0,
        cache_kv=cache_kv,
    )
    torch.testing.assert_close(generated, torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("cache_kv", [False, True])
def test_generate_prompt_eos_does_not_pad_continuation(cache_kv):
    net = TransformerWrapper(
        num_tokens=4,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=1, attn_dim_head=8),
    )
    with torch.no_grad():
        net.to_logits.weight.zero_()
    model = AutoregressiveWrapper(net, pad_value=3)
    prompts = torch.tensor([[1, 2, 1], [1, 1, 1]])
    generated = model.generate(
        prompts, seq_len=3, eos_token=2, temperature=0, cache_kv=cache_kv
    )
    torch.testing.assert_close(generated, torch.zeros((2, 3), dtype=torch.long))



@pytest.mark.parametrize("cache_kv", [False, True])
def test_generate_zero_tokens_returns_empty_continuation(cache_kv):
    net = TransformerWrapper(
        num_tokens=4,
        max_seq_len=8,
        attn_layers=Decoder(dim=8, depth=1, heads=1, attn_dim_head=8),
    )
    model = AutoregressiveWrapper(net)
    for prompt in (torch.tensor([1, 2, 1]), torch.tensor([[1, 2, 1], [1, 1, 1]])):
        for eos_token in (None, 2):
            generated, cache = model.generate(
                prompt,
                seq_len=0,
                eos_token=eos_token,
                temperature=0,
                cache_kv=cache_kv,
                return_intermediates=True,
            )
            assert generated.shape == (*prompt.shape[:-1], 0)
            assert generated.dtype == prompt.dtype
            assert cache is None
