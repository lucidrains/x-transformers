import torch

from x_transformers import (
    TransformerWrapper,
    Decoder,
    AutoregressiveWrapper
)

def test_kv_cache():

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 8,
            depth = 1,
            heads = 4
        )
    )

    model.eval()

    prompts = torch.zeros((1, 16))

    logits, cache = model(
        prompts,
        return_intermediates = True
    )

    sampled = logits[:, -1].argmax(dim = -1, keepdim = True)
    prompts = torch.cat((prompts, sampled), dim = -1)

    next_logits = model(prompts)
    next_logits_with_cache = model(prompts, cache = cache)

    assert torch.allclose(next_logits[:, -1], next_logits_with_cache[:, -1], atol = 1e-6)
