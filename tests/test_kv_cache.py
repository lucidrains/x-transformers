import torch

from x_transformers.x_transformers import (
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
            depth = 2,
            heads = 4,
            cross_attend = True
        )
    )

    model.eval()

    prompts = torch.zeros((2, 16))
    context = torch.randn(2, 8, 8)

    logits, cache = model(
        prompts,
        context = context,
        return_intermediates = True
    )

    sampled = logits[:, -1].argmax(dim = -1, keepdim = True)
    prompts = torch.cat((prompts, sampled), dim = -1)

    next_logits = model(prompts, context = context)
    next_logits_with_cache = model(prompts, context = context, cache = cache)

    assert torch.allclose(next_logits[:, -1], next_logits_with_cache[:, -1], atol = 1e-6)

def test_cope():
    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 8,
            depth = 2,
            heads = 4,
            attn_use_cope = True
        )
    )

    seq = torch.randint(0, 256, (1, 1024))
    logits = model(seq)
