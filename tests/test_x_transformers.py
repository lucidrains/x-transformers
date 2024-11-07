import pytest
import torch

from x_transformers.x_transformers import (
    XTransformer,
    TransformerWrapper,
    Encoder,
    Decoder,
    AutoregressiveWrapper,
)

from x_transformers.neo_mlp import (
    NeoMLP
)

from x_transformers.multi_input import MultiInputTransformerWrapper

def test_readme():
    model = XTransformer(
        dim = 512,
        enc_num_tokens = 256,
        enc_depth = 6,
        enc_heads = 8,
        enc_max_seq_len = 1024,
        dec_num_tokens = 256,
        dec_depth = 6,
        dec_heads = 8,
        dec_max_seq_len = 1024,
        tie_token_emb = True
    )

    src = torch.randint(0, 256, (1, 1024))
    src_mask = torch.ones_like(src).bool()
    tgt = torch.randint(0, 256, (1, 1024))

    loss = model(src, tgt, mask = src_mask)
    loss.backward()

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

def test_adaptive_layernorm():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            dim_condition = 768,
            depth = 12,
            heads = 8,
            use_adaptive_layernorm = True,
            use_adaptive_layerscale = True
        )
    )

    x = torch.randint(0, 256, (2, 1024))
    condition = torch.randn(2, 768)

    model(x, condition = condition)

def test_adaptive_rmsnorm():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            dim_condition = 768,
            depth = 12,
            heads = 8,
            use_adaptive_rmsnorm = True,
            adaptive_condition_mlp = True
        )
    )

    x = torch.randint(0, 256, (2, 1024))
    condition = torch.randn(2, 768)

    model(x, condition = condition)

def test_attn_softclamp_logits():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            dim_condition = 768,
            depth = 12,
            heads = 8,
            attn_softclamp_logits = True,
        )
    )

    x = torch.randint(0, 256, (1, 1024))

    model(x)

def test_multiple_input_embeds():
    model = MultiInputTransformerWrapper(
        num_tokens = dict(
            note = 20000,
            pitch = 32,
            tone = 16
        ),
        max_seq_len = 1024,
        return_only_embed = True,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = dict(
        note = torch.randint(0, 20000, (2, 1024)),
        pitch = torch.randint(0, 32, (2, 1024)),
        tone = torch.randint(0, 16, (2, 1024))
    )

    embed = model(x)

    assert embed.shape == (2, 1024, 128)

def test_average_pool_embed():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        num_memory_tokens = 2,
        average_pool_embed = True,
        attn_layers = Encoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = torch.randint(0, 20000, (2, 1024))
    mask = torch.randint(0, 2, (2, 1024)).bool()

    logits = model(x, mask = mask)

    assert logits.shape == (2, 20000)

@pytest.mark.parametrize('num_cls_tokens', (1, 2))
def test_cls_token(num_cls_tokens):
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        num_memory_tokens = 2,
        use_cls_token = True,
        num_cls_tokens=num_cls_tokens,
        attn_layers = Encoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = torch.randint(0, 20000, (2, 1024))
    mask = torch.randint(0, 2, (2, 1024)).bool()

    logits = model(x, mask = mask)

    if num_cls_tokens == 1:
        expected_shape = (2, 20000)
    else:
        expected_shape = (2, num_cls_tokens, 20000)

    assert logits.shape == expected_shape

def test_squeeze_logit_dim_one():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        logits_dim = 1,
        average_pool_embed = True,
        squeeze_out_last_dim = True,
        attn_layers = Encoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = torch.randint(0, 20000, (2, 1024))
    mask = torch.randint(0, 2, (2, 1024)).bool()

    logits = model(x, mask = mask)

    assert logits.shape == (2,)

@pytest.mark.parametrize('depth', (4, 5))
def test_unet_skip(depth):

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Encoder(
            dim = 128,
            depth = depth,
            heads = 8,
            unet_skips = True
        )
    )

    x = torch.randint(0, 20000, (2, 1024))
    mask = torch.randint(0, 2, (2, 1024)).bool()

    model(x, mask = mask)

def test_recycling():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        recycling = True,
        train_max_recycle_steps = 5,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    logits = model(x)

    model.eval()

    eval_logits = model(x, recycle_steps = 3)

def test_mos():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        mixture_of_softmax = True,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    logits = model(x)

    model.eval()

    eval_logits = model(x)

@pytest.mark.parametrize('attn_one_kv_head', (True, False))
def test_l2_distance(attn_one_kv_head):

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 12,
            heads = 8,
            attn_l2_distance = True,
            attn_one_kv_head = attn_one_kv_head,
        )
    )

    x = torch.randint(0, 256, (1, 1024))

    model(x)

def test_reinject_input():

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        recycling = True,
        attn_layers = Decoder(
            dim = 512,
            depth = 12,
            heads = 8,
            reinject_input = True
        )
    )

    x = torch.randint(0, 256, (1, 1024))

    model(x) # (1, 1024, 20000)

def test_value_residual():

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            add_value_residual = True,
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

def test_forgetting_transformer():

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_data_dependent_alibi = False
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    embed = model(x)

def test_neo_mlp():

    mlp = NeoMLP(
        dim_in = 5,
        dim_out = 7,
        dim_hidden = 16,
        depth = 5,
        dim_model = 64,
    )

    x = torch.randn(3, 5)

    out = mlp(x)
    assert out.shape == (3, 7)

def test_custom_alibi():
    model = TransformerWrapper(
        num_tokens = 20_000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 2,
            heads = 8,
            alibi_pos_bias = True
        )
    )

    x = torch.randint(0, 20000, (2, 4))

    pos = torch.tensor([[0, 1, 2, 4], [1, 3, 5, 7]])

    logits = model(x, pos = pos)
