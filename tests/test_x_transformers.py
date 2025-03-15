import pytest

import torch
from torch import nn
from torch.nn import Module

from x_transformers.x_transformers import (
    XTransformer,
    TransformerWrapper,
    Encoder,
    Decoder,
    LinearNoBias,
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

@pytest.mark.parametrize('learned_value_residual_mix', (False, True))
def test_value_residual(
    learned_value_residual_mix: bool
):

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            add_value_residual = True,
            learned_value_residual_mix = learned_value_residual_mix
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

@pytest.mark.parametrize('has_num_mem_kv', (False, True))
def test_forgetting_transformer(
    has_num_mem_kv: bool
):

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_num_mem_kv = 1 if has_num_mem_kv else 0,
            attn_data_dependent_alibi = True
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

@pytest.mark.parametrize('flash', (True, False))
def test_custom_alibi(flash: bool):

    model = TransformerWrapper(
        num_tokens = 20_000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 2,
            heads = 8,
            alibi_pos_bias = True,
            attn_flash = flash
        )
    )

    x = torch.randint(0, 20000, (2, 4))

    pos = torch.tensor([[0, 1, 2, 4], [1, 3, 5, 7]])

    logits = model(x, pos = pos)

@pytest.mark.parametrize('rotary_xpos', (True, False))
def test_custom_rotary_pos_emb(rotary_xpos):
    from einops import repeat

    model = TransformerWrapper(
        num_tokens = 20_000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 2,
            heads = 8,
            rotary_pos_emb = True,
            rotary_xpos = rotary_xpos
        )
    )

    x = torch.randint(0, 20000, (4, 4))

    pos = repeat(torch.arange(0, 4), "n -> b n", b=4)

    logits1 = model(x, pos = pos)
    logits2 = model(x)
    assert torch.allclose(logits1, logits2)

@pytest.mark.parametrize('flash', (True, False))
def test_custom_alibi_across_heads(flash: bool):
    model = Decoder(
        dim = 512,
        depth = 2,
        heads = 2,
        alibi_pos_bias = True,
        rel_pos_kwargs = dict(
            slopes = [1, 1]
        ),
        attn_flash = flash
    )

    x = torch.randn(2, 4, 512)

    pos = torch.tensor([
        [[0, 1, 2, 4], [1, 3, 5, 7]],
        [[2, 3, 4, 5], [6, 8, 9, 10]]
    ])

    embed = model(x, pos = pos)

@pytest.mark.parametrize('embedder_type', ('embedding', 'none', 'custom'))
def test_embedder(embedder_type):
    num_tokens = 20000
    dim = 128
    token_emb_kwargs = {}

    if embedder_type == 'embedding':
        embedder = nn.Embedding(num_tokens, dim)
    elif embedder_type == 'none':
        embedder = None
    else:
        class CustomEmbedder(Module):
            """
            Made up embedder that sums two embeddings. Just to check if we can pass additional input to the embedder's
            forward pass without breaking the model.
            """
            def __init__(self, num_tokens, dim):
                super().__init__()
                self.embed_x = nn.Embedding(num_tokens, dim)
                self.embed_y = nn.Embedding(num_tokens, dim)

            def forward(self, x, y):
                return self.embed_x(x) + self.embed_y(y)

            def init_(self):
                pass

        embedder = CustomEmbedder(num_tokens, dim)
        token_emb_kwargs['y'] = torch.randint(0, num_tokens, (2, 1024))

    model = TransformerWrapper(
        num_tokens = num_tokens,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = dim,
            depth = 6,
            heads = 8,
        ),
        token_emb = embedder,
    )

    x = torch.randint(0, 20000, (2, 1024))

    output = model(x, token_emb_kwargs=token_emb_kwargs)
    assert output.shape == (2, 1024, 20000)


@pytest.mark.parametrize("to_logits", ('linear', 'none', 'pointer'))
def test_to_logits(to_logits):
    num_tokens = 20000
    dim = 128

    to_logits_kwargs = {}

    if to_logits == 'linear':
        logit_mapper = LinearNoBias(dim, num_tokens)
    elif to_logits == 'none':
        logit_mapper = None
    else:
        class PointerNetworkLogits(Module):
            def __init__(self, dim):
                super().__init__()
                self.proj_to_pointers = nn.Linear(dim, dim)

            def forward(self, model_embeddings, input_embeddings):
                pointers = self.proj_to_pointers(model_embeddings)
                logits = torch.matmul(pointers, input_embeddings.permute(0, 2, 1))
                return logits

        logit_mapper = PointerNetworkLogits(dim)
        to_logits_kwargs['input_embeddings'] = torch.randn(2, 20000, dim)

    model = TransformerWrapper(
        num_tokens = num_tokens,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = dim,
            depth = 6,
            heads = 8,
        ),
        to_logits = logit_mapper,
    )

    x = torch.randint(0, num_tokens, (2, 1024))

    output = model(x, to_logits_kwargs=to_logits_kwargs)

    assert output.shape == (2, 1024, 20000)

def test_laser():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_laser = True
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

@pytest.mark.parametrize('self_attn_custom_pos', (True, False))
@pytest.mark.parametrize('cross_attn_rotary', (True, False))
def test_cross_attn_rotary(
    self_attn_custom_pos: bool,
    cross_attn_rotary: bool
):

    x = torch.randn((1, 64, 256))
    mask = torch.ones((1, 64)).bool()
    context = torch.randn((1, 128, 512))
    context_mask = torch.ones((1, 128)).bool()

    model = Encoder(
        dim = 256,
        depth = 4,
        heads = 4,
        rotary_pos_emb = True,
        cross_attend = True,
        cross_attn_dim_context = 512
    )

    pos = torch.arange(64) if self_attn_custom_pos else None
    context_pos = torch.arange(128) if cross_attn_rotary else None

    embed = model(
      x = x,
      mask = mask,
      context = context,
      pos = pos,
      context_pos = context_pos,
      context_mask = context_mask
    )

@pytest.mark.parametrize('tanh', (True, False))
def test_hyper_connections(tanh):

    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            num_residual_streams = 8, # 8 dynamic hyper connection residual streams
            residual_fn_kwargs = dict(
                tanh = tanh
            )
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

@pytest.mark.parametrize('hybrid_axial_dim', (1, 4))
def test_hybrid(hybrid_axial_dim):
    from torch.nn import GRU

    dec = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_dim_head = 64,
            attn_hybrid_fold_axial_dim = hybrid_axial_dim,
            attn_hybrid_module = GRU(128, 64 * 8, batch_first = True)
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    embed = dec(x)

    enc = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Encoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_dim_head = 64,
            attn_hybrid_fold_axial_dim = hybrid_axial_dim,
            attn_hybrid_module = GRU(128, 64 * 4, batch_first = True, bidirectional = True)
        )
    )

    mask = torch.randint(0, 2, (2, 1024)).bool()
    embed = enc(x, mask = mask)

def test_multi_latent_attention():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            attn_use_latent_q = True,
            attn_dim_latent_q = 128,
            attn_use_latent_kv = True,
            attn_dim_latent_kv = 128,
            attn_latent_rope_subheads = 4,
            rotary_pos_emb = False
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

@pytest.mark.parametrize('num_residual_streams', (1, 4))
@pytest.mark.parametrize('integrate_layers', (False, True))
def test_lime(
    num_residual_streams,
    integrate_layers
):
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            num_residual_streams = num_residual_streams,
            integrate_layers = integrate_layers
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)

@pytest.mark.parametrize('backward_ar_loss_weight', (1., 0.5))
@pytest.mark.parametrize('goal_suffix', (False, True))
@pytest.mark.parametrize('pred_distance', (False, True))
@pytest.mark.parametrize('variable_len', (False, True))
def test_belief_state_wrapper(
    backward_ar_loss_weight,
    goal_suffix,
    pred_distance,
    variable_len
):
    from x_transformers.belief_state_wrapper import BeliefStateWrapper

    forward_model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = True
        )
    )

    backward_model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = True
        )
    )

    model = BeliefStateWrapper(
        forward_decoder = forward_model,
        backward_decoder = backward_model,
        backward_ar_loss_weight = backward_ar_loss_weight,
        pred_distance = pred_distance
    )

    seq = torch.randint(0, 20000, (2, 16))

    lens = None

    if variable_len:
        lens = torch.randint(4, 16, (2,))

    loss = model(seq, lens = lens) # backwards happen automatically
    loss.backward()

    suffix = None
    if goal_suffix:
        suffix = torch.randint(0, 20000, (2, 2))

    sampled = model.generate_with_suffix_cond(seq[:, :1], 16, suffix = suffix)
    assert sampled.shape == (2, 16)

def test_dynamic_tanh():
    model = TransformerWrapper(
        num_tokens = 20000,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 128,
            depth = 6,
            heads = 8,
            use_dynamic_tanh = True,
            dynamic_tanh_init_alpha = 1.5
        )
    )

    x = torch.randint(0, 20000, (2, 1024))

    model(x)
