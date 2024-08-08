import torch
from torch import nn    
from x_transformers.x_transformers import (
    XTransformer,
    TransformerWrapper,
    Decoder,
    Encoder,
    AutoregressiveWrapper
)



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

def test_classification():
    # CLS token test
    transformer = TransformerWrapper(
        num_tokens=6,
        max_seq_len=10,
        logits_dim=2, # num_classes 
        use_cls_token=True,
        attn_layers = Encoder(
            dim = 6,
            depth = 1,
            heads = 2,
        )
    )

    x = torch.randint(0, 5, (2, 10))
    y = torch.tensor([0, 1])

    print(x.shape)
    logits = transformer(x)
    print(logits.shape)
    loss = nn.CrossEntropyLoss()(logits, y)

    print(loss)

    # BCE cls token

    transformer = TransformerWrapper(
        num_tokens=6,
        max_seq_len=10,
        logits_dim=1, # num_classes 
        use_cls_token=True,
        squeeze_out_last_dim = True,
        attn_layers = Encoder(
            dim = 6,
            depth = 1,
            heads = 2,
        )
    )

    x = torch.randint(0, 5, (2, 10)).float()
    y = torch.tensor([0, 1]).float()

    print(x.shape)
    logits = transformer(x).squeeze()
    loss = nn.BCEWithLogitsLoss()(logits, y)

    print(loss)

    # pooling test
    transformer = TransformerWrapper(
        num_tokens=6,
        max_seq_len=10,
        logits_dim=2, # num_classes 
        average_pool_embed = True,
        attn_layers = Encoder(
            dim = 6,
            depth = 1,
            heads = 2,
        )
    )

    x = torch.randint(0, 5, (2, 10))
    y = torch.tensor([0, 1])

    print(x.shape)
    logits = transformer(x)
    print(logits.shape)
    loss = nn.CrossEntropyLoss()(logits, y)

    print(loss)

    # pooling BCE test

    # pooling test
    transformer = TransformerWrapper(
        num_tokens=6,
        max_seq_len=10,
        logits_dim=1, # num_classes 
        average_pool_embed = True,
        squeeze_out_last_dim = True,
        attn_layers = Encoder(
            dim = 6,
            depth = 1,
            heads = 2,
        )
    )

    x = torch.randint(0, 5, (2, 10)).float()
    y = torch.tensor([0, 1]).float()

    print(x.shape)
    logits = transformer(x).squeeze()
    print(logits.shape)
    loss = nn.BCEWithLogitsLoss()(logits, y)

    print(loss)

    # normal test 

    transformer = TransformerWrapper(
        num_tokens=6,
        max_seq_len=10,
        logits_dim=2, # num_classes 
        average_pool_embed = True,
        attn_layers = Encoder(
            dim = 6,
            depth = 1,
            heads = 2,
        )
    )

    x = torch.randint(0, 5, (1, 10))
    y = torch.tensor([0])

    print(x.shape)
    logits = transformer(x)
    print(logits.shape)