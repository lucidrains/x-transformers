import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange, repeat

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, mask = None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context = None, mask = None, context_mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, dots.shape[-2]), device = device).bool())
            k_mask = default(context_mask, lambda: torch.ones((b, dots.shape[-1]), device = device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            mask = q_mask * k_mask
            dots.masked_fill_(mask, float('-inf'))
            del mask

        if self.causal:
            mask = torch.ones((n, n), device = device).triu_(1).bool()
            dots.masked_fill_(mask, float('-inf'))
            del mask

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, dim_head = 64, heads = 8):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))
    def forward(self, x, context = None, mask = None):
        for (self_attn, ff) in self.layers:
            x = self_attn(x, mask = mask)
            x = ff(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, dim_head = 64, heads = 8, cross_attend = False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads, causal = True))),
                Residual(PreNorm(dim, Attention(dim, dim_head = dim_head, heads = heads))) if cross_attend else None,
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))
    def forward(self, x, context = None, mask = None, context_mask = None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x)
            if exists(cross_attn):
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask)
            x = ff(x)
        return x

class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, layer_blocks, heads = 8, return_logits = True):
        super().__init__()
        dim = layer_blocks.dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layer_blocks = layer_blocks
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device = device))
        x = self.layer_blocks(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)

class XTransformer(nn.Module):
    def __init__(self, *, num_tokens, dim, depth, max_seq_len, heads = 8, return_tgt_loss = False):
        super().__init__()

        self.encoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            layer_blocks = Encoder(dim, depth, heads),
            return_logits = False
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            layer_blocks = Decoder(dim, depth, heads, cross_attend = True),
            return_logits = True
        )

        if return_tgt_loss:
            self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        enc = self.encoder(src, mask = src_mask)
        out = self.decoder(tgt, context = enc, mask = tgt_mask, context_mask = src_mask)
        return out
