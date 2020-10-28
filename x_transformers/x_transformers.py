import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
    def __init__(self, dim, heads = 8, causal = False, mask = None):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'
        self.scale = (dim // heads) ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_kv = nn.Linear(dim, dim * 2, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, context = None, mask = None):
        _, n, _, h, device = *x.shape, self.heads, x.device
        kv_input = default(context, x)

        q = self.to_q(x)
        kv = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones((n, n), device = device).triu_(1).bool()
            dots.masked_fill_(mask, float('-inf'))

        attn = dots.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim)))
            ]))
    def forward(self, x, context = None):
        for (self_attn, ff) in self.layers:
            x = self_attn(x)
            x = ff(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, heads = 8):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, causal = True))),
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim))),
            ]))
    def forward(self, x, context = None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x)
            x = cross_attn(x, context = context)
            x = ff(x)
        return x

class TransformerWrapper(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, layer_blocks, heads = 8, return_logits = True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layer_blocks = layer_blocks
        self.to_logits = nn.Linear(dim, num_tokens) if return_logits else nn.Identity()

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device = device))
        x = self.layer_blocks(x, **kwargs)
        return self.to_logits(x)

class XTransformer(nn.Module):
    def __init__(self, *, num_tokens, dim, depth, max_seq_len, heads = 8):
        super().__init__()

        self.encoder = TransformerWrapper(
            num_tokens = num_tokens,
            dim = dim,
            max_seq_len = max_seq_len,
            layer_blocks = Encoder(dim, depth, heads),
            return_logits = False
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            dim = dim,
            max_seq_len = max_seq_len,
            layer_blocks = Decoder(dim, depth, heads),
            return_logits = True
        )

    def forward(self, src, tgt, mems = None):
        enc = self.encoder(src)
        out = self.decoder(tgt, context = enc)
        return out
