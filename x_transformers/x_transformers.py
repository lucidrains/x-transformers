import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
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

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim = -1, keepdim = True).clamp(min = self.eps)
        return x / n * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm_class = nn.LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm_class(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4, glu = False, dropout = 0.):
        super().__init__()
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU()
        ) if not glu else GEGLU(dim, dim * mult)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim_out)
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
    def __init__(self, dim, depth, dim_head = 64, heads = 8, use_scalenorm = False, ff_glu = False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        prenorm_fn = partial(PreNorm, dim, norm_class = norm_class)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                prenorm_fn(Attention(dim, dim_head = dim_head, heads = heads)),
                prenorm_fn(FeedForward(dim, glu = ff_glu))
            ]))
    def forward(self, x, context = None, mask = None):
        for (self_attn, ff) in self.layers:
            x = self_attn(x, mask = mask) + x
            x = ff(x) + x
        return x

class Decoder(nn.Module):
    def __init__(self, dim, depth, dim_head = 64, heads = 8, cross_attend = False, use_scalenorm = False, ff_glu = False):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        prenorm_fn = partial(PreNorm, dim, norm_class = norm_class)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                prenorm_fn(Attention(dim, dim_head = dim_head, heads = heads, causal = True)),
                prenorm_fn(Attention(dim, dim_head = dim_head, heads = heads)) if cross_attend else None,
                prenorm_fn(FeedForward(dim, glu = ff_glu)),
            ]))
    def forward(self, x, context = None, mask = None, context_mask = None):
        for (self_attn, cross_attn, ff) in self.layers:
            x = self_attn(x) + x
            if exists(cross_attn):
                x = cross_attn(x, context = context, mask = mask, context_mask = context_mask) + x
            x = ff(x) + x
        return x

class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        num_classes = None,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = FeedForward(dim, dim_out = num_classes, dropout = dropout) if exists(num_classes) else None

    def forward(self, img):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head):
            return x

        return self.mlp_head(x[:, 0])

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        heads = 8,
        return_logits = True
    ):
        super().__init__()
        dim = attn_layers.dim
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()
        self.to_logits = lambda t: t @ self.token_emb.weight.t() if return_logits else nn.Identity()

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std = 0.02)
        nn.init.normal_(self.pos_emb.weight, std = 0.02)

    def forward(self, x, **kwargs):
        _, n, device = *x.shape, x.device
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device = device))
        x = self.attn_layers(x, **kwargs)
        x = self.norm(x)
        return self.to_logits(x)

class XTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        return_tgt_loss = False
    ):
        super().__init__()

        self.encoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Encoder(dim, depth, heads),
            return_logits = False
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(dim, depth, heads, cross_attend = True),
            return_logits = True
        )

        if return_tgt_loss:
            self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        enc = self.encoder(src, mask = src_mask)
        out = self.decoder(tgt, context = enc, mask = tgt_mask, context_mask = src_mask)
        return out
