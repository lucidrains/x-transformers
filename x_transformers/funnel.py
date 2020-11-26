import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from x_transformers.x_transformers import Attention, FeedForward, ScaleNorm, PreNorm, Rezero, groupby_prefix_and_trim

from einops import repeat, reduce

# helpers

def exists(val):
    return val is not None

def residualize(f):
    def fn(x, *args, **kwargs):
        out, *rest = f(x, *args, **kwargs)
        return (out + x, *rest)
    return fn

# classes

class AttentionWithDownsample(Attention):
    def forward(self, x, num_memory_tokens = 0, downsample = False, **kwargs):
        if not downsample:
            return super().forward(x, **kwargs), None

        b, n, *_, orig_x = *x.shape, x
        is_odd = (n % 2) == 1

        mem, x       = x[:, :num_memory_tokens], x[:, num_memory_tokens:]
        x, remainder = (x[:, :-1], x[:, -1:]) if is_odd else (x[:, :], x[:, 0:0])
        x = reduce(x, 'b (n c) d -> b n d', 'mean', c = 2)
        x = torch.cat((mem, x, remainder), dim = 1)

        mask = kwargs.pop('mask', None)
        orig_mask = mask

        if exists(mask):
            mask = mask[:, num_memory_tokens:]
            mask = F.pad(mask, (0, 1), value = False) if is_odd else mask
            mask = mask.reshape(b, -1, 2).any(dim = -1)
            mask = F.pad(mask, (num_memory_tokens, 0), value = True)

        return super().forward(
            x,
            mask = mask,
            context = orig_x,
            context_mask = orig_mask,
            **kwargs
        ), mask

class FunnelEncoder(nn.Module):
    def __init__(self, dim, depths, heads = 8, use_scalenorm = False, use_rezero = False, rel_pos_bias = False, num_memory_tokens = 0, **kwargs):
        super().__init__()
        assert isinstance(depths, tuple), 'depths must be a tuple, where each element specifies the number of layers before the next bottleneck'
        assert len(depths) > 1, 'there must be at least 1 bottleneck'

        self.dim = dim
        self.num_memory_tokens = num_memory_tokens
        self.bottlenecks = nn.ModuleList([])

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        prenorm_fn = partial(PreNorm, dim, norm_class = norm_class)
        prenorm_fn = Rezero if use_rezero else prenorm_fn

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        for depth in depths:
            layers = nn.ModuleList([])
            rel_pos = RelativePositionBias() if rel_pos_bias else None

            for _ in range(depth):
                layers.append(nn.ModuleList([
                    prenorm_fn(AttentionWithDownsample(dim, heads = heads, **attn_kwargs)),
                    prenorm_fn(FeedForward(dim, **ff_kwargs))
                ]))

            self.bottlenecks.append(nn.ModuleList([rel_pos, layers]))

    def forward(self, x, context = None, mask = None):
        n = x.shape[1]
        num_mem = self.num_memory_tokens
        num_downsamples = len(self.bottlenecks)

        for layer_ind, (rel_pos, layers) in enumerate(self.bottlenecks):
            if layer_ind == 1:
                res = x

            for ind, (self_attn, ff) in enumerate(layers):
                downsample = layer_ind != 0 and ind == 0
                self_attn = residualize(self_attn) if not downsample else self_attn

                x, new_mask = self_attn(x, mask = mask, rel_pos = rel_pos, downsample = downsample, num_memory_tokens = num_mem)
                x = ff(x) + x

                if exists(new_mask):
                    mask = new_mask

        mem, x = x[:, :num_mem], x[:, num_mem:]
        # upsample by repeating tokens as specified in paper
        x = repeat(x, 'b n d -> b (n m) d', m = 2 ** (num_downsamples - 1))
        # curtail any excessive tokens
        x = x[:, :(n - num_mem)]
        x = torch.cat((mem, x), dim = 1)
        # add to residual before start of first downsample
        return x + res