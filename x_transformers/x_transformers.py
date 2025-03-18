from __future__ import annotations
from typing import Callable

import math
from copy import deepcopy
from random import random, randrange
from packaging import version

import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import nn, einsum, tensor, Tensor, cat, stack, arange, is_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.nn import Module, ModuleList, ModuleDict

from functools import partial, wraps
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass

from loguru import logger

from x_transformers.attend import Attend, Intermediates
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

# einstein notation

# b - batch
# n - sequence
# d - feature dimension
# h - attention heads
# i, j - sequence (source, target)

# constants

DEFAULT_DIM_HEAD = 64

@dataclass
class LayerIntermediates:
    hiddens:            list[Tensor] | None = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Tensor | None = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: list[Intermediates] | None = None
    layer_hiddens:      list[Tensor] | None = None
    attn_z_loss:        Tensor | None = None
    mems:               Tensor | None = None
    memory_tokens:      Tensor | None = None
    logit_entropies:    Tensor | None = None

LinearNoBias = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def first(it, default = None):
    return it[0] if len(it) > 0 else default

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def softclamp(t, value):
    return (t / value).tanh() * value

def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# entropy

def calc_entropy(
    t: Tensor,
    is_prob = False
):
    prob = t.softmax(dim = -1) if not is_prob else t
    return -(prob * log(prob)).sum(dim = -1)

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: list[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)

# positional embeddings

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = arange(j - i, j, dtype = torch.long, device = device)
        k_pos = arange(j, dtype = torch.long, device = device)
        rel_pos = einx.subtract('j, i -> i j', k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """
    def __init__ (
        self,
        dim,
        heads,
        max_pos,
        soft_onehot = False,
        talking_heads = False,
        soft_onehot_temp = 5e-2
    ):
        super () . __init__ ()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer('positions', arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim = -1).flip(-1)
        pos = pos.clamp(max = self.max_pos - 1)

        logits_int = einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract('i, j -> i j', pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim = -1)
            cope_pos_emb = einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb

class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = arange(j - i, j, device = device)
        context_arange = arange(j, device = device)
        indices = einx.subtract('i, j -> i j', seq_arange, context_arange)
        indices += (j - 1)

        # input to continuous positions MLP
        pos = arange(-j + 1, j, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(Module):
    def __init__(
        self,
        heads,
        total_heads = None,
        slopes: list[int] | None = None,
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, 'h -> h 1 1')

        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    @property
    def device(self):
        return next(self.buffers()).device

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward_custom_pos(
        self,
        pos_i: Tensor,
        pos_j: Tensor | None = None
    ):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract('... j, ... i -> ... i j', pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, 'b i j -> b 1 i j')

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = arange(j - i, j, device = device)
        context_arange = arange(j, device = device)
        bias = -einx.subtract('j, i -> 1 i j', context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

class DataDependentAlibi(Module):
    """ https://openreview.net/forum?id=q2Lnyegkr8 """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        bias_init = 5.,
        post_log_scale = 1.,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n h -> b h n'),
            nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim = -1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim = 1)

        forget_gates = einx.subtract('b h i, b h j -> b h i j', forget_gates, forget_gates)

        if bidirectional:
            forget_gates_reversed = einx.subtract('b h j, b h i -> b h i j', forget_gates_reversed, forget_gates_reversed)
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates

class PerRowDataDependentAlibi(Module):
    """ same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        dim_head = 8,
        post_log_scale = 1.
    ):
        super().__init__()
        assert causal, 'bidirectional not supported yet'

        self.scale = dim_head ** -0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias = False)

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n (qk h d) -> qk b h n d', qk = 2, d = dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype = torch.bool, device = x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims = (-1,))
        forget_gates = forget_gates.cumsum(dim = -1)
        forget_gates = forget_gates.flip(dims = (-1,))

        return forget_gates

class RotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

# norms

class Scale(Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class ScaleNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = F.normalize(x, dim = -1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.)

class SimpleRMSNorm(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = SimpleRMSNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

class DynamicTanh(Module):
    """ https://arxiv.org/abs/2503.10622 """
    def __init__(
        self,
        dim,
        init_alpha = 1.,
        gamma = 1.,
        beta = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.pre_tanh_scale = nn.Parameter(tensor(init_alpha))

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.pre_tanh_scale_offset = init_alpha if unit_offset else 0.
        self.gamma_offset = float(unit_offset)

        nn.init.constant_(self.pre_tanh_scale, 0 if unit_offset else init_alpha)
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        pre_tanh_scale = self.pre_tanh_scale + self.pre_tanh_scale_offset
        gamma = self.gamma + self.gamma_offset
        return (x * pre_tanh_scale).tanh() * gamma + self.beta

# residual and residual gates

class Residual(Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1., **kwargs):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# hyper connections

class HyperConnection(Module):
    def __init__(
        self,
        dim,
        *,
        layer_index,
        num_residual_streams,
        num_input_views = 1,
        tanh = True,
        **kwargs
    ):
        """
        https://arxiv.org/abs/2409.19606
        Appendix J - Algorithm 2, Dynamic only
        """
        super().__init__()

        self.act = nn.Tanh() if tanh else nn.Identity()

        self.norm = nn.LayerNorm(dim, bias = False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
        init_alpha0[layer_index % num_residual_streams, :] = 1.

        self.static_alpha = nn.Parameter(cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + num_input_views))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.num_input_views = num_input_views

        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

    def prepare(self, residuals):

        residuals = rearrange(residuals, '(b s) n d -> b n s d', s = self.num_residual_streams)

        normed = self.norm(residuals)

        wc_weight = self.act(normed @ self.dynamic_alpha_fn)
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        dc_weight = self.act(normed @ self.dynamic_beta_fn)
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        # width connection

        mix_h = einsum('... s t, ... s d -> ... t d', alpha, residuals)

        views = self.num_input_views

        if views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :views, :], mix_h[..., views:, :]
            branch_input = rearrange(branch_input, '... v d -> v ... d')

        return branch_input, residuals, dict(beta = beta)

    def forward(self, x, residuals, *, beta):
        residuals = einsum('b n d, b n s -> b n s d', x, beta) + residuals
        return rearrange(residuals, 'b n s d -> (b s) n d')

# LIMe - layer integrated memory (dynamic version)

class DynamicLIMe(Module):
    def __init__(
        self,
        dim,
        num_layers,
        num_views = 1,
        norm = True,
        use_softmax = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.multiple_views = num_views > 1

        self.to_weights = Sequential(
            RMSNorm(dim) if norm else None,
            nn.Linear(dim, num_views * num_layers),
            Rearrange('... (views layers) -> views ... layers', views = num_views),
            nn.Softmax(dim = -1) if use_softmax else nn.ReLU()
        )

    def forward(
        self,
        x,
        hiddens
    ):

        if not is_tensor(hiddens):
            hiddens = stack(hiddens)

        assert hiddens.shape[0] == self.num_layers, f'expected hiddens to have {self.num_layers} layers but received {tuple(hiddens.shape)} instead (first dimension must be layers)'

        weights = self.to_weights(x)

        out = einsum('l b n d, v b n l -> v b n d', hiddens, weights)

        if self.multiple_views:
            return out

        return rearrange(out, '1 ... -> ...')

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class ShiftTokens(Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = [shift(*args, mask = mask) for args in zip(segments_to_shift, shifts)]
        x = cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

class FoldAxially(Module):
    def __init__(
        self,
        axial_dim,
        fn: Module
    ):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim # will fold the sequence as rearrange("b (n axial_dim) ... -> (b axial_dim) n ...")

    def forward(
        self,
        x,
        **kwargs
    ):
        if self.axial_dim == 1:
            return self.fn(x, **kwargs)

        seq_len, axial_dim = x.shape[1], self.axial_dim

        next_multiple = math.ceil(seq_len / axial_dim) * axial_dim
        x = pad_at_dim(x, (0, next_multiple - seq_len), dim = 1)

        x = rearrange(x, 'b (n axial_dim) ... -> (b axial_dim) n ...', axial_dim = axial_dim)

        out = self.fn(x, **kwargs)

        (out, *rest_out), tree_spec = tree_flatten(out)

        out = rearrange(out, '(b axial_dim) n ... -> b (n axial_dim) ...', axial_dim = axial_dim)

        out = out[:, :seq_len]
        out = tree_unflatten((out, *rest_out), tree_spec)

        return out

# post branch operator

class LayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        init_value = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset

        self.fn = fn
        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        gamma = self.gamma + float(self.unit_offset)

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

class AdaptiveLayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        self.fn = fn

        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition, **kwargs):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        out = self.fn(x, **kwargs)
        gamma = self.to_gamma(condition).sigmoid()

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

# skip connection combining

class ConcatCombine(Module):
    def __init__(self, dim, prev_layer_ind):
        super().__init__()
        self.prev_layer_ind = prev_layer_ind
        self.combine = LinearNoBias(dim * 2, dim)

    def forward(self, x, prev_layers: list[Tensor]):
        skip = prev_layers[self.prev_layer_ind]
        concatted_skip = cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

# feedforward

class GLU(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias

class FeedForward(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

# attention. it is all we need

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        dim_context = None,
        heads = 8,
        causal = False,
        flash = False,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        sparse_topk_straight_through = False,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        swiglu_values = False,
        gate_values = False,
        zero_init_output = False,
        hard = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        l2_distance = False,
        sigmoid = False,
        selective = False,
        custom_attn_fn: Callable | None = None,
        hybrid_module: Module | None = None,
        hybrid_mask_kwarg: str | None = None,
        hybrid_fold_axial_dim: int | None = None,
        hybrid_learned_mix = False,
        one_kv_head = False,
        kv_heads = None,
        value_dim_head = None,
        dim_out = None,
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        rotate_num_heads = None,
        data_dependent_alibi = False,
        data_dependent_alibi_per_row = False,
        data_dependent_alibi_per_row_dim_head = 8,
        data_dependent_alibi_kwargs: dict = dict(),
        use_cope = False,
        cope_max_pos = 16,
        cope_soft_onehot_pos = False,
        cope_talking_heads = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        learned_value_residual_mix = False,
        laser = False,                # https://arxiv.org/abs/2411.03493v1
        laser_softclamp_value = 15.,
        qkv_receive_diff_residuals = False,
        use_latent_q = False,
        dim_latent_q = None,
        use_latent_kv = False,
        dim_latent_kv = None,
        latent_rope_subheads = None,
        onnxable = False,
        attend_sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        dim_kv = default(dim_context, dim)

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), 'either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both'

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        # determine input dimensions to qkv based on whether intermediate latent q and kv are being used
        # for eventually supporting multi-latent attention (MLA)

        self.to_latent_q = None
        self.to_latent_kv = None
        self.to_rotateable_k = None # for their "decoupled rope", subheads of keys that comes directly from base sequence (does not go through latents)

        dim_q_input = dim
        dim_kv_input = dim_kv

        if use_latent_q:
            assert exists(dim_latent_q)
            self.to_latent_q = LinearNoBias(dim, dim_latent_q)
            dim_q_input = dim_latent_q

        if use_latent_kv:
            assert exists(dim_latent_kv)
            self.to_latent_kv = LinearNoBias(dim, dim_latent_kv)
            dim_kv_input = dim_latent_kv

        if exists(latent_rope_subheads):
            assert not exists(rotate_num_heads), '`rotate_num_heads` cannot be set when multi-latent attention is being used'
            rotate_num_heads = latent_rope_subheads

            k_dim = dim_head * (kv_heads - latent_rope_subheads)

            self.to_rotateable_k = LinearNoBias(dim, dim_head * latent_rope_subheads)
            self.split_rotateable_k_heads = Rearrange('b n (h d) -> b h n d', h = latent_rope_subheads)

        self.use_latent_q = use_latent_q
        self.use_latent_kv = use_latent_kv

        # query key projection

        self.to_q = LinearNoBias(dim_q_input, q_dim)
        self.to_k = LinearNoBias(dim_kv_input, k_dim)
        self.to_v = LinearNoBias(dim_kv_input, v_dim)

        # split and merge of attention heads

        self.split_q_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_k_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.split_v_heads = Rearrange('b n (h d) -> b h n d', d = value_dim_head)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # whether qkv receives different residual stream combinations from hyper connections or lime

        self.qkv_receive_diff_residuals = qkv_receive_diff_residuals

        # enhancing gradients to attention through exponentiated values

        self.laser = laser
        self.laser_softclamp_value = laser_softclamp_value

        # add GLU gating for aggregated values, from alphafold2

        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            self.to_v_gate_activation = F.silu if swiglu_values else F.sigmoid
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 10)

        # add per head gating of the output values, from 'Attend to nothing' paper

        self.to_v_head_gate = None
        if gate_value_heads:
            self.to_v_head_gate = nn.Linear(dim, heads)
            nn.init.constant_(self.to_v_head_gate.weight, 0)
            nn.init.constant_(self.to_v_head_gate.bias, 10)

        # cosine sim attention

        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442

        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # contextual positional encoding
        # https://arxiv.org/html/2405.18719v2

        cope = None

        if use_cope:
            assert causal, 'CoPE was designed for causal attention'
            assert not flash, 'CoPE is not flash attention compatible'

            cope = CoPE(
                dim = dim_head,
                heads = heads,
                max_pos = cope_max_pos,
                talking_heads = cope_talking_heads,
                soft_onehot = cope_soft_onehot_pos
            )

        # data dependent alibi
        # https://openreview.net/forum?id=q2Lnyegkr8

        self.data_dependent_alibi = None

        if data_dependent_alibi:

            dda_klass = DataDependentAlibi if not data_dependent_alibi_per_row else PerRowDataDependentAlibi
            dda_kwargs = dict(dim = dim, heads = heads, causal = causal)

            if data_dependent_alibi_per_row:
                dda_kwargs.update(dim_head = data_dependent_alibi_per_row_dim_head)

            self.data_dependent_alibi = dda_klass(**dda_kwargs, **data_dependent_alibi_kwargs)

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            pre_talking_heads = pre_talking_heads,
            post_talking_heads = post_talking_heads,
            pre_scale_post_talking_heads = pre_scale_post_talking_heads,
            dropout = dropout,
            sparse_topk = sparse_topk,
            sparse_topk_straight_through = sparse_topk_straight_through,
            hard = hard,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            l2_distance = l2_distance,
            sigmoid = sigmoid,
            selective = selective,
            custom_attn_fn = custom_attn_fn,
            add_zero_kv = add_zero_kv,
            flash = flash,
            softclamp_logits = softclamp_logits,
            logit_softclamp_value = logit_softclamp_value,
            cope = cope,
            onnxable = onnxable,
            sdp_kwargs = attend_sdp_kwargs
        )

        # head scaling

        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention

        self.sparse_topk = sparse_topk

        # add memory key / values

        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))

        # maybe learned value residual mixer per token

        self.to_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
         ) if learned_value_residual_mix else always(0.5)

        # attention on attention

        self.attn_on_attn = on_attn

        # hybrid module, in same vein as hymba https://www.arxiv.org/abs/2411.13676

        hybrid_mix = None
        hybrid_norms = None
        hybrid_module = maybe(deepcopy)(hybrid_module)

        if exists(hybrid_module) and exists(hybrid_fold_axial_dim):
            hybrid_module = FoldAxially(axial_dim = hybrid_fold_axial_dim, fn = hybrid_module)
            hybrid_mix = LinearNoBias(dim, heads) if hybrid_learned_mix else None

            hybrid_norms = ModuleList([
                MultiheadRMSNorm(dim_head, heads = heads),
                MultiheadRMSNorm(dim_head, heads = heads)
            ])

        self.hybrid_module = hybrid_module
        self.hybrid_norms = hybrid_norms
        self.hybrid_mix = hybrid_mix
        self.hybrid_mask_kwarg = hybrid_mask_kwarg # for bidirectional, can forward `mask` into the hybrid module and let it handle variable lengths

        # output dimension by default same as input, but can be overridden

        dim_out = default(dim_out, dim)
        self.to_out = nn.Sequential(LinearNoBias(out_dim, dim_out * 2), nn.GLU()) if on_attn else LinearNoBias(out_dim, dim_out)

        # the number of attention heads to rotate, for decoupled rope in multi-latent attention

        rotate_num_heads = default(rotate_num_heads, heads)

        assert 0 < rotate_num_heads <= heads
        is_partial_rotate_heads = rotate_num_heads < heads
        assert not (is_partial_rotate_heads and kv_heads < heads), 'grouped query attention not compatible with partial rotate heads (decoupled rope for multi-latent attention), yet'

        self.rotate_num_heads = rotate_num_heads

        # whether parent can kv cache

        self.can_cache_kv = not selective

        # init output projection 0

        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        attn_bias = None,
        rotary_pos_emb = None,
        context_rotary_pos_emb = None,
        pos = None, # for custom alibi positions
        prev_attn = None,
        mem = None,
        mem_mask = None,
        return_intermediates = False,
        cache: Intermediates | None = None,
        value_residual = None
    ):
        b, n, h, kv_h, head_scale, num_mem_kv, device, has_context, qkv_receive_diff_residuals, is_multi_latent_attn = x.shape[0], x.shape[1], self.heads, self.kv_heads, self.head_scale, self.num_mem_kv, x.device, exists(context), self.qkv_receive_diff_residuals, self.use_latent_kv

        # an interesting possibility with hyper connections
        # having queries, keys, values be routed from different layers

        assert not (qkv_receive_diff_residuals and has_context), 'qkv receiving different sequences can only be used for self attention'

        if qkv_receive_diff_residuals:
            assert x.ndim == 4 and x.shape[0] == 3

            q_input, k_input, v_input = x
        else:
            kv_input = default(context, x)
            q_input, k_input, v_input = x, kv_input, kv_input

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        # multi-latent attention logic
        # https://arxiv.org/abs/2405.04434 - Deepseek-AI team

        k_sub_heads = None # the rotateable subheads of keys derived from base sequence

        if self.use_latent_q:
            q_input = self.to_latent_q(q_input)

        if is_multi_latent_attn:
            assert not qkv_receive_diff_residuals
            needs_k_sub_heads = exists(self.to_rotateable_k)

            latent_kv_input = self.to_latent_kv(k_input)

            if needs_k_sub_heads:
                rotateable_k = self.to_rotateable_k(k_input)
                k_sub_heads = self.split_rotateable_k_heads(rotateable_k)

            if exists(cache):
                cached_latent_kv, maybe_cached_k_sub_heads = cache.cached_kv
                latent_kv_input = cat((cached_latent_kv, latent_kv_input), dim = -2)

                if exists(maybe_cached_k_sub_heads):
                    k_sub_heads = cat((maybe_cached_k_sub_heads, k_sub_heads), dim = -2)

            if return_intermediates:
                cached_kv = (latent_kv_input, k_sub_heads)

            k_input = v_input = latent_kv_input

        # query, key, value projection

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = self.split_q_heads(q)
        k = self.split_k_heads(k)
        v = self.split_v_heads(v)

        # take care of decoupled rope from multi-latent attention

        if exists(k_sub_heads):
            k = cat((k, k_sub_heads), dim = 1)

        # if previous values passed in for residual, either invoke resformer

        orig_values = v

        # https://arxiv.org/abs/2410.17897v1

        if exists(value_residual):
            value_residual_mix = self.to_value_residual_mix(q_input)
            v = value_residual.lerp(v, value_residual_mix)

        # qk normalization

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        # take care of caching

        if not is_multi_latent_attn:
            if exists(cache):
                ck, cv = cache.cached_kv

                if exists(mem):
                    mk, k = unpack(k, mem_packed_shape, 'b h * d')
                    mv, v = unpack(v, mem_packed_shape, 'b h * d')

                k = cat((ck, k), dim = -2)
                v = cat((cv, v), dim = -2)

                if exists(mem):
                    k = cat((mk, k), dim = -2)
                    v = cat((mv, v), dim = -2)

            if return_intermediates:
                mem_len = mem.shape[-2] if exists(mem) else 0
                cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if exists(rotary_pos_emb):
            rotate_num_heads = self.rotate_num_heads
            partial_rotate_heads = rotate_num_heads < h

            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            if partial_rotate_heads:
                q_rest, q = q[:, :-rotate_num_heads], q[:, -rotate_num_heads:]
                k_rest, k = k[:, :-rotate_num_heads], k[:, -rotate_num_heads:]

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)

            if has_context:
                # override with `context_rotary_pos_emb` if provided

                freqs, xpos_scale = context_rotary_pos_emb
                _, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if partial_rotate_heads:
                q = cat((q_rest, q), dim = 1)
                k = cat((k_rest, k), dim = 1)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

            if (exists(input_mask) or exists(mem_mask)) and exists(mem):
                seq_len, mem_len = n, mem.shape[-2]

                if not exists(mem_mask):
                    input_mask = pad_at_dim(input_mask, (mem_len, 0), dim = -1, value = True)
                elif not exists(input_mask):
                    input_mask = pad_at_dim(mem_mask, (0, seq_len), dim = -1, value = True)
                else:
                    input_mask = cat((mem_mask, input_mask), dim = -1)

        # i, j determined for relative positional bias, excluding memory key / values

        i, j = tuple(t.shape[-2] for t in (q, k))

        # maybe append memory key / values

        if num_mem_kv > 0:
            mem_k, mem_v = tuple(repeat(t, 'h n d -> b h n d', b = b) for t in (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = cat((mem_k, k), dim = -2)
            v = cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = arange(j - i, j, device = device)
            range_k = arange(j, device = device)
            dist = einx.subtract('i, j -> 1 1 i j', range_q, range_k)
            max_attend_past_mask = dist > self.max_attend_past
            max_attend_past_mask = pad_at_dim(max_attend_past_mask, (num_mem_kv, 0), value = False, dim = -1) # handle memory key / values
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        if exists(rel_pos):
            assert not exists(attn_bias)

            if exists(pos):
                assert isinstance(rel_pos, AlibiPositionalBias), 'only alibi allowed for custom positions at the moment'
                # allow for custom positions to be passed in
                attn_bias = rel_pos.forward_custom_pos(pos)
            else:
                attn_bias = rel_pos(i, j)

            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0)) # handle memory key / values

        # prepare data dependent alibi from forgetting transformers paper, if needed

        if exists(self.data_dependent_alibi):
            attn_bias = self.data_dependent_alibi(x)

            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0))

        if self.laser:
            v = softclamp(v, self.laser_softclamp_value)
            v = v.exp()

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # laser

        if self.laser:
            out = log(out)

        # store the values for resformer

        intermediates.values = orig_values

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = einx.multiply('b n h, b h n d ->b h n d', head_gate.sigmoid(), out)

        # if exists hybrid module, must do a normalization

         # hybrid module

        if exists(self.hybrid_module):

            # hybrid input

            hybrid_forward_kwargs = dict()

            if not self.causal and exists(self.hybrid_mask_kwarg):
                hybrid_forward_kwargs = {self.hybrid_mask_kwarg: mask}

            # hybrid forward

            hybrid_outputs = self.hybrid_module(x, **hybrid_forward_kwargs)

            # handle hybrid out

            (hybrid_out, *rest_hybrid_outs), _ = tree_flatten(hybrid_outputs)

            # handle variable hybrid output and multi rmsnorm before summing to main attention output (also normed)

            if hybrid_out.ndim == 3:
                hybrid_out = rearrange(hybrid_out, 'b n (h d) -> b h n d', h = h)

            out_norm, hybrid_out_norm = self.hybrid_norms

            out = out_norm(out)
            hybrid_out = hybrid_out_norm(hybrid_out)

            if exists(self.hybrid_mix):
                mix = self.hybrid_mix(x)
                mix = rearrange(mix, 'b n h -> b h n 1')
                out = out.lerp(hybrid_out, mix.sigmoid())
            else:
                out = 0.5 * (out + hybrid_out)

        # merge heads

        out = self.merge_heads(out)

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * self.to_v_gate_activation(gates)

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(Module):
    def __init__(
        self,
        dim,
        depth = None,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_dynamic_tanh = False,
        dynamic_tanh_init_alpha = 1.,
        use_simple_rmsnorm = False,
        use_adaptive_layernorm = False,
        use_adaptive_rmsnorm = False,
        use_adaptive_layerscale = False, # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
        norm_add_unit_offset = True,
        dim_condition = None,
        adaptive_condition_mlp = False,
        adaptive_condition_mlp_expansion = 4,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        rotate_num_heads = None,
        weight_tie_layers = False,
        custom_layers: tuple[str, ...] | None = None,
        layers_execute_order: tuple[int, ...] | None = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        softclamp_output = False,
        softclamp_output_value = 30.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        use_layerscale = False,
        layerscale_init_value = 0.,
        unet_skips = False,
        integrate_layers = False,
        layer_integrate_use_softmax = True,
        num_residual_streams = 1,
        qkv_receive_diff_residuals = False,
        reinject_input = False,              # seen first in DEQ paper https://arxiv.org/abs/1909.01377, but later used in a number of papers trying to achieve depthwise generalization https://arxiv.org/abs/2410.03020v1
        learned_reinject_input_gate = False,
        add_value_residual = False,          # resformer from Zhou et al - https://arxiv.org/abs/2410.17897v1 - further corroboration by https://arxiv.org/abs/2412.15113 (faster emergence of ICL) - looks like this setting may becoming a necessity for every transformer soon
        learned_value_residual_mix = True,   # seeing big improvements when the value residual mix value is learned per token - credit goes to @faresobeid for taking the first step with learned scalar mix, then @Blinkdl for taking it a step further with data dependent. here we will use per token learned
        rel_pos_kwargs: dict = dict(),
        residual_fn_kwargs: dict = dict(),
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim('cross_attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)
        data_dependent_alibi = attn_kwargs.get('data_dependent_alibi', False)

        assert len(kwargs) == 0, f'unrecognized kwargs passed in {kwargs.keys()}'

        self.dim = dim
        self.causal = causal
        self.layers = ModuleList([])

        # routing related
        # 1. greater than one residual stream, proposed in Hyper-Connections paper https://arxiv.org/abs/2409.19606
        # 2. integrating more than one past layer, from LIMe paper https://arxiv.org/abs/2502.09245

        qkv_receive_diff_residuals |= integrate_layers # qkv always receives different views if integrating layers

        # hyper connections

        assert num_residual_streams > 0
        has_hyper_connections = num_residual_streams > 1

        self.num_residual_streams = num_residual_streams
        self.stream_emb = nn.Parameter(torch.zeros(num_residual_streams, dim)) if num_residual_streams > 1 else None

        assert not (has_hyper_connections and gate_residual)

        hyper_conn_produce_diff_views = qkv_receive_diff_residuals and not integrate_layers

        # LIMe

        hiddens_counter = 0
        self.layer_integrators = ModuleList([])

        assert not (qkv_receive_diff_residuals and not (hyper_conn_produce_diff_views or integrate_layers))

        # positions related

        self.disable_abs_pos_emb = default(disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb))

        rotary_emb_dim = default(rotary_emb_dim, dim_head // 2)

        assert rotary_emb_dim <= dim_head, f'rotary emb dim {rotary_emb_dim} must be less than or equal to attention head dimension {dim_head}'

        if rotary_emb_dim < 32:
            logger.warning('when training language model, rotary embedding dimension should be at least 32')

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert at_most_one_of(alibi_pos_bias, rel_pos_bias, data_dependent_alibi), 'you can only choose one of Alibi positional bias, data dependent Alibi (forgetting transformers), or T5 relative positional bias'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert at_most_one_of(rel_pos_bias, dynamic_pos_bias, alibi_pos_bias), 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None

        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance, **rel_pos_kwargs)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm, **rel_pos_kwargs)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads, **rel_pos_kwargs)

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        # determine norm

        assert at_most_one_of(use_scalenorm, use_rmsnorm, use_dynamic_tanh, use_simple_rmsnorm, use_adaptive_layernorm, use_adaptive_rmsnorm), 'you can only use either scalenorm, rmsnorm, adaptive layernorm, adaptive rmsnorm, or simple rmsnorm'

        norm_need_condition = False
        dim_condition = default(dim_condition, dim)
        dim_condition_mult = 1

        if adaptive_condition_mlp:
            dim_condition_mult = adaptive_condition_mlp_expansion

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        elif use_dynamic_tanh:
            assert pre_norm, 'dynamic tanh norm only tested for pre-norm'
            norm_class = partial(DynamicTanh, init_alpha = dynamic_tanh_init_alpha)
        elif use_adaptive_layernorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveLayerNorm, dim_condition = dim_condition * dim_condition_mult)
        elif use_adaptive_rmsnorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveRMSNorm, dim_condition = dim_condition * dim_condition_mult)
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if not norm_need_condition and norm_add_unit_offset:
            # researcher Ohad Rubin shares in a blog post by adding an offset to gammas, they can be subjected to weight decay safely
            norm_fn = partial(norm_fn, unit_offset = True)

        self.norm_need_condition = norm_need_condition
        self.dim_condition = dim_condition

        # determine default block layer type order

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # determine post branch wrapper

        assert at_most_one_of(use_layerscale, use_adaptive_layerscale)

        post_branch_fn = None
        post_branch_fn_needs_condition = False

        if use_layerscale:
            post_branch_fn = partial(LayerScale, dim = dim, init_value = layerscale_init_value)
        elif use_adaptive_layerscale:
            post_branch_fn = partial(AdaptiveLayerScale, dim = dim, dim_condition = dim_condition * dim_condition_mult)
            post_branch_fn_needs_condition = True

        self.post_branch_fn_needs_condition = post_branch_fn_needs_condition

        if exists(post_branch_fn) and not post_branch_fn_needs_condition and norm_add_unit_offset:
            post_branch_fn = partial(post_branch_fn, unit_offset = True)

        # setup mlp for conditioning

        self.need_condition = norm_need_condition or post_branch_fn_needs_condition

        self.adaptive_mlp = nn.Identity()

        if self.need_condition and adaptive_condition_mlp:
            self.adaptive_mlp = nn.Sequential(
                LinearNoBias(dim_condition, dim_condition * dim_condition_mult),
                nn.SiLU()
            )

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (exists(layers_execute_order) and exists(custom_layers) and exists(depth)), 'depth should not be passed in if using custom layers and custom layer execution order'

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert exists(depth), 'depth must be passed in with `weight_tie_layers` = True'
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        len_default_block = 1

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            assert exists(depth), '`depth` must be passed in for `Decoder` or `Encoder`'
            layer_types = default_block * depth
            len_default_block = len(default_block)

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # set the depth

        depth = default(depth, len(self.layers_execute_order))
        self.depth = depth

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # optional soft clamping just before the final norm
        # used in gemma 2

        self.softclamp_output = softclamp_output
        self.softclamp_output_value = softclamp_output_value

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm else nn.Identity()

        # whether unet or not

        self.unet_skips = unet_skips
        num_skips = self.depth // len_default_block

        assert not (unet_skips and num_skips == 0), 'must have depth of at least 2 for unet skip connections'

        skip_indices = [i * len_default_block for i in range(num_skips)]

        self.skip_combines = ModuleList([])

        # whether there is reinjection of input at every layer

        self.reinject_input = reinject_input
        self.reinject_input_proj = nn.Linear(dim, dim, bias = False) if reinject_input else None
        self.learned_reinject_input_gate = nn.Linear(dim, 1, bias = False) if learned_reinject_input_gate else None

        # add the value from the first self attention block to all latter projected self attention values as a residual

        self.add_value_residual = add_value_residual

        is_first_self_attn = True
        is_first_cross_attn = True
        learned_value_residual_mix &= add_value_residual

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):

            # `ind` is the index of each module - attention, feedforward, cross attention
            # but `block_ind` refers to the typical enumeration of a transformer block (attn + ff + [optional] cross attn)

            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            is_last_layer = ind == (len(self.layer_types) - 1)

            # attention, cross attention, feedforward

            layer_qkv_receives_diff_view = layer_type == 'a' and qkv_receive_diff_residuals and not (is_first_self_attn and integrate_layers)

            if layer_type == 'a':
                self_attn_learned_value_residual = learned_value_residual_mix and not is_first_self_attn

                layer = Attention(dim, heads = heads, causal = causal, qkv_receive_diff_residuals = layer_qkv_receives_diff_view, learned_value_residual_mix = self_attn_learned_value_residual, rotate_num_heads = rotate_num_heads, **attn_kwargs)
                is_first_self_attn = False

            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **{**attn_kwargs, **cross_attn_kwargs})
                is_first_cross_attn = False

            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)

            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(post_branch_fn):
                layer = post_branch_fn(layer)

            layer_integrate = None

            if integrate_layers:
                num_layer_hiddens = ind + 1
                layer_integrate_num_view = 3 if layer_qkv_receives_diff_view else 1

                layer_integrate = DynamicLIMe(dim, num_layer_hiddens, num_views = layer_integrate_num_view, use_softmax = layer_integrate_use_softmax)

            if has_hyper_connections:
                residual_fn = partial(HyperConnection, num_residual_streams = num_residual_streams)

                if layer_type == 'a' and hyper_conn_produce_diff_views:
                    residual_fn = partial(residual_fn, num_input_views = 3)

            elif gate_residual:
                residual_fn = GRUGating
            else:
                residual_fn = Residual

            residual = residual_fn(dim, layer_index = ind, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant, **residual_fn_kwargs)

            # handle unet skip connection

            skip_combine = None
            is_latter_half = block_begin and block_ind >= (self.depth / 2)

            if self.unet_skips and is_latter_half:
                skip_combine = ConcatCombine(dim, skip_indices.pop())

            # all normalizations of the layer

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.skip_combines.append(skip_combine)

            self.layer_integrators.append(layer_integrate)

            self.layers.append(ModuleList([
                norms,
                layer,
                residual
            ]))

        # determine whether can cache kv

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention)])

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        mem_masks = None,
        seq_start_pos: Tensor | None = None,
        cache: LayerIntermediates | None = None,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None,
        pos = None,
        context_pos = None,
        attn_bias = None,
        condition = None,
        in_attn_cond = None, # https://arxiv.org/abs/2105.04090
        layers_execute_order: tuple[int, ...] | None = None
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'
        assert not (exists(condition) ^ self.need_condition), 'condition needs to be passed in if using adaptive layernorm or vice versa'

        # handle condition

        if exists(condition):
            assert condition.shape[-1] == self.dim_condition, f'expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}'

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, 'b d -> b 1 d')

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()

        if self.norm_need_condition:
            norm_kwargs.update(condition = condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition = condition)

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        cross_attn_rotary_pos_emb = dict()

        if exists(self.rotary_pos_emb):
            if not exists(rotary_pos_emb):
                maybe_mem = first(mems, None) # todo - handle edge case where different layers get different memory lengths. don't think this will ever come up but who knows
                mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

                if not exists(pos):
                    pos = arange(x.shape[1] + mem_len, device = x.device) - mem_len

                rotary_pos_emb = self.rotary_pos_emb(pos)

            # allow for rotary positions for context if provided

            if exists(context_pos):
                assert self.cross_attend
                context_rotary_pos_emb = self.rotary_pos_emb(context_pos)

                cross_attn_rotary_pos_emb.update(
                    rotary_pos_emb = rotary_pos_emb,
                    context_rotary_pos_emb = context_rotary_pos_emb
                )

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert self.causal and not any([*map(exists, (mask, attn_mask))])

            if exists(context):
                context = context[:, :0]

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # setup multistreams if needed

        streams = self.num_residual_streams
        is_multistream = streams > 1

        if is_multistream:
            x = einx.add('b n d, s d -> (b s) n d', x, self.stream_emb)

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts,
            self.layer_integrators
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)
        layer_variables = tuple(tuple(layer_variable[i] for i in layers_execute_order) for layer_variable in layer_variables)

        # derived input for reinjection if needed

        inp_inject = None

        if self.reinject_input:
            assert not exists(in_attn_cond)
            inp_inject = self.reinject_input_proj(x)

        elif exists(in_attn_cond):
            # handle in-attention conditioning, which serves the same purpose of having the network learn the residual
            inp_inject = in_attn_cond if in_attn_cond.ndim == 3 else rearrange(in_attn_cond, 'b d -> b 1 d')

        if exists(inp_inject) and exists(self.learned_reinject_input_gate):
            inp_inject_gate = self.learned_reinject_input_gate(x).sigmoid()
            inp_inject = inp_inject * inp_inject_gate

        # store all hiddens for skips

        skip_hiddens = []

        # for value residuals

        first_self_attn_inter = None
        first_cross_attn_inter = None

        # go through the attention and feedforward layers

        for ind, (layer_type, skip_combine, (norm, block, residual_fn), layer_dropout, layer_integrator) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            x, inner_residual, residual_kwargs = residual_fn.prepare(x)

            layer_hiddens.append(x)

            if exists(layer_integrator):
                x = layer_integrator(x, layer_hiddens)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if exists(inp_inject):
                x = x + inp_inject

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == 'a' and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            block = partial(block, **block_forward_kwargs)

            # handle maybe value residuals

            maybe_self_attn_value_residual = None
            maybe_cross_attn_value_residual = None

            if self.add_value_residual:
                if exists(first_self_attn_inter):
                    maybe_self_attn_value_residual = first_self_attn_inter.values

                if exists(first_cross_attn_inter):
                    maybe_cross_attn_value_residual = first_cross_attn_inter.values

            # forward depending on layer type

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, pos = pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, mem_mask = layer_mem_mask, attn_bias = attn_bias, value_residual = maybe_self_attn_value_residual, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), value_residual = maybe_cross_attn_value_residual, **cross_attn_rotary_pos_emb, return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            # store first self or cross attention intermediate for value residual

            if not exists(first_self_attn_inter) and layer_type == 'a':
                first_self_attn_inter = inter

            if not exists(first_cross_attn_inter) and layer_type == 'c':
                first_cross_attn_inter = inter

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual, **residual_kwargs)

            if layer_type in ('a', 'c') and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        # take care of multistreams if needed, use sum for now

        if is_multistream:
            x = reduce(x, '(b s) n d -> b n d', 'sum', s = streams)

        x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
        )

        return x, intermediates

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class PrefixDecoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = False, **kwargs)

    def forward(
        self,
        x,
        *args,
        attn_mask = None,
        prefix_attn_len = None,
        **kwargs
    ):
        b, n, device = x.shape[0], x.shape[1], x.device
        causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)

        forwarded_mask = ~causal_mask

        if exists(prefix_attn_len):
            if isinstance(prefix_attn_len, int):
                prefix_attn_len = torch.full((b,), prefix_attn_len, device = device)

            prefix_mask = arange(n, device = device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
            forwarded_mask = forwarded_mask | prefix_mask

        if exists(attn_mask):
            forwarded_mask = forwarded_mask & attn_mask

        return super().forward(x, *args, attn_mask = forwarded_mask, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend = True, only_cross = True, **kwargs)

class ViTransformerWrapper(Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers: Encoder,
        channels = 3,
        num_classes = None,
        post_emb_norm = False,
        num_register_tokens = 0,
        emb_dropout = 0.
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim)
        )

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    def forward(
        self,
        img,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, 'n d -> b n d', b = b)
            x, ps = pack((x, r), 'b * d')

        embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, 'b * d')

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

        if not exists(self.mlp_head) or return_embeddings:
            return embed

        pooled = embed.mean(dim = -2)
        logits = self.mlp_head(pooled)

        if not return_logits_and_embeddings:
            return logits

        return logits, embed

class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        return_only_embed = False,
        num_output_heads = 1,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        recycling = False,            # from Jumper et al. - Alphafold2
        train_max_recycle_steps = 4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient = 1.,       # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
        average_pool_embed = False,
        use_cls_token = False,
        num_cls_tokens = 1,
        squeeze_out_last_dim = False,
        token_emb: TokenEmbedding | None = None,
        mixture_of_softmax = False,
        mixture_of_softmax_k = 4,
        sigsoftmax_logits = False,
        to_logits: Module | None = None,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_cls_tokens = num_cls_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed

        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        self.token_emb = token_emb

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std = 0.02)

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output type

        self.output_is_log_prob = mixture_of_softmax

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange('... (k d) -> ... k d', k = mixture_of_softmax_k)
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        # sig softmax

        self.sigsoftmax_logits = sigsoftmax_logits

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), 'can only tie embedding if using `TokenEmbedding`'
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList([LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)])
        else:
            self.to_logits = LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0 and not recycling and self.attn_layers.can_cache_kv
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if hasattr(self.token_emb, 'init_'):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)

    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        return_logit_entropies = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,
    ):

        # if sequence is None, auto create an empty one if `prepend_embeds` was supplied

        if not exists(x):
            assert exists(prepend_embeds)
            x = prepend_embeds.new_empty((prepend_embeds.shape[0], 0), dtype = torch.long)

        # shapes and variables

        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient, orig_mask = x.shape[0], x.shape[1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient, mask

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss
        return_embeddings = return_embeddings | (not exists(self.to_logits))

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, '... -> b ...', b = b)
            x, cls_packed_shape = pack([cls_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value = True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim = -2, value = 0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # attention layers

        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

            # regular

            attended, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        else:
            # recycling

            recycle_steps = default(recycle_steps, (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
            assert exists(recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                    attended, intermediates = self.attn_layers(x + maybe_recycled, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :mem_seq]

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        if exists(self.cls_token):
            x, _ = unpack(x, cls_packed_shape, 'b * d')
            x = x.squeeze(1)  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim = -1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple((rearrange(t, '... 1 -> ...') if t.shape[-1] == 1 else t) for t in cast_tuple(logits))

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in  intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = [cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out

class XTransformer(Module):
    def __init__(
        self,
        *,
        dim,
        tie_token_emb = False,
        ignore_index = -100,
        pad_value = 0,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout  # how many tokens from the encoder to dropout when cross attending from decoder - seen in a couple papers, including Perceiver AR - this will also be very effective regularization when cross attending to very long memories

        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs,
            return_only_embed = True,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim, cross_attend = True, **dec_kwargs)
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=ignore_index, pad_value=pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):

        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        out = self.decoder(tgt, context = enc, context_mask = mask)
        return out
