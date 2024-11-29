from __future__ import annotations

from functools import partial
from typing import Tuple, Callable

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat, pack, unpack

# constants

@dataclass
class Intermediates:
    qk_similarities:    Tensor | None = None
    pre_softmax_attn:   Tensor | None = None
    post_softmax_attn:  Tensor | None = None
    values:             Tensor | None = None
    cached_kv:          Tuple[Tensor, Tensor] | None = None
    layer_type:         str | None = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

def compact(arr):
    return [*filter(exists, arr)]

@torch.jit.script
def softclamp(t: Tensor, value: float):
    return (t / value).tanh() * value

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# selective attention
# https://arxiv.org/abs/2410.02703 - section 3.3
# it is a technique to allow each token to prevent itself from being attended to by future tokens
# if sim_head_gate not supplied, will use the first head of the attention logits (sim in this framework)

def selective_attn(
    sim,
    sim_head_gate = None,
    no_mask_sos = True
):
    i, j, device = *sim.shape[-2:], sim.device
    sim_head_gate = default(sim_head_gate, sim[:, 0])

    gate = F.relu(sim_head_gate) # only positive

    if no_mask_sos:
        gate = gate.clone()
        gate[..., -i] = 0.

    eye = torch.eye(i, device = device)

    if j > i:
        eye = F.pad(eye, (j - i, 0), value = 1.)

    gate = (1. - eye) * gate
    gate = F.pad(gate, (0, 0, 1, -1), value = 0.) # only allow for masking the future
    gate = gate.cumsum(dim = -2)

    return sim - rearrange(gate, 'b i j -> b 1 i j')

# alternative distance functions

def qk_l2_dist_squared(q, k):
    if k.ndim == 3:
        k = repeat(k, 'b j d -> b h j d', h = q.shape[1])

    q, packed_shape = pack_one(q, '* i d')
    k, _ = pack_one(k, '* j d')

    l2_dist_squared = torch.cdist(q, k) ** 2
    return unpack_one(l2_dist_squared, packed_shape, '* i j')

# one-hot straight through softmax

def one_hot_straight_through(logits, temperature = 1.):
    one_hot_indices = logits.argmax(dim = -1, keepdim = True)
    one_hot = torch.zeros_like(logits).scatter(-1, one_hot_indices, 1.)

    soft_attn = (logits / temperature).softmax(dim = -1)
    return one_hot + soft_attn - soft_attn.detach()

# sparse topk attention - only keep topk attn logits for softmax
# optional straight through with masked out logits by setting `attn_sparse_topk_straight_through = True`

def sparse_topk_attn(
    logits,
    sparse_topk,
    temperature = 1.,
    straight_through = False
):
    orig_logits = logits

    mask_value = -torch.finfo(logits.dtype).max
    top_values, _ = logits.topk(sparse_topk, dim = -1)
    sparse_topk_mask = (logits >= top_values[..., -1:]) & (logits > mask_value)
    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    topk_attn = logits.softmax(dim = -1)

    if not straight_through:
        return topk_attn

    soft_attn = (orig_logits / temperature).softmax(dim = -1)
    return topk_attn.detach() + soft_attn - soft_attn.detach()

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        sparse_topk = None,
        sparse_topk_straight_through = False,
        scale = None,
        qk_norm = False,
        l2_distance = False,
        sigmoid = False,
        custom_attn_fn: Callable | None = None,
        flash = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        add_zero_kv = False,
        selective = False,
        hard = False,
        cope = None,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.scale = scale

        # causal related

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        # attention type

        is_sparse_topk_attn = exists(sparse_topk)

        assert not (flash and sigmoid), 'sigmoid attention not available for flash'
        assert not (flash and hard), 'hard attention not available for flash'
        assert not (flash and is_sparse_topk_attn), 'topk attention not available for flash'

        assert at_most_one_of(sigmoid, hard, l2_distance, is_sparse_topk_attn)

        if exists(custom_attn_fn):
            self.attn_fn = custom_attn_fn
        elif sigmoid:
            self.attn_fn = F.sigmoid
        elif hard:
            self.attn_fn = one_hot_straight_through
        elif is_sparse_topk_attn:
            self.attn_fn = partial(sparse_topk_attn, sparse_topk = sparse_topk, straight_through = sparse_topk_straight_through)
        else:
            softmax_fn = partial(F.softmax, dim = -1)
            self.attn_fn = partial(softmax_fn, dtype = torch.float32) if not qk_norm else softmax_fn

        # dropouts

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and (pre_talking_heads or post_talking_heads or pre_scale_post_talking_heads)), 'talking heads not compatible with flash attention'

        self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_talking_heads else None
        self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if post_talking_heads else None
        self.pre_scale_post_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_scale_post_talking_heads else None

        if exists(self.pre_softmax_talking_heads):
            nn.init.dirac_(self.pre_softmax_talking_heads.weight)

        if exists(self.post_softmax_talking_heads):
            nn.init.dirac_(self.post_softmax_talking_heads.weight)

        if exists(self.pre_scale_post_talking_heads):
            # an improvisation where heads are combined pre-softmax attention, then used to scale post-softmax attention
            nn.init.dirac_(self.pre_scale_post_talking_heads.weight)

        # selective attention

        assert not (flash and selective), 'selective attention cannot work on flash attention'
        assert not (selective and not causal), 'selective attention is designed for autoregressive'
        self.selective = selective

        # l2 distance attention

        self.l2_distance = l2_distance

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # soft clamp attention logit value

        if softclamp_logits:
            assert not flash, 'flash attention not compatible with logit softclamp value yet'
            assert logit_softclamp_value > 0.

        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

        # contextual positional encoding

        self.cope = cope

        # flash attention

        self.flash = flash

        torch_version = version.parse(torch.__version__)
        assert not (flash and torch_version < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # torch 2.3 uses new backend and context manager

        if torch_version >= version.parse('2.3'):
            from torch.nn.attention import SDPBackend

            str_to_backend = dict(
                enable_flash = SDPBackend.FLASH_ATTENTION,
                enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
                enable_math = SDPBackend.MATH,
                enable_cudnn = SDPBackend.CUDNN_ATTENTION
            )

            sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

            self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
        else:
            self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # handle maybe l2 distance

        if self.l2_distance:
            k_norm_sq = k.norm(dim = -1, keepdim = True) ** 2
            k = F.pad(k, (0, 1), value = -1.)
            k = torch.cat((k, k_norm_sq), dim = -1)

            q_norm_sq = q.norm(dim = -1, keepdim = True) ** 2
            q = torch.cat((2 * q, q_norm_sq), dim = -1)
            q = F.pad(q, (0, 1), value = -1.)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask
            causal = False

        # protect against an entire row being masked out

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = attn_bias.expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle key padding mask

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = tuple(rearrange(t, 'b 1 n d -> b n d') for t in (k, v))
        elif kv_heads < heads:
            k, v = tuple(repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads) for t in (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value = 0.) for t in (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if not self.l2_distance:
            sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
        else:
            sim = -qk_l2_dist_squared(q, k)

        sim = sim * scale

        if exists(prev_attn):
            sim = sim + prev_attn

        qk_similarities = sim.clone()

        if exists(self.pre_scale_post_talking_heads):
            pre_to_post_scale = self.pre_scale_post_talking_heads(sim)

        if exists(self.pre_softmax_talking_heads):
            sim = sim + self.pre_softmax_talking_heads(sim)

        if exists(attn_bias):
            sim = sim + attn_bias

        if self.softclamp_logits:
            sim = softclamp(sim, self.logit_softclamp_value)

        i, j, dtype = *sim.shape[-2:], sim.dtype

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            sim = sim.masked_fill(causal_mask, mask_value)

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        if exists(self.cope):
            sim = sim + self.cope(q, sim)

        if self.selective:
            sim = selective_attn(sim)

        pre_softmax_attn = sim

        attn = self.attn_fn(sim)

        attn = attn.type(dtype)

        post_softmax_attn = attn

        attn = self.attn_dropout(attn)

        if exists(self.post_softmax_talking_heads):
            attn = self.post_softmax_talking_heads(attn)

        if exists(self.pre_scale_post_talking_heads):
            attn = attn * pre_to_post_scale

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, intermediates
