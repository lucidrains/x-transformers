from functools import partial

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

@dataclass
class Intermediates:
    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

# main class

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        talking_heads = False,
        scale = None,
        qk_norm = False,
        flash = False,
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm
        self.causal = causal
        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and talking_heads), 'talking heads not compatible with flash attention'

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

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
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (default_scale / self.scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = torch.ones((q_len, k_len), dtype = torch.bool, device = device).triu(k_len - q_len + 1)
                mask = mask | causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, -1, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(mask, mask_value // 2)
            elif causal:
                causal_mask = torch.ones((q_len, k_len), dtype = torch.bool, device = device).triu(k_len - q_len + 1)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

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

        n, device = q.shape[-2], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(attn_bias):
            dots = dots + attn_bias

        dtype = dots.dtype
        pre_softmax_attn = dots.clone()

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(mask, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            dots = dots.masked_fill(causal_mask, mask_value)

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return out, intermediates
