from __future__ import annotations

# Causal Sequence Conditioners
# Generalized from Latent Lattice Language Models - Duggal et al. https://arxiv.org/abs/2611.21608
# Paired with depthwise-separable conv conditioner (default kernel_size = 4)

import math
from copy import deepcopy
from itertools import chain
from functools import partial, wraps

import torch
from torch import nn, cat, is_tensor
from torch.amp import autocast
from torch.nn import Module, ModuleList, RMSNorm
import torch.nn.functional as F

from assoc_scan import AssocScan

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch_einops_utils import (
    slice_right_at_dim,
    temp_eval
)
from torch_einops_utils.shape import size

LinearNoBias = partial(nn.Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    if exists(v):
        return v
    return d() if callable(d) else d

def first(arr):
    return arr[0]

def cast_tuple(v, length = 1):
    return v if isinstance(v, tuple) else (v,) * length

def divisible_by(num, den):
    return (num % den) == 0

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def disable_autocast(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        tensor_arg = next(arg for arg in (*args, *kwargs.values()) if is_tensor(arg))

        with autocast(tensor_arg.device.type, enabled = False):
            return fn(*args, **kwargs)

    return inner

# scale and offset modulation (FiLM)

@disable_autocast
def apply_scale_offset(t, scale, offset):
    n = size(offset, '... [n] d')

    if size(t, '... [n] d') > n:
        mem, t = t[..., :-n, :], t[..., -n:, :]
        return cat((mem, apply_scale_offset(t, scale, offset)), dim = -2)

    if not exists(scale):
        return (t.float() + offset.float()).to(t.dtype)

    return torch.addcmul(offset.float(), t.float(), 1. + scale.float()).to(t.dtype)

class GroupedFiLM(Module):
    def __init__(
        self,
        dim_cond,
        target_dims: int | tuple[int, ...],
        scale_groups = None
    ):
        super().__init__()
        self.single_target = isinstance(target_dims, int)
        target_dims = cast_tuple(target_dims)
        scale_groups = cast_tuple(scale_groups if bool(scale_groups) else 0, len(target_dims))

        assert all(g >= 0 and (g == 0 or divisible_by(d, g)) for d, g in zip(target_dims, scale_groups))

        self.target_dims = target_dims
        self.scale_groups = scale_groups
        self.split_dims = tuple(dim for g, d in zip(scale_groups, target_dims) for dim in ((g, d) if g > 0 else (d,)))

        self.to_scale_offset = LinearNoBias(dim_cond, sum(self.split_dims))
        init_zero_(self.to_scale_offset)

    def forward(self, cond):
        projected = self.to_scale_offset(cond).float()
        chunks = iter(projected.split(self.split_dims, dim = -1))
        out = []

        for d, g in zip(self.target_dims, self.scale_groups):
            raw_scale = next(chunks) if g > 0 else None
            offset = next(chunks)
            scale = repeat(raw_scale.tanh(), 'b n g -> b n (g r)', r = d // g) if exists(raw_scale) else None
            out.append((scale, offset))

        return first(out) if self.single_target else tuple(out)

# causal sequence conditioners
# contract: forward(x, mem = None) -> (cond, next_mem)

class DSConvConditioner(Module):
    def __init__(
        self,
        dim,
        dim_cond = None,
        kernel_size = 4,
        dim_cond_frac = 0.25,
        min_dim_cond = 16
    ):
        super().__init__()
        assert kernel_size >= 1

        dim_cond = default(dim_cond, max(min_dim_cond, int(dim * dim_cond_frac)))
        self.dim_cond = dim_cond
        self.mem_len = kernel_size - 1

        self.to_in = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_cond),
            Rearrange('b n d -> b d n')
        )

        self.dw_conv = nn.Conv1d(dim_cond, dim_cond, kernel_size, groups = dim_cond)

        self.to_out = nn.Sequential(
            Rearrange('b d n -> b n d'),
            LinearNoBias(dim_cond, dim_cond),
            nn.SiLU(),
            RMSNorm(dim_cond)
        )

    def forward(
        self,
        x,
        mem = None
    ):
        h = self.to_in(x)

        window = cat((mem, h), dim = -1) if exists(mem) else F.pad(h, (self.mem_len, 0))
        next_mem = slice_right_at_dim(window, self.mem_len, dim = -1)

        cond = self.to_out(self.dw_conv(window))
        return cond, next_mem

class EMAConditioner(Module):
    def __init__(
        self,
        dim,
        dim_cond = None,
        use_accelerated_scan = False,
        dt_min = 1e-3,
        dt_max = 1e-1,
        a_init_range = (1., 16.),
        init_state_std = 0.02,
        dim_cond_frac = 0.25,
        min_dim_cond = 16
    ):
        super().__init__()
        dim_cond = default(dim_cond, max(min_dim_cond, int(dim * dim_cond_frac)))
        self.dim_cond = dim_cond

        self.norm_in = RMSNorm(dim)
        self.to_dt = nn.Linear(dim, dim_cond)
        self.to_cand = nn.Linear(dim, dim_cond)

        nn.init.normal_(self.to_dt.weight, std = 0.02)
        nn.init.normal_(self.to_cand.weight, std = 0.02)

        dt = torch.exp(torch.empty(dim_cond).uniform_(math.log(dt_min), math.log(dt_max)))
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            self.to_dt.bias.copy_(inv_softplus_dt)
            self.to_cand.bias.zero_()

        a_lo, a_hi = a_init_range
        self.a_log = nn.Parameter(torch.empty(dim_cond).uniform_(a_lo, a_hi).log())
        self.init_state = nn.Parameter(torch.randn(dim_cond) * init_state_std)

        self.scan = AssocScan(use_accelerated = use_accelerated_scan)
        self.norm_out = RMSNorm(dim_cond)

    @disable_autocast
    def scan_ema(self, dt_pre, cand_pre, prev_state):
        dt = F.softplus(dt_pre.float())
        decay = (-self.a_log.float().exp() * dt).exp()
        weighted_cand = (1. - decay) * F.silu(cand_pre.float())
        prev_state = prev_state.float()

        if size(dt_pre, 'b [n] d') == 1:
            return decay * rearrange(prev_state, 'b d -> b 1 d') + weighted_cand

        return self.scan(decay, weighted_cand, prev = prev_state)

    def forward(
        self,
        x,
        mem = None
    ):
        batch, dtype = size(x, '[b] n d'), x.dtype
        normed = self.norm_in(x)

        dt_pre = self.to_dt(normed)
        cand_pre = self.to_cand(normed)
        prev_state = default(mem, lambda: repeat(self.init_state, 'd -> b d', b = batch))

        states = self.scan_ema(dt_pre, cand_pre, prev_state)
        next_mem = states[:, -1]

        cond = self.norm_out(states.to(dtype))
        return cond, next_mem

# decoder conditioner router

def should_trigger(ind, every = 1):
    every = default(every, 1)

    if isinstance(every, int):
        assert every >= 1
        return divisible_by(ind, every)

    return ind in every

def get_dim_cond(module: Module, dim: int):
    if hasattr(module, 'dim_cond'):
        return module.dim_cond

    param = next(chain(module.parameters(), module.buffers()), None)
    device = param.device if exists(param) else None
    dtype = param.dtype if (exists(param) and param.is_floating_point()) else None

    with temp_eval(module), torch.no_grad():
        cond_out = module(torch.zeros((1, 1, dim), device = device, dtype = dtype), None)

    assert isinstance(cond_out, tuple) and len(cond_out) == 2, 'sequence conditioner must obey `(x, mem) -> (cond, next_mem)`'
    cond, _ = cond_out
    return size(cond, 'b n [d]')

def build_sublayer_film(layer_type, block, dim_cond, scale_groups):
    g = scale_groups if bool(scale_groups) else 0

    if layer_type == 'a':
        attn = next(m for m in block.modules() if hasattr(m, 'to_q'))
        q_dim, k_dim, v_dim = attn.to_q.out_features, attn.to_k.out_features, attn.to_v.out_features
        k_heads = k_dim // (q_dim // attn.heads)
        groups = (attn.heads * g, k_heads * g, attn.kv_heads * g) if g > 0 else None
        return GroupedFiLM(dim_cond, (q_dim, k_dim, v_dim), scale_groups = groups)

    ff = next(m for m in block.modules() if hasattr(m, 'ff'))
    proj_in = ff.ff[0]
    is_glu = hasattr(proj_in, 'proj')
    target_dim = proj_in.proj.out_features if is_glu else proj_in[0].out_features
    groups = g * (2 if is_glu else 1) if g > 0 else None
    return GroupedFiLM(dim_cond, target_dim, scale_groups = groups)

class SeqCondRouter(Module):
    def __init__(
        self,
        *,
        layers: ModuleList,
        layer_types: tuple[str, ...],
        layer_dims: tuple[int, ...],
        len_default_block: int,
        seq_cond: Module | tuple[Module | None, Module | None] | None = None,
        attn_seq_cond: Module | None = None,
        ff_seq_cond: Module | None = None,
        every = 1,
        attn_every = None,
        ff_every = None,
        scale_groups = 4,
        attn_scale_groups = None,
        ff_scale_groups = None
    ):
        super().__init__()

        if isinstance(seq_cond, tuple):
            assert len(seq_cond) == 2 and not (exists(attn_seq_cond) or exists(ff_seq_cond))
            attn_seq_cond, ff_seq_cond = seq_cond
            seq_cond = None

        is_shared = exists(seq_cond)
        assert not (is_shared and (exists(attn_seq_cond) or exists(ff_seq_cond))), 'pass either shared `seq_cond` or per-sublayer `attn_seq_cond` / `ff_seq_cond`'
        assert not (exists(attn_seq_cond) and 'a' not in layer_types), 'no self-attention layers available for `attn_seq_cond`'
        assert not (exists(ff_seq_cond) and 'f' not in layer_types), 'no feedforward layers available for `ff_seq_cond`'

        attn_every = default(attn_every, every)
        ff_every = default(ff_every, every)

        default_attn_groups, default_ff_groups = scale_groups if isinstance(scale_groups, tuple) else (scale_groups, None)
        attn_groups = default(attn_scale_groups, default_attn_groups)
        ff_groups = default(ff_scale_groups, default_ff_groups)

        conds, films = [], []
        attn_dim_cond = ff_dim_cond = None
        attn_ind = ff_ind = 0

        def clone_cond(proto, dim):
            assert isinstance(proto, Module), 'sequence conditioner must be an `nn.Module` obeying `(x, mem) -> (cond, next_mem)`'
            cloned = deepcopy(proto)
            return cloned, get_dim_cond(cloned, dim)

        for ind, (layer_type, (_, block, _), dim) in enumerate(zip(layer_types, layers, layer_dims)):
            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            cond_fn = film = None

            if is_shared and block_begin and should_trigger(block_ind, every):
                cond_fn, shared_dim = clone_cond(seq_cond, dim)
                attn_dim_cond = ff_dim_cond = shared_dim

            if layer_type == 'a':
                if exists(attn_seq_cond) and should_trigger(attn_ind, attn_every):
                    cond_fn, attn_dim_cond = clone_cond(attn_seq_cond, dim)

                if exists(attn_dim_cond):
                    film = build_sublayer_film('a', block, attn_dim_cond, attn_groups)

                attn_ind += 1

            elif layer_type == 'f':
                if exists(ff_seq_cond) and should_trigger(ff_ind, ff_every):
                    cond_fn, ff_dim_cond = clone_cond(ff_seq_cond, dim)

                if exists(ff_dim_cond):
                    film = build_sublayer_film('f', block, ff_dim_cond, ff_groups)

                ff_ind += 1

            conds.append(cond_fn)
            films.append(film)

        self.is_shared = is_shared
        self.layer_types = layer_types
        self.conds = ModuleList(conds)
        self.films = ModuleList(films)

    def init_forward(self, cache = None):
        iter_cache = iter(default(cache.seq_cond_caches if exists(cache) else None, ()))
        next_caches = []
        attn_cond = ff_cond = None

        def step(ind, x):
            nonlocal attn_cond, ff_cond

            layer_type, cond_fn, film = self.layer_types[ind], self.conds[ind], self.films[ind]

            if exists(cond_fn):
                cond, mem = cond_fn(x, next(iter_cache, None))
                next_caches.append(mem)

                if self.is_shared:
                    attn_cond = ff_cond = cond
                elif layer_type == 'a':
                    attn_cond = cond
                else:
                    ff_cond = cond

            if not exists(film):
                return None

            return film(attn_cond if layer_type == 'a' else ff_cond)

        return step, next_caches
