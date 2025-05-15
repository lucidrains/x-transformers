from __future__ import annotations

import torch
from torch import nn, cat, stack
from torch.nn import Module
import torch.nn.functional as F
from torch.distributions import Normal

import einx
from einops import rearrange, reduce, pack, repeat, unpack

from x_transformers.x_transformers import (
    Attention,
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    LayerNorm,
    masked_mean,
    always,
    pad_at_dim
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if not isinstance(d, Module) and callable(d) else d

def masked_mean(t, mask):
    t = einx.where('b n, b n d, -> b n d', mask, t, 0.)

    num = reduce(t, 'b n d -> b', 'sum')
    den = mask.sum(dim = -1)

    masked_average = num / den.clamp(min = 1.)
    return masked_average

# probabilistic loss fn

class GaussianNLL(Module):
    def forward(self, pred, target):
        mean, var = pred
        dist = Normal(mean, var)
        return -dist.log_prob(target)

# main classes

class ContinuousTransformerWrapper(Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        average_pool_embed = False,
        probabilistic = False
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len
        
        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # average pool

        self.average_pool_embed = average_pool_embed

        # project in and out

        self.project_in = nn.Linear(dim_in, dim, bias = False) if exists(dim_in) else nn.Identity()

        # output is multipled by 2 for outputting mean and log variance

        self.probabilistic = probabilistic

        self.project_out = nn.Linear(dim, dim_out * (2 if probabilistic else 1), bias = False) if exists(dim_out) else nn.Identity()

        # can cache kv

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention)])

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        lens = None,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        **kwargs
    ):
        batch, seq, orig_mask, device = *x.shape[:2], mask, x.device

        # maybe seq lengths passed in

        if exists(lens):
            assert not exists(mask), 'either `mask` or `lens` passed in, but not both'
            seq_arange = torch.arange(seq, device = device)

            mask = einx.less('j, i -> i j', seq_arange, lens)

        # project in + positional embedding

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        # memory tokens

        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((batch, seq), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((batch, prepend_seq), device = device, dtype = torch.bool))

                mask = cat((prepend_mask, mask), dim = -1)

        x = self.emb_dropout(x)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, return_hiddens = True, **kwargs)

        # splice out memory tokens

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask)

        # maybe linear project out

        out = self.project_out(x) if not return_embeddings else x

        if not return_embeddings and self.probabilistic:
            mean, log_var = rearrange(out, '... (d mean_log_var) -> mean_log_var ... d', mean_log_var = 2)
            variance = log_var.exp()
            out = stack((mean, variance))

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = tuple(t[..., -self.max_mem_len:, :].detach() for t in hiddens)
            return out, new_mems

        if return_attn:
            attn_maps = tuple(t.post_softmax_attn for t in intermediates.attn_intermediates)
            return out, attn_maps

        return out

class ContinuousAutoregressiveWrapper(Module):
    def __init__(
        self,
        net: ContinuousTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        loss_fn: Module | None = None,
        equal_loss_weight_batch = False  # setting this to True, if the mask is passed in and sequences are variable in length, each sequence will be weighted the same (as opposed to each token)
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len

        probabilistic = net.probabilistic
        self.probabilistic = probabilistic

        loss_fn = default(loss_fn, nn.MSELoss(reduction = 'none') if not probabilistic else GaussianNLL())

        self.loss_fn = loss_fn
        self.equal_loss_weight_batch = equal_loss_weight_batch

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        temperature = 1.,
        cache_kv = True,
        **kwargs
    ):
        should_cache_kv = cache_kv and self.net.can_cache_kv
        device = start_tokens.device

        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        if num_dims == 2:
            start_tokens = start_tokens[None, :]        

        b, t, _, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens

        cache = None

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            net_out, new_cache = self.net(x, cache = cache, return_intermediates = True, **kwargs)

            last_output = net_out[..., -1:, :]

            if self.probabilistic:
                mean, var = last_output
                last_output = torch.normal(mean, var * temperature)

            out = cat((out, last_output), dim = -2)

            if should_cache_kv:
                cache = new_cache

        out = out[:, t:]

        if num_dims == 2:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(
        self,
        x,
        **kwargs
    ):
        inp, target = x[:, :-1], x[:, 1:]

        assert 'prepend_embeds' not in kwargs

        # lens

        lens = kwargs.pop('lens', None)

        if exists(lens):
            assert 'mask' not in kwargs, 'either `mask` or `lens` passed in, but not both'
            seq_len, device = inp.shape[1], inp.device
            seq_arange = torch.arange(seq_len, device = device)
            mask = einx.less('j, i -> i j', seq_arange, lens)

            kwargs['mask'] = mask

        # mask

        mask = kwargs.get('mask', None)

        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(inp, **kwargs)

        loss = self.loss_fn(out, target)

        if exists(mask):
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'

            if self.equal_loss_weight_batch:
                loss = masked_mean(loss, mask)
            else:
                loss = loss[mask]

        return loss.mean()
