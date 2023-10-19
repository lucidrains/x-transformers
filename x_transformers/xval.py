"""
regular transformer with discrete tokens, but continuous for number
generalizes better for arithmetic
https://arxiv.org/abs/2310.02989
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable
from collections import namedtuple

from einops import rearrange
from einops.layers.torch import Rearrange

from x_transformers.x_transformers import (
    AttentionLayers,
    TokenEmbedding,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding
)

from x_transformers.autoregressive_wrapper import (
    top_k,
    top_p
)

# constants

LossBreakdown = namedtuple('LossBreakdown', ['cross_entropy_loss', 'numerical_mse_loss'])

GenerateReturn = namedtuple('GenerateReturn', ['sampled_token_ids', 'sampled_numbers', 'is_number_mask'])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class XValTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        numerical_token_id,
        attn_layers: AttentionLayers,
        emb_dim = None,
        logits_dim = None,
        tie_embedding = False,
        max_mem_len = 0,
        num_memory_tokens = None,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.emb_dim = emb_dim
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)

        self.numerical_token_id = numerical_token_id

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # to logits

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        self.to_numerical_output = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):
        assert x.shape == x_num.shape

        batch = x.shape[0]

        is_number_mask = x == self.numerical_token_id

        x = self.token_emb(x)

        scale = torch.where(is_number_mask, x_num, 1.)
        scale = rearrange(scale, '... -> ... 1')

        x = x * scale

        x = x + self.pos_emb(x, pos = pos)

        # memory tokens

        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        x = self.emb_dropout(x)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        # splice out memory tokens

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        if not return_embeddings:
            logits = self.to_logits(x)
            numerical_pred = self.to_numerical_output(x)
            out = (logits, numerical_pred)
        else:
            out = x

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class XValAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net: XValTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        numerical_loss_weight = 1.
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.numerical_loss_weight = numerical_loss_weight
        self.ignore_index = ignore_index

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        start_numbers: Tensor,
        seq_len,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        **kwargs
    ):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        assert start_tokens.shape == start_numbers.shape

        b, t, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens
        num_out = start_numbers

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            x_num = num_out[:, -self.max_seq_len:]

            logits, numerical_pred = self.net(x, x_num, **kwargs)

            last_logits = logits[:, -1]
            last_num_pred = numerical_pred[:, -1:]

            filtered_logits = filter_logits_fn(last_logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim = -1)
            num_out = torch.cat((num_out, last_num_pred), dim = -1)

        out = out[:, t:]
        num_out = num_out[:, t:]

        is_number = out == self.net.numerical_token_id
        num_out = torch.where(is_number, num_out, float('nan'))

        self.net.train(was_training)
        return GenerateReturn(out, num_out, is_number)

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_loss_breakdown = False,
        **kwargs
    ):
        inp, target = x[:, :-1], x[:, 1:]
        x_num_inp, x_num_target = x_num[:, :-1], x_num[:, 1:]

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        logits, numerical_pred = self.net(inp, x_num_inp, **kwargs)

        logits = rearrange(logits, 'b n c -> b c n')

        cross_entropy_loss = F.cross_entropy(logits, target, reduction = 'none', ignore_index = self.ignore_index)

        target_mask = target != self.ignore_index

        numerical_mse_loss = F.mse_loss(numerical_pred, x_num_target, reduction = 'none')

        numerical_mse_loss = numerical_mse_loss * target_mask

        loss = cross_entropy_loss + numerical_mse_loss * self.numerical_loss_weight

        if exists(mask):
            loss = loss[mask]

        loss = loss.mean()

        if not return_loss_breakdown:
            return loss

        return loss, LossBreakdown(cross_entropy_loss, numerical_mse_loss)
