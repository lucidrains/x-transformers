import torch
from torch import nn
import torch.nn.functional as F

from einops import pack, repeat, unpack

from x_transformers.x_transformers import (
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding,
    always,
    pad_at_dim
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class ContinuousTransformerWrapper(nn.Module):
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
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # project in and out

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
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
        batch = x.shape[0]

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

        out = self.project_out(x) if not return_embeddings else x

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

class ContinuousAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net: ContinuousTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        loss_fn = nn.MSELoss(reduction = 'none')
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.loss_fn = loss_fn

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        if num_dims == 2:
            start_tokens = start_tokens[None, :]        

        b, t, _, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            last = self.net(x, **kwargs)[:, -1:]
            out = torch.cat((out, last), dim = -2)

        out = out[:, t:]

        if num_dims == 2:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        inp, target = x[:, :-1], x[:, 1:]

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(inp, **kwargs)

        loss = self.loss_fn(out, target)

        if exists(mask):
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'
            loss = loss[mask]

        return loss.mean()
