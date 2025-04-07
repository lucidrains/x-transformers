from collections import namedtuple

import torch
from torch import nn, tensor, pi, is_tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, einsum, pack, unpack

from x_transformers.x_transformers import (
    Encoder
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# random fourier

class RandomFourierEmbed(Module):

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):

        times = rearrange(times, '... -> ... 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)

# class

class NeoMLP(Module):
    """ https://openreview.net/forum?id=A8Vuf2e8y6 """
    """ https://haian-jin.github.io/projects/LVSM/ """

    def __init__(
        self,
        *,
        dim_in,
        dim_hidden,
        dim_out,
        dim_model,
        depth,
        encoder_kwargs: dict = dict(
            attn_dim_head = 16,
            heads = 4
        )
    ):
        super().__init__()

        # input and output embeddings

        self.input_embed = nn.Parameter(torch.zeros(dim_in, dim_model))
        self.hidden_embed = nn.Parameter(torch.zeros(dim_hidden, dim_model))
        self.output_embed = nn.Parameter(torch.zeros(dim_out, dim_model))

        nn.init.normal_(self.input_embed, std = 0.02)
        nn.init.normal_(self.hidden_embed, std = 0.02)
        nn.init.normal_(self.output_embed, std = 0.02)

        # they use random fourier for continuous features

        self.random_fourier = nn.Sequential(
            RandomFourierEmbed(dim_model),
            nn.Linear(dim_model, dim_model)
        )

        # hidden dimensions of mlp replaced with nodes with message passing
        # which comes back to self attention as a fully connected graph.

        self.transformer = Encoder(
            dim = dim_model,
            depth = depth,
            **encoder_kwargs
        )

        # output

        self.to_output_weights = nn.Parameter(torch.randn(dim_out, dim_model))
        self.to_output_bias = nn.Parameter(torch.zeros(dim_out))

    def forward(
        self,
        x,
        return_embeds = False
    ):
        no_batch = x.ndim == 1

        if no_batch:
            x = rearrange(x, '... -> 1 ...')

        batch = x.shape[0]

        fouriered_input = self.random_fourier(x)

        # add fouriered input to the input embedding

        input_embed = fouriered_input + self.input_embed

        hidden_embed, output_embed = tuple(repeat(t, '... -> b ...', b = batch) for t in (self.hidden_embed, self.output_embed))

        # pack all the inputs into one string of tokens for self attention

        embed, packed_shape = pack([input_embed, hidden_embed, output_embed], 'b * d')

        # attention is all you need

        embed = self.transformer(embed)

        # unpack

        input_embed, hidden_embed, output_embed = unpack(embed, packed_shape, 'b * d')

        # project for output

        output = einsum(output_embed, self.to_output_weights, 'b n d, n d -> b n')
        output = output + self.to_output_bias

        if no_batch:
            output = rearrange(output, '1 ... -> ...')

        if not return_embeds:
            return output

        return output, (input_embed, hidden_embed, output_embed)
