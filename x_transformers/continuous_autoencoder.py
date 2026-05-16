from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module

from x_transformers.x_transformers import Encoder, Decoder
from x_transformers.continuous import ContinuousTransformerWrapper

from einops.layers.torch import Rearrange

from torch_einops_utils import masked_mean, lens_to_mask

# rl token (continuous autoencoder)
# https://www.pi.website/research/rlt

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# bottleneck modules
# forward returns (latents, aux_loss | None)

class VariationalLatentBottleneck(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        kl_div_floor = 0.
    ):
        super().__init__()
        self.kl_div_floor = kl_div_floor

        self.to_mean_log_var = nn.Sequential(
            nn.Linear(dim, dim_latent * 2),
            Rearrange('b (two d) -> two b d', two = 2)
        )

    def forward(self, x):
        mean, log_var = self.to_mean_log_var(x)
        std = (0.5 * log_var).exp()

        latents = mean + std * torch.randn_like(mean)

        kl_loss = 0.5 * (
            log_var.exp() + mean.square() - log_var - 1.
        )

        kl_loss = F.relu(kl_loss - self.kl_div_floor)
        kl_loss = kl_loss.sum(dim = -1).mean()

        return latents, kl_loss

class DeterministicLatentBottleneck(Module):
    def __init__(
        self,
        dim,
        dim_latent,
        activation = None
    ):
        super().__init__()
        self.proj = nn.Linear(dim, dim_latent)
        self.activation = default(activation, nn.Identity())

    def forward(self, x):
        latents = self.activation(self.proj(x))
        return latents, None

# main class

class ContinuousTransformerAutoencoder(Module):
    def __init__(
        self,
        *,
        dim,
        enc_depth,
        dec_depth,
        max_seq_len,
        dim_latent = None,
        bottleneck: Module | None = None,
        bottleneck_type: Literal['deterministic', 'variational'] = 'deterministic',
        deterministic_bottleneck_kwargs: dict = dict(),
        variational_bottleneck_kwargs: dict = dict(),
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        latents_dropout_prob = 0.5,
        **kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.encoder = ContinuousTransformerWrapper(
            dim_in = dim,
            max_seq_len = max_seq_len,
            attn_layers = Encoder(
                dim = dim,
                depth = enc_depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **enc_kwargs
            )
        )

        if exists(bottleneck):
            self.bottleneck = bottleneck
        elif bottleneck_type == 'deterministic':
            self.bottleneck = DeterministicLatentBottleneck(dim, dim_latent, **deterministic_bottleneck_kwargs)
        elif bottleneck_type == 'variational':
            self.bottleneck = VariationalLatentBottleneck(dim, dim_latent, **variational_bottleneck_kwargs)
        else:
            raise ValueError(f'unknown bottleneck type {bottleneck_type}')

        self.from_latent_to_prepend_token = nn.Sequential(
            nn.Linear(dim_latent, dim),
            Rearrange('b d -> b 1 d')
        )

        self.decoder = ContinuousTransformerWrapper(
            dim_in = dim,
            dim_out = dim,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = dec_depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **dec_kwargs
            )
        )

        self.latents_dropout = nn.Dropout(latents_dropout_prob)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, seq, lens = None, return_aux_loss = False):
        mask = lens_to_mask(lens, max_len = seq.shape[1]) if exists(lens) else None

        encoded = self.encoder(seq, mask = mask, return_embeddings = True)

        pooled = masked_mean(encoded, mask, dim = 1)

        latents, aux_loss = self.bottleneck(pooled)

        if not return_aux_loss:
            return latents

        return latents, aux_loss

    def forward(
        self,
        seq,
        lens = None,
        return_all_losses = False
    ):
        batch, seq_len, device = *seq.shape[:2], seq.device

        latents, aux_loss = self.encode(seq, lens = lens, return_aux_loss = True)

        # latent dropout

        dropped_latents = ~self.latents_dropout(torch.ones((batch,), device = device)).bool()

        prepend_embeds = self.from_latent_to_prepend_token(latents)

        recon = self.decoder(
            seq,
            prepend_embeds = prepend_embeds,
            seq_start_pos = dropped_latents.long()
        )

        # slice out the prepended latent position

        recon = recon[:, 1:]

        # reconstruction loss

        recon_loss = F.mse_loss(recon, seq, reduction = 'none')

        mask = lens_to_mask(lens, max_len = seq_len) if exists(lens) else None

        recon_loss = masked_mean(recon_loss, mask, dim = 1).mean()

        # total loss

        total_loss = recon_loss

        if exists(aux_loss):
            total_loss = total_loss + aux_loss

        if not return_all_losses:
            return total_loss

        return total_loss, (recon_loss, aux_loss)
