from __future__ import annotations

# applying the cvae + detr design from ACT (Zhou et al.) to GPT
# for steering, diversity rlvr, map-elites in epo, and other possibilities

import torch
from torch import nn, Tensor, is_tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers.x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper
)

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class GPTVAE(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        enc_depth,
        max_seq_len,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vae_kl_loss_weight = 1.,
        vae_kl_div_floor = 0.,      # what was done in free transformer, which in turn came from Kingma 2016
        latents_dropout_prob = 0.5, # what percentage of the time to dropout the latents completely
        pad_id = -1,
        encoder: Module | None = None,
        **kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        if not exists(encoder):
            encoder = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len + 1,
                return_only_embed = True,
                average_pool_embed = True,
                attn_layers = Encoder(
                    dim = dim,
                    depth = enc_depth,
                    attn_dim_head = attn_dim_head,
                    heads = heads,
                    **kwargs,
                    **enc_kwargs
                ),
            )

        self.encoder = encoder

        self.to_latent_mean_log_variance = nn.Sequential(
            nn.Linear(dim, dim_latent * 2),
            Rearrange('b (two d) -> two b d', two = 2)
        )

        self.from_latent_to_prepend_token = nn.Sequential(
            nn.Linear(dim_latent, dim),
            Rearrange('b d -> b 1 d')
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **dec_kwargs
            ),
        )

        self.ar_wrapped_decoder = AutoregressiveWrapper(self.decoder, ignore_index = pad_id)

        self.pad_id = pad_id

        # loss weights - vae kl loss

        self.vae_kl_div_floor = vae_kl_div_floor
        self.vae_kl_loss_weight = vae_kl_loss_weight

        self.latents_dropout = nn.Dropout(latents_dropout_prob)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        seq,
        return_mean_log_var = False
    ):
        mask = seq != self.pad_id
        pooled = self.encoder(seq, mask = mask)

        latents_mean, latents_log_var = self.to_latent_mean_log_variance(pooled)
        latents_std = (0.5 * latents_log_var).exp()

        # reparam trick

        latents = latents_mean + latents_std * torch.randn_like(latents_mean)

        if not return_mean_log_var:
            return latents

        return latents, (latents_mean, latents_log_var)

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        latents = None,
        seq_for_latents = None,
        **generate_kwargs
    ):
        assert prompts.ndim in {1, 2}
        batch = prompts.shape[0] if prompts.ndim == 2 else 1

        # if seq_for_latents passed in, derive latents from it

        if exists(seq_for_latents):
            assert not exists(latents), 'latents should not be passed in if given the seq from which to derive them'

            latents = self.encode_to_latents(seq_for_latents)

        # prepend embeds

        prepend_embeds = None
        if exists(latents):
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            if latents.ndim == 1: # repeat latents
                latents = repeat(latents, 'd -> b d', b = batch)

            prepend_embeds = self.from_latent_to_prepend_token(latents)

        # generated

        generated = self.ar_wrapped_decoder.generate(
            prompts,
            seq_len,
            prepend_embeds = prepend_embeds,
            **generate_kwargs
        )

        return generated

    def forward(
        self,
        seq,
        seq_for_latents = None,
        return_all_losses = False
    ):
        batch, device = seq.shape[0], seq.device

        seq_for_latents = default(seq_for_latents, seq)

        latents, (latents_mean, latents_log_var) = self.encode_to_latents(seq_for_latents, return_mean_log_var = True)

        dropped_latents = ~self.latents_dropout(torch.ones((batch,), device = device)).bool()

        prepend_embeds = self.from_latent_to_prepend_token(latents)

        ar_loss = self.ar_wrapped_decoder(
            seq,
            prepend_embeds = prepend_embeds,
            seq_start_pos = dropped_latents.long() # sequence starts at 1 and does not attend to the first style latent
        )

        # vae kl loss

        vae_kl_loss = 0.5 * (
            latents_log_var.exp()
            + latents_mean.square()
            - latents_log_var
            - 1.
        )

        vae_kl_loss = F.relu(vae_kl_loss - self.vae_kl_div_floor)

        vae_kl_loss = vae_kl_loss.sum(dim = -1).mean()

        # return losses

        total_loss = (
            ar_loss +
            vae_kl_loss * self.vae_kl_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (ar_loss, vae_kl_loss)

        return total_loss, losses
