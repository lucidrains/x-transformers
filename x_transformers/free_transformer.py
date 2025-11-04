from __future__ import annotations

# https://arxiv.org/abs/2510.17558
# François Fleuret
# https://www.youtube.com/watch?v=Nao16-6l6dQ

import torch
from torch import nn, Tensor, is_tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from x_transformers.x_transformers import (
    Encoder,
    Decoder,
    TransformerWrapper
)

from x_transformers.autoregressive_wrapper import (
    gumbel_sample,
    top_p,
    top_k
)

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

# classes

class FreeTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dec_head_depth,
        dec_tail_depth,
        enc_depth,
        max_seq_len,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vae_kl_loss_weight = 1.,
        pad_id = -1,
        encoder: Module | None = None,
        **kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.token_unembed = nn.Linear(dim, num_tokens, bias = False)

        if not exists(encoder):
            encoder = Encoder(
                dim = dim,
                depth = enc_depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **enc_kwargs
            )

        self.encoder = encoder

        self.to_latent_mean_log_variance = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim_latent * 2),
            Rearrange('b ... (two d) -> two b ... d', two = 2)
        )

        self.from_latent_to_condition = nn.Sequential(
            nn.Linear(dim_latent, dim),
            Rearrange('b d -> b 1 d')
        )

        self.decoder_head = Decoder(
            dim = dim,
            depth = dec_head_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            pre_norm_has_final_norm = False,
            **kwargs,
            **dec_kwargs
        )

        self.decoder_tail = Decoder(
            dim = dim,
            depth = dec_tail_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            pre_norm_has_final_norm = True,
            **kwargs,
            **dec_kwargs
        )

        self.pad_id = pad_id

        # loss weights - vae kl loss

        self.vae_kl_loss_weight = vae_kl_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        seq,
        mask = None,
        return_mean_log_var = False
    ):
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
        filter_logits_fn = top_p,
        logit_filter_kwargs: dict = dict(thres = 0.9)
    ):
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')

        batch = prompts.shape[0]

        # if seq_for_latents passed in, derive latents from it

        if exists(seq_for_latents):
            assert not exists(latents), 'latents should not be passed in if given the seq from which to derive them'

            latents = self.encode_to_latents(seq_for_latents)

        # prepend embeds

        condition = None
        if exists(latents):
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            if latents.ndim == 1: # repeat latents
                latents = repeat(latents, 'd -> b d', b = batch)

            condition = self.from_latent_to_condition(latents)

        # generated

        prompt_len = prompts.shape[-1]

        generated = prompts

        tokens = self.token_emb(generated)

        for _ in range(max(0, seq_len - prompt_len)):

            head_embed = self.decoder_head(tokens)

            if exists(condition):
                head_embed = head_embed + condition

            tail_embed = self.decoder_tail(head_embed)

            tail_embed = tail_embed[:, -1]

            logits = self.token_unembed(tail_embed)

            logits = filter_logits_fn(logits, **logit_filter_kwargs)

            sampled = gumbel_sample(logits)

            generated, _ = pack((generated, sampled), 'b *')
            tokens, _ = pack((tokens, self.token_emb(sampled)), 'b * d')

        return inverse_pack(generated)

    def forward(
        self,
        seq,
        return_all_losses = False
    ):
        batch, device = seq.shape[0], seq.device

        seq, labels = seq[:, :-1], seq[:, 1:]

        encoder_mask = seq != self.pad_id

        tokens = self.token_emb(seq)

        # decoder head

        tokens = self.decoder_head(tokens)

        # get latent Z

        latents, (latents_mean, latents_log_var) = self.encode_to_latents(tokens, mask = encoder_mask, return_mean_log_var = True)

        condition = self.from_latent_to_condition(latents)

        # decoder tail

        tokens = self.decoder_tail(tokens)

        # cross entropy loss

        logits = self.token_unembed(tokens)

        ar_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.pad_id
        )

        # vae kl loss

        vae_kl_loss = (
            latents_log_var.exp()
            + latents_mean.square()
            - latents_log_var
            - 1.
        ).sum(dim = -1).mean()

        # return losses

        total_loss = (
            ar_loss +
            vae_kl_loss * self.vae_kl_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (ar_loss, vae_kl_loss)

        return total_loss, losses
