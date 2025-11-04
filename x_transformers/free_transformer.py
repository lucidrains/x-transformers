from __future__ import annotations

# https://arxiv.org/abs/2510.17558
# François Fleuret
# https://www.youtube.com/watch?v=Nao16-6l6dQ

import math

import torch
from torch import nn, Tensor, is_tensor, tensor, arange
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
from einops import rearrange, reduce, repeat, einsum, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def pack_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

# binary mapper

NAT = math.log(2)

def binary_entropy(logits):
    prob = logits.sigmoid()
    not_prob = 1. - prob
    return -(prob * F.logsigmoid(logits) + not_prob * F.logsigmoid(-logits)).sum(dim = -1)

class BinaryMapper(Module):
    def __init__(
        self,
        bits = 1,
        kl_loss_threshold = NAT # 1 bit
    ):
        super().__init__()

        self.bits = bits
        self.num_codes = 2 ** bits
        self.kl_loss_threshold = kl_loss_threshold

        power_two = 2 ** arange(bits)
        codes = (arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).byte().bool()

        self.register_buffer('power_two', power_two, persistent = False)
        self.register_buffer('codes', codes, persistent = False)

    def forward(
        self,
        logits,
        temperature = 1.,
        straight_through = None
    ):
        straight_through = default(straight_through, self.training)

        assert logits.shape[-1] == self.bits, f'logits must have a last dimension of {self.bits}'

        # temperature and prob for sampling

        prob_for_sample = (logits / temperature).sigmoid()

        # sampling

        sampled_bits = (torch.rand_like(logits) <= prob_for_sample).long()
        indices = (self.power_two * sampled_bits).sum(dim = -1)

        one_hot = F.one_hot(indices, self.num_codes).float()

        # return hard one hot if not training or overridden

        if not straight_through:
            return one_hot

        # calculate negative entropy

        kl_div = self.bits * NAT - binary_entropy(logits)
        aux_kl_loss = F.relu(kl_div - self.kl_loss_threshold).mean()

        # get the soft G for the gradients and do a straight through

        soft_G = (
            einsum(F.logsigmoid(logits), self.codes.float(), '... bits, codes bits -> ... codes') +
            einsum(F.logsigmoid(-logits), (~self.codes).float(), '... bits, codes bits -> ... codes')
        ).exp()

        # straight through

        one_hot = one_hot + soft_G - soft_G.detach()

        return one_hot, aux_kl_loss

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
        latent_bits = 16,
        kl_loss_threshold = NAT,
        binary_mapper_kwargs: dict = dict(),
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        kl_loss_weight = 1.,
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

        self.to_latent_bit_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, latent_bits, bias = False),
        )

        self.binary_mapper = BinaryMapper(
            latent_bits,
            kl_loss_threshold,
            **binary_mapper_kwargs
        )

        self.from_latent_to_condition = nn.Sequential(
            nn.Linear(2 ** latent_bits, dim, bias = False),
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

        self.kl_loss_weight = kl_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        seq,
        mask = None,
        return_kl_loss = False
    ):
        pooled = self.encoder(seq, mask = mask)

        bit_logits = self.to_latent_bit_logits(pooled)

        one_hot_latents, kl_loss = self.binary_mapper(bit_logits, straight_through = True)

        if not return_kl_loss:
            return one_hot_latents

        return one_hot_latents, kl_loss

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        latents = None,
        filter_logits_fn = top_p,
        logit_filter_kwargs: dict = dict(thres = 0.9)
    ):
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')

        batch = prompts.shape[0]

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

        latents, kl_loss = self.encode_to_latents(tokens, mask = encoder_mask, return_kl_loss = True)

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

        # return losses

        total_loss = (
            ar_loss +
            kl_loss * self.kl_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (ar_loss, kl_loss)

        return total_loss, losses
