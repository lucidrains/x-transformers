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

        power_two = 2 ** arange(bits)
        codes = (arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).byte().bool()

        self.register_buffer('power_two', power_two, persistent = False)
        self.register_buffer('codes', codes, persistent = False)

        # aux loss

        self.kl_loss_threshold = kl_loss_threshold
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        logits,
        temperature = 1.,
        straight_through = None,
        calc_aux_loss = None
    ):
        straight_through = default(straight_through, self.training)
        calc_aux_loss = default(calc_aux_loss, self.training)

        assert logits.shape[-1] == self.bits, f'logits must have a last dimension of {self.bits}'

        # temperature and prob for sampling

        prob_for_sample = (logits / temperature).sigmoid()

        # sampling

        sampled_bits = (torch.rand_like(logits) <= prob_for_sample).long()
        indices = (self.power_two * sampled_bits).sum(dim = -1)

        one_hot = F.one_hot(indices, self.num_codes).float()

        # maybe calculate aux loss

        aux_kl_loss = self.zero

        if calc_aux_loss:
            # calculate negative entropy

            kl_div = self.bits * NAT - binary_entropy(logits)
            aux_kl_loss = F.relu(kl_div - self.kl_loss_threshold).mean()

        # maybe straight through

        if straight_through:
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
        max_seq_len,
        enc_depth = 1,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        latent_bits = 16,
        per_token_latents = True,  # they use a latent per token in the sequence, instead of one for entire sequence, iiuc
        kl_loss_threshold = NAT,
        binary_mapper_kwargs: dict = dict(),
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        kl_loss_weight = 1.,
        latent_dropout_prob = 0.,
        pad_id = -1,
        **kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.token_unembed = nn.Linear(dim, num_tokens, bias = False)

        self.query_token_for_latents = nn.Parameter(torch.randn(dim) * 1e-2)

        self.per_token_latents = per_token_latents

        self.encoder = Encoder(
            dim = dim,
            depth = enc_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            only_cross = True,
            cross_attend = True,
            use_rmsnorm = True,
            rotary_pos_emb = True,
            pre_norm_has_final_norm = True,
            **kwargs,
            **enc_kwargs
        )

        self.to_latent_bit_logits = nn.Linear(dim, latent_bits, bias = False)

        self.binary_mapper = BinaryMapper(
            latent_bits,
            kl_loss_threshold,
            **binary_mapper_kwargs
        )

        self.from_latent_to_condition = nn.Linear(self.binary_mapper.num_codes, dim, bias = False)

        self.latent_dropout = nn.Dropout(latent_dropout_prob)

        self.decoder_head = Decoder(
            dim = dim,
            depth = dec_head_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            rotary_pos_emb = True,
            use_rmsnorm = True,
            pre_norm_has_final_norm = False,
            **kwargs,
            **dec_kwargs
        ) if dec_head_depth > 0 else None

        assert dec_tail_depth > 0

        self.decoder_tail = Decoder(
            dim = dim,
            depth = dec_tail_depth,
            attn_dim_head = attn_dim_head,
            heads = heads,
            rotary_pos_emb = True,
            use_rmsnorm = True,
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
        decoder_head_embeds,
        mask = None,
        return_kl_loss = False,
        per_token_latents = None
    ):
        per_token_latents = default(per_token_latents, self.per_token_latents)

        batch, seq_len, device = *decoder_head_embeds.shape[:2], decoder_head_embeds.device

        query_tokens = repeat(self.query_token_for_latents, 'd -> b 1 d', b = batch)

        encoder_kwargs = dict()

        # handle the interesting per query token latents, as in the paper

        if per_token_latents:
            query_tokens = repeat(query_tokens, 'b 1 d -> b n d', n = seq_len)

            rotary_pos = torch.arange(seq_len, device = device)

            encoder_kwargs.update(
                pos = rotary_pos,
                context_pos = rotary_pos
            )

        pooled = self.encoder(
            query_tokens,
            context = decoder_head_embeds,
            context_mask = mask,
            **encoder_kwargs
        )

        bit_logits = self.to_latent_bit_logits(pooled)

        one_hot_latents, kl_loss = self.binary_mapper(bit_logits, calc_aux_loss = return_kl_loss)

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
        logit_filter_kwargs: dict = dict(thres = 0.9),
        use_kv_cache = True
    ):
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')

        batch = prompts.shape[0]

        # prepend embeds

        condition = None
        if exists(latents):
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            if latents.dtype in (torch.int, torch.long):
                # if given as indices
                latents = F.one_hot(latents, self.binary_mapper.num_codes).float()

            if latents.ndim == 1: # repeat latents
                latents = repeat(latents, 'd -> b 1 d', b = batch)
            elif latents.ndim == 2:
                latents = rearrange(latents, 'b d -> b 1 d')

            condition = self.from_latent_to_condition(latents)

        # kv cache

        head_cache = tail_cache = None

        # generated

        prompt_len = prompts.shape[-1]

        generated = prompts

        tokens = self.token_emb(generated)

        for _ in range(max(0, seq_len - prompt_len)):

            # head, which may not exist

            if exists(self.decoder_head):
                head_embed, next_head_cache = self.decoder_head(tokens, cache = head_cache, return_hiddens = True)
            else:
                head_embed, next_head_cache = tokens, None

            # handle one token being given to the decoder tail when doing kv caching - rotary embedding needs to know the seq position offset

            seq_pos_offset = head_cache.cache_length if exists(head_cache) else 0

            # tail

            tail_embed, next_tail_cache = self.decoder_tail(head_embed, cache = tail_cache, seq_pos_offset = seq_pos_offset, self_attn_kv_residuals = condition, return_hiddens = True)

            tail_embed = tail_embed[:, -1]

            logits = self.token_unembed(tail_embed)

            logits = filter_logits_fn(logits, **logit_filter_kwargs)

            sampled = gumbel_sample(logits)

            generated, _ = pack((generated, sampled), 'b *')
            tokens, _ = pack((tokens, self.token_emb(sampled)), 'b * d')

            if use_kv_cache:
                head_cache = next_head_cache
                tail_cache = next_tail_cache

        return inverse_pack(generated)

    def forward(
        self,
        seq,
        seq_for_latents = None,
        return_all_losses = False
    ):
        batch, device = seq.shape[0], seq.device

        seq, labels = seq[:, :-1], seq[:, 1:]


        tokens = self.token_emb(seq)

        # decoder head

        if exists(self.decoder_head):
            tokens = self.decoder_head(tokens)

        # determine whether to use a separate sequence for encoding latents

        if exists(seq_for_latents):
            tokens_for_latents = self.token_emb(seq_for_latents)

            if exists(self.decoder_head):
                tokens_for_latents = self.decoder_head(tokens_for_latents)

            encoder_mask = seq_for_latents != self.pad_id
            per_token_latents = False
        else:

            tokens_for_latents = tokens
            encoder_mask = seq != self.pad_id
            per_token_latents = None

        # get latent Z

        latents, kl_loss = self.encode_to_latents(tokens_for_latents, mask = encoder_mask, per_token_latents = per_token_latents, return_kl_loss = True)

        latents = self.latent_dropout(latents)

        condition = self.from_latent_to_condition(latents)

        # decoder tail

        tokens = self.decoder_tail(tokens, self_attn_kv_residuals = condition)

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
