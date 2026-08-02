from __future__ import annotations

from random import random
from typing import Callable

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce
from torch_einops_utils import temp_eval, batched_index_select

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg

# winner callback helpers

def lowest_entropy_winner_fn(logits: Tensor) -> Tensor:
    """
    Given candidate logits of shape (batch, candidates, seq_len, vocab_size),
    returns candidate index with lowest mean token entropy across the sequence for each batch item.
    """
    probs = F.softmax(logits, dim = -1)
    log_probs = F.log_softmax(logits, dim = -1)
    entropy = - (probs * log_probs).sum(dim = -1).mean(dim = -1)
    return entropy.argmin(dim = -1)

# main class

class XMInducedLatentDecoder(Module):
    """
    Latent Variable Decoder based on Explorative Modeling (Forward XM)
    by Alexi Gladstone et al. (https://arxiv.org/abs/2607.27372).
    """
    def __init__(
        self,
        net: Module,
        num_latents = 4,
        dim = None,
        latent_dim = None,
        candidates = 2,
        max_batch_size = None,
        ignore_index = -100,
        latent_drop_prob = 0.,
        always_latent_proj = False
    ):
        super().__init__()
        self.net = net

        assert candidates >= 1, 'candidates must be at least 1'
        self.candidates = candidates
        self.max_batch_size = max_batch_size
        self.ignore_index = ignore_index

        dim = default(dim, latent_dim, net.attn_layers.dim)

        self.dim = dim
        self.num_latents = num_latents
        self.latent_drop_prob = latent_drop_prob

        self.latent_dim = default(latent_dim, dim)

        has_latent_proj = self.latent_dim != self.dim or always_latent_proj

        self.latent_proj = nn.Linear(self.latent_dim, self.dim) if has_latent_proj else nn.Identity()

    @property
    def max_seq_len(self):
        return self.net.max_seq_len

    @temp_eval
    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        seq_len: int,
        latents: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        batch, device = start_tokens.shape[0], start_tokens.device

        if not exists(latents):
            latents = torch.randn(batch, self.num_latents, self.latent_dim, device = device)

        latent_cond = self.latent_proj(latents)

        auto_wrapper = AutoregressiveWrapper(self.net)
        return auto_wrapper.generate(start_tokens, seq_len, prepend_embeds = latent_cond, excise_prepend_embeds = True, **kwargs)

    @temp_eval
    @torch.no_grad()
    def generate_with_candidate_latents(
        self,
        start_tokens: Tensor,
        seq_len: int,
        candidates: int | None = None,
        latents: Tensor | None = None,
        winner_fn: Callable = lowest_entropy_winner_fn,
        return_best_latents = False,
        **kwargs
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:

        candidate_logits, latents = self(
            start_tokens,
            latents = latents,
            candidates = candidates,
            return_loss = False,
            **kwargs
        )

        winner = winner_fn(candidate_logits)

        if winner.ndim <= 1:
            best_latents = batched_index_select(latents, winner, dim = 1)
        else:
            best_latents = winner

        out = self.generate(
            start_tokens = start_tokens,
            seq_len = seq_len,
            latents = best_latents,
            **kwargs
        )

        if not return_best_latents:
            return out

        return out, (latents, winner)

    def forward(
        self,
        seq: Tensor,
        latents: Tensor | None = None,
        candidates = None,
        max_batch_size = None,
        latent_drop_prob = None,
        return_loss = True,
        **kwargs
    ):
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)
        latent_drop_prob = default(latent_drop_prob, self.latent_drop_prob)

        batch, device = seq.shape[0], seq.device

        # autoregressive sequence targets

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        # check if latents are dropped during training

        if self.training and latent_drop_prob > 0. and random() < latent_drop_prob:
            logits = self.net(seq, **kwargs)

            if not return_loss:
                return logits

            return F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(labels, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

        total = batch * candidates
        chunk_size = default(max_batch_size, total)

        # handle custom or random Gaussian noise latent candidates

        if not exists(latents):
            latents = torch.randn(batch, candidates, self.num_latents, self.latent_dim, device = device)
        elif latents.ndim == 3:
            latents = repeat(latents, 'b n d -> b k n d', k = candidates)

        latent_cond = self.latent_proj(rearrange(latents, 'b k n d -> (b k) n d'))

        # repeat input sequence and targets K times across batch dimension

        seq_candidates = repeat(seq, 'b ... -> (b k) ...', k = candidates)

        if return_loss:
            labels_candidates = repeat(labels, 'b ... -> (b k) ...', k = candidates)

        losses = []
        all_logits = []

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)

            chunk_seq = seq_candidates[start:end]
            chunk_latents = latent_cond[start:end]

            logits = self.net(chunk_seq, prepend_embeds = chunk_latents, excise_prepend_embeds = True, **kwargs)

            if not return_loss:
                all_logits.append(logits)
                continue

            chunk_labels = labels_candidates[start:end]

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(chunk_labels, 'b n -> (b n)'),
                reduction = 'none',
                ignore_index = self.ignore_index
            )

            losses.append(reduce(loss, '(b n) -> b', 'mean', b = end - start))

        if not return_loss:
            raw_logits = cat(all_logits, dim = 0)
            candidate_logits = rearrange(raw_logits, '(b k) ... -> b k ...', b = batch, k = candidates)
            return candidate_logits, latents

        # winner-takes-all candidate selection (Forward XM)

        candidate_losses = reduce(cat(losses, dim = 0), '(b k) -> b k', 'mean', b = batch, k = candidates)

        return candidate_losses.amin(dim = -1).mean()
