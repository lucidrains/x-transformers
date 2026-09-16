from __future__ import annotations

import math
from random import random, randrange
from typing import Callable, Sequence

import torch
from torch import nn, Tensor, cat
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce
from torch_einops_utils import temp_eval, batched_index_select, masked_mean, and_masks

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# helpers

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg

def cast_tuple(t, length = 1):
    return t if isinstance(t, (tuple, list)) else ((t,) * length)

# winner callback helpers

def lowest_loss_winner_fn(losses: Tensor, intermediates = None) -> Tensor:
    return losses.argmin(dim = -1)

# main class

class XMLatentDecoder(Module):
    """
    Latent Variable Decoder based on Explorative Modeling (Forward XM)
    by Alexi Gladstone et al. (https://arxiv.org/abs/2607.27372).
    and Lei Yang (https://arxiv.org/abs/2401.00036)
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
        always_latent_proj = False,
        repulsive_loss_weight = 0.,
        winner_fn: Callable | Sequence[Callable] = lowest_loss_winner_fn,
        winner_fns: Sequence[Callable] | None = None
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

        # repulsive loss force between candidate distributions

        self.repulsive_loss_weight = repulsive_loss_weight

        winner_fns = cast_tuple(default(winner_fns, winner_fn), num_latents)
        assert len(winner_fns) == num_latents, f'winner_fns length ({len(winner_fns)}) must match num_latents ({num_latents})'

        self.winner_fns = tuple(winner_fns)

    @property
    def max_seq_len(self):
        return self.net.max_seq_len

    def init_latents(
        self,
        latents: Tensor | Sequence[Tensor | None] | None,
        batch: int,
        device = None
    ) -> Tensor:
        if not exists(latents):
            return torch.randn(batch, self.num_latents, self.latent_dim, device = device)

        # a specific latent or None can be given for each slot, with None randomly initialized

        if not isinstance(latents, (tuple, list)):
            return latents

        assert len(latents) == self.num_latents, f'latents length ({len(latents)}) must match num_latents ({self.num_latents})'

        ref = default(*latents)
        device, dtype = (ref.device, ref.dtype) if exists(ref) else (device, None)

        latents = [default(latent, torch.randn(batch, self.latent_dim, device = device, dtype = dtype)) for latent in latents]

        return torch.stack(latents, dim = 1)

    @temp_eval
    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        seq_len: int,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        **kwargs
    ) -> Tensor:
        batch, device = start_tokens.shape[0], start_tokens.device

        latents = self.init_latents(latents, batch, device = device)

        latent_cond = self.latent_proj(latents)

        auto_wrapper = AutoregressiveWrapper(self.net)
        return auto_wrapper.generate(start_tokens, seq_len, prepend_embeds = latent_cond, excise_prepend_embeds = True, **kwargs)

    @temp_eval
    @torch.no_grad()
    def generate_with_candidate_latents(
        self,
        start_tokens: Tensor,
        seq_len: int,
        winner_fn: Callable | Sequence[Callable] | None = None,
        winner_fns: Sequence[Callable] | None = None,
        candidates: int | None = None,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        active_latent_index: int | None = None,
        return_best_latents = False,
        **kwargs
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:

        passed_winner_fns = default(winner_fns, winner_fn)
        is_seq_winners = isinstance(passed_winner_fns, (tuple, list))

        winner_fns = cast_tuple(default(passed_winner_fns, self.winner_fns), self.num_latents)

        batch, device = start_tokens.shape[0], start_tokens.device

        latents = self.init_latents(latents, batch, device = device)

        # a single winner fn selects winners for the first latent, while a sequence of winner fns sweeps all latents

        latent_indices = [active_latent_index] if exists(active_latent_index) else (range(self.num_latents) if is_seq_winners else [0])

        all_winners = []

        for latent_idx in latent_indices:
            candidate_logits, candidate_latents = self(
                start_tokens,
                latents = latents,
                candidates = candidates,
                active_latent_index = latent_idx,
                return_loss = False,
                **kwargs
            )

            winner = winner_fns[latent_idx](candidate_logits)

            if winner.ndim > 1:
                winner = winner.argmax(dim = -1)

            all_winners.append(winner)

            latents = batched_index_select(candidate_latents, winner, dim = 1)

        out = self.generate(
            start_tokens = start_tokens,
            seq_len = seq_len,
            latents = latents,
            **kwargs
        )

        if not return_best_latents:
            return out

        winner_result = all_winners[0] if len(all_winners) == 1 else torch.stack(all_winners, dim = -1)
        return out, (candidate_latents, winner_result)

    def forward(
        self,
        seq: Tensor,
        latents: Tensor | Sequence[Tensor | None] | None = None,
        candidates = None,
        max_batch_size = None,
        latent_drop_prob = None,
        return_loss = True,
        repulsive_loss_weight = None,
        mask: Tensor | None = None,
        winner_fn: Callable | Sequence[Callable] | None = None,
        winner_fns: Sequence[Callable] | None = None,
        active_latent_index: int | None = None,
        **kwargs
    ):
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)
        latent_drop_prob = default(latent_drop_prob, self.latent_drop_prob)

        repulsive_loss_weight = default(repulsive_loss_weight, self.repulsive_loss_weight)

        winner_fns = cast_tuple(default(winner_fns, winner_fn, self.winner_fns), self.num_latents)

        active_latent_index = default(active_latent_index, randrange(self.num_latents))
        assert 0 <= active_latent_index < self.num_latents

        winner_fn = winner_fns[active_latent_index]

        batch, device = seq.shape[0], seq.device

        # autoregressive sequence targets

        if return_loss:
            seq, labels = seq[:, :-1], seq[:, 1:]

        seq_mask = mask[:, 1:] if (exists(mask) and return_loss) else mask
        ignore_mask = (labels != self.ignore_index) if return_loss else (seq != self.ignore_index)

        loss_mask = and_masks((seq_mask, ignore_mask))

        # check if latents are dropped during training

        if self.training and latent_drop_prob > 0. and random() < latent_drop_prob:
            logits = self.net(seq, mask = mask, **kwargs)

            if not return_loss:
                return logits

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(labels, 'b n -> (b n)'),
                reduction = 'none',
                ignore_index = self.ignore_index
            )

            return masked_mean(rearrange(loss, '(b n) -> b n', b = batch), loss_mask)

        total = batch * candidates
        chunk_size = default(max_batch_size, total)

        # handle custom or random Gaussian noise latent candidates

        latents = self.init_latents(latents, batch, device = device)

        # only the active latent is explored, all other latents held constant across candidates

        if latents.ndim == 3:
            latents = repeat(latents, 'b n d -> b k n d', k = candidates).clone()
            latents[:, :, active_latent_index] = torch.randn(batch, candidates, self.latent_dim, device = device, dtype = latents.dtype)

        latent_cond = self.latent_proj(rearrange(latents, 'b k n d -> (b k) n d'))

        # repeat input sequence and targets K times across batch dimension

        seq_candidates = repeat(seq, 'b ... -> (b k) ...', k = candidates)

        if return_loss:
            labels_candidates = repeat(labels, 'b ... -> (b k) ...', k = candidates)

        mask_candidates = repeat(loss_mask, 'b ... -> (b k) ...', k = candidates)

        losses = []
        all_logits = []
        all_intermediates = []

        calc_repulsion = repulsive_loss_weight > 0. and candidates > 1
        needs_intermediates = winner_fn is not lowest_loss_winner_fn
        collect_logits = calc_repulsion or needs_intermediates

        for start in range(0, total, chunk_size):
            chunk_batch_size = min(total - start, chunk_size)
            chunk = slice(start, start + chunk_batch_size)

            chunk_seq = seq_candidates[chunk]
            chunk_latents = latent_cond[chunk]

            chunk_kwargs = kwargs.copy()
            if exists(mask):
                chunk_kwargs['mask'] = mask_candidates[chunk]

            net_out = self.net(
                chunk_seq,
                prepend_embeds = chunk_latents,
                excise_prepend_embeds = True,
                return_intermediates = needs_intermediates,
                **chunk_kwargs
            )

            logits, intermediates = net_out if needs_intermediates else (net_out, None)

            if needs_intermediates:
                all_intermediates.append(intermediates)

            if not return_loss:
                all_logits.append(logits)
                continue

            if collect_logits:
                all_logits.append(logits)

            chunk_labels = labels_candidates[chunk]
            chunk_mask = mask_candidates[chunk]

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> (b n) c'),
                rearrange(chunk_labels, 'b n -> (b n)'),
                reduction = 'none',
                ignore_index = self.ignore_index
            )

            loss = rearrange(loss, '(b n) -> b n', b = chunk_batch_size)
            losses.append(masked_mean(loss, mask = chunk_mask, dim = -1))

        if not return_loss:
            raw_logits = cat(all_logits, dim = 0)
            candidate_logits = rearrange(raw_logits, '(b k) ... -> b k ...', b = batch, k = candidates)
            return candidate_logits, latents

        if collect_logits:
            raw_logits = cat(all_logits, dim = 0)
            candidate_logits = rearrange(raw_logits, '(b k) ... -> b k ...', b = batch, k = candidates)

        # selection (winner-takes-all candidate selection - Forward XM)

        candidate_losses = reduce(cat(losses, dim = 0), '(b k) -> b k', 'mean', b = batch, k = candidates)

        intermediates = None

        if needs_intermediates:
            intermediates = all_intermediates[0] if len(all_intermediates) == 1 else all_intermediates
            intermediates.logits = candidate_logits

        try:
            winner = winner_fn(candidate_losses, intermediates)
        except TypeError:
            winner = winner_fn(candidate_losses)

        if winner.ndim > 1:
            winner = winner.argmax(dim = -1)

        winner_loss = batched_index_select(candidate_losses, winner, dim = 1).mean()

        if not calc_repulsion:
            return winner_loss

        # diversity (repulsive loss via jensen-shannon divergence)

        log_prob = candidate_logits.log_softmax(dim = -1)

        log_mixture = log_prob.logsumexp(dim = 1) - math.log(candidates)
        log_mixture = repeat(log_mixture, 'b n c -> b k n c', k = candidates)

        kl = F.kl_div(log_mixture, log_prob, log_target = True, reduction = 'none').sum(dim = -1)
        kl = reduce(kl, 'b k n -> b n', 'mean')

        repulsive_kl = masked_mean(kl, mask = loss_mask)

        return winner_loss - repulsive_loss_weight * repulsive_kl
