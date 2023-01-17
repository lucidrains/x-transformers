import math
from random import random

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# sampling helpers

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# schedules

def linear_schedule(t):
    return 1 - t

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# wrapper class

class NonAutoregressiveWrapper(nn.Module):
    """
    https://arxiv.org/abs/1904.09324
    https://arxiv.org/abs/2202.04200
    """

    def __init__(
        self,
        net,
        *,
        mask_id,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        schedule = 'linear'
    ):
        super().__init__()
        self.net = net

        dim = net.emb_dim
        self.dim = dim

        self.mask_id = mask_id

        self.max_seq_len = net.max_seq_len
        self.steps = steps

        if callable(schedule):
            self.schedule_fn = schedule
        if schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')
    
        self.self_cond = self_cond

        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(dim))
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob

    @torch.no_grad()
    def generate(
        self,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.9,
        **kwargs
    ):
        sample_one = not exists(batch_size)
        batch_size = default(batch_size, 1)

        device = next(self.net.parameters()).device

        was_training = self.training
        self.eval()

        times = torch.linspace(0., 1., self.steps + 1)

        # sequence starts off as all masked

        shape = (batch_size, self.max_seq_len)

        seq = torch.full(shape, self.mask_id, device = device)
        mask = torch.full(shape, True, device = device)

        # slowly demask

        all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()

        # self conditioning

        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None

        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):

            self_cond = self.to_self_cond(last_embed) if has_self_cond else None

            logits, embeds = self.net(
                seq,
                sum_embeds = self_cond,
                return_logits_and_embeddings = True,
                **kwargs
            )

            if has_self_cond:
                last_embed = embeds

            if exists(filter_thres):
                logits = top_k(logits, filter_thres)

            temperature = start_temperature * (steps_until_x0 / self.steps)

            probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

            packed_probs, packed_shape = pack([probs], '* c')

            sampled_ids = torch.multinomial(packed_probs, 1)

            sampled_ids = rearrange(sampled_ids, '... 1 -> ...')
            sampled_ids, = unpack(sampled_ids, packed_shape, '*')

            seq = torch.where(mask, sampled_ids, seq)

            scores = (1 - probs).gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
            scores = rearrange(scores, 'b n 1 -> b n')

            if mask_num_tokens == 0:
                pass

            scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
            mask_indices = scores.topk(mask_num_tokens, dim = -1).indices
            mask = torch.zeros_like(scores, dtype = torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, self.mask_id)

        self.train(was_training)

        if sample_one:
            seq = rearrange(seq, '1 n -> n')

        return seq

    def forward(
        self,
        x,
        **kwargs
    ):
        b, n, device = *x.shape, x.device
        assert n == self.max_seq_len

        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        batched_randperm = torch.rand((b, n), device = device).argsort(dim = -1).float()

        def get_mask_from_times(rand_times, randperm):
            rand_probs = self.schedule_fn(rand_times)
            num_tokens_mask = (rand_probs * n).clamp(min = 1.)
            return randperm < rearrange(num_tokens_mask, 'b -> b 1')

        mask = get_mask_from_times(rand_times, batched_randperm)
        masked = torch.where(mask, self.mask_id, x)

        if self.self_cond:
            if random() > self.self_cond_train_prob:
                self_cond = self.null_embed
            else:
                with torch.no_grad():
                    self_cond = self.net(masked, return_embeddings = True, **kwargs).detach()

            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        logits = self.net(masked, **kwargs)

        loss = F.cross_entropy(
            logits[mask],
            x[mask]
        )

        return loss
