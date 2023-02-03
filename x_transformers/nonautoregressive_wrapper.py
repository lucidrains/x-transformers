import math
from random import random
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, pack, unpack

from x_transformers.x_transformers import TransformerWrapper
from typing import Optional

# constants

Losses = namedtuple('Losses', ['loss', 'generator_loss', 'critic_loss'])

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

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# prob helpers

def sample_prob(prob):
    return random() < prob

def coin_flip():
    return sample_prob(0.5)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# schedules

def linear_schedule(t):
    return 1 - t

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# self token critic
# inspired by Nijkamp et al. - https://aclanthology.org/2021.naacl-main.409/

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

        dim = net.attn_layers.dim
        self.to_logits = nn.Linear(dim, 1)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

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
        no_replace_prob = 0.15,          # which percentage of the tokens masked will stay the same, done in original MLM paper
        random_token_prob = 0.1,         # which percentage of tokens to be replaced with random token, done in original MLM paper
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # when unmasking, whether it can remask previously unmasked
        token_critic: Optional[TransformerWrapper] = None,
        self_token_critic = False,
        critic_loss_weight = 1.
    ):
        super().__init__()
        assert not (self_token_critic and exists(token_critic))

        self.net = net

        dim = net.emb_dim
        self.dim = dim
        self.num_tokens = net.num_tokens

        self.mask_id = mask_id

        # afaict, maskgit paper did not do this
        # but may help for self conditioning, as used successfully in original BERT

        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob

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

        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # self conditioning

        self.self_cond = self_cond

        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(dim))
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob

        # token critic

        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(net)

        self.critic_loss_weight = critic_loss_weight

    @torch.no_grad()
    def generate(
        self,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
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

            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale

            probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

            sampled_ids = gumbel_sample(logits, temperature = max(temperature, 1e-3))

            seq = torch.where(mask, sampled_ids, seq)

            if exists(self.token_critic):
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                scores = 1 - logits.softmax(dim = -1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')

            if mask_num_tokens == 0:
                pass

            if not self.can_mask_prev_unmasked:
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
        only_train_generator = False,
        only_train_critic = False,
        generator_sample_temperature = None,
        **kwargs
    ):
        b, n, device = *x.shape, x.device
        assert n == self.max_seq_len

        orig_seq = x.clone()

        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        batched_randperm = torch.rand((b, n), device = device).argsort(dim = -1).float()

        rand_probs = self.schedule_fn(rand_times)
        num_tokens_mask = (rand_probs * n).clamp(min = 1.)
        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        # to ensure all tokens produce embeddings, instead of just the ones with [mask] input, as done in seminal BERT MLM paper
        # potentially needed for self-conditioning (on embedding) to work well

        replace_mask_id_mask = mask.clone()
        frac_seq_left = 1.

        if self.no_replace_prob > 0. and coin_flip():
            frac_seq_left -= self.no_replace_prob

            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            replace_mask_id_mask &= ~no_replace_prob_mask

        if self.random_token_prob > 0. and coin_flip():
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
            random_tokens = torch.randint(0, self.num_tokens, (b, n), device = device)

            x = torch.where(random_token_prob_mask, random_tokens, x)
            replace_mask_id_mask &= ~random_token_prob_mask

        masked = torch.where(replace_mask_id_mask, self.mask_id, x)

        # self conditioning

        if self.self_cond:
            self_cond = self.null_embed

            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    self_cond = self.net(masked, return_embeddings = True, **kwargs).detach()

            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # logits

        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            logits = self.net(masked, **kwargs)

        # cross entropy loss

        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )

        if not exists(self.token_critic) or only_train_generator:
            return Losses(loss, loss, None)

        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        generated = torch.where(mask, sampled_ids, orig_seq)

        critic_logits = self.token_critic(generated)
        critic_labels = (sampled_ids != orig_seq).float()

        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )

        # determine losses to be returned based on what researcher wants to train

        if only_train_critic:
            total_loss = critic_loss
            loss = None
        else:
            total_loss = loss + critic_loss * self.critic_loss_weight

        return Losses(total_loss, loss,  critic_loss)
