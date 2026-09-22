from __future__ import annotations

# Full-bandwidth Transformer - Xi Wang et al. https://arxiv.org/abs/2608.08888

from collections import namedtuple

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import pad_left_at_dim, maybe_return

from x_transformers.x_transformers import (
    Decoder,
    TransformerWrapper,
    LinearNoBias,
    RMSNorm,
    Identity
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def uniform_like(t, low = -1., high = 1.):
    return torch.empty_like(t).uniform_(low, high)

# loss breakdown

LossBreakdown = namedtuple('LossBreakdown', [
    'first_pass_loss',
    'feedback_pass_losses',
    'all_pass_losses'
])

# GLU cross transition from Full-bandwidth Transformer (Eq. 4)

class GLUCrossTransition(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_state = LinearNoBias(dim, dim)
        self.to_gate = LinearNoBias(dim, dim)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        latents,
        encoded_token_ids
    ):
        # they use this scheme to encourage using the previous latent state
        fused = self.to_state(latents) * self.to_gate(encoded_token_ids).sigmoid()
        return self.norm(fused)

# main class

class FullBandwidth(Module):
    def __init__(
        self,
        net: TransformerWrapper,
        temporal_parallel_passes = 2,
        transition: Module | None = None,
        feedback_pass_weight = 1.,
        ignore_index = -100,
        jitter_noise_delta = 0.02, # paper used sigma = 0.02
    ):
        super().__init__()
        assert isinstance(net.attn_layers, Decoder), 'must be a decoder'
        assert exists(net.attn_layers.residual_scale), 'depth scaling must be turned on'
        assert not isinstance(net.to_logits, nn.Module), 'weight tying must be turned on'
        assert not isinstance(net.post_emb_norm, Identity), 'post embedding norm must be turned on'

        self.net = net
        self.temporal_parallel_passes = temporal_parallel_passes
        self.ignore_index = ignore_index
        self.jitter_noise_delta = jitter_noise_delta
        self.feedback_pass_weight = feedback_pass_weight

        dim = net.attn_layers.dim
        self.transition = default(transition, GLUCrossTransition(dim))

    # able to optionally return loss breakdown or all pass logits
    # primary is dynamically loss or logits based on return_loss

    @maybe_return(
        'loss_breakdown',
        'all_pass_logits',
        primary = lambda kwargs: 'loss' if kwargs.get('return_loss', True) else 'logits'
    )
    def forward(
        self,
        token_ids: Tensor,
        return_loss = True,
        temporal_parallel_passes: int | None = None,
        return_loss_breakdown = False,
        return_all_pass_logits = False,
        **kwargs
    ):
        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        assert temporal_parallel_passes >= 1, 'must have at least 1 pass'

        # split for next token prediction

        if return_loss:
            inp, target = token_ids[:, :-1], token_ids[:, 1:]
        else:
            inp, target = token_ids, None

        # initial token embeddings

        encoded_token_ids = self.net.token_emb(inp)

        # first pass (standard)

        logits, intermediates = self.net(inp, return_intermediates = True, **kwargs)

        latents = intermediates.last_hidden
        all_logits = [logits]

        total_loss = 0.
        first_pass_loss = None
        feedback_pass_losses = []

        if return_loss:
            first_pass_loss = F.cross_entropy(
                rearrange(logits, 'b n l -> (b n) l'),
                rearrange(target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

            total_loss = total_loss + first_pass_loss

        # feedback passes (temporal parallel)

        for _ in range(temporal_parallel_passes - 1):
            if self.training and self.jitter_noise_delta > 0.:
                latents = latents + uniform_like(latents, -self.jitter_noise_delta, self.jitter_noise_delta)

            # shift previous top-layer latents rightward by one position

            shifted_latents = pad_left_at_dim(latents[:, :-1], 1, dim = -2)

            # fuse with token embeddings via transition

            fused = self.transition(shifted_latents, encoded_token_ids)

            # feed fused representation back into the stack

            logits, intermediates = self.net(inp, sum_embeds = fused, return_intermediates = True, **kwargs)

            latents = intermediates.last_hidden
            all_logits.append(logits)

            # calculate next token prediction loss

            if not return_loss:
                continue

            # only calculate feedback loss on positions that received latent states (1:)

            feedback_logits = logits[:, 1:] if logits.shape[1] > 1 else logits
            feedback_target = target[:, 1:] if target.shape[1] > 1 else target

            pass_loss = F.cross_entropy(
                rearrange(feedback_logits, 'b n l -> (b n) l'),
                rearrange(feedback_target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

            feedback_pass_losses.append(pass_loss)
            total_loss = total_loss + pass_loss * self.feedback_pass_weight

        # returns

        if not return_loss:
            return all_logits[-1], dict(all_pass_logits = all_logits)

        loss_breakdown = LossBreakdown(
            first_pass_loss = first_pass_loss,
            feedback_pass_losses = feedback_pass_losses,
            all_pass_losses = [first_pass_loss, *feedback_pass_losses]
        )

        return total_loss, dict(
            loss_breakdown = loss_breakdown,
            all_pass_logits = all_logits
        )
