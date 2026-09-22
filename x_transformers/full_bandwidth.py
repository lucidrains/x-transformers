from __future__ import annotations

# Full-bandwidth Transformer - Xi Wang et al. https://arxiv.org/abs/2608.08888

from collections import namedtuple
from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import (
    pad_left_at_dim,
    maybe_return,
    temp_eval,
    pack_with_inverse,
    cast_tensor,
    exclusive_cumsum
)

from x_transformers.x_transformers import (
    Decoder,
    TransformerWrapper,
    LinearNoBias,
    RMSNorm,
    Identity
)

from x_transformers.autoregressive_wrapper import top_k, FILTER_LOGITS_FN

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def uniform_like(t, low = -1., high = 1.):
    return torch.empty_like(t).uniform_(low, high)

# dynamic rollout loss weighting
# weight each rollout step by exp(-decay * cumulative loss of preceding steps), as errors compound through the dynamics model
# threshold accounts for the irreducible entropy floor of language (~1.0 nat), penalizing only excess modeling error

def dynamic_rollout_loss_weights(
    step_losses,
    decay = 1.,
    threshold = 1.
):
    excess_losses = (step_losses - threshold).clamp(min = 0.)
    cum_step_losses = exclusive_cumsum(excess_losses, dim = 0)
    return (-decay * cum_step_losses).exp()

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
        self.token_norm = RMSNorm(dim)
        self.norm = RMSNorm(dim)

    def forward(
        self,
        latents,
        encoded_token_ids
    ):
        # they use this scheme to encourage using the previous latent state
        # rmsnorm the gate token embeddings, and the fused state before feeding back into the stack

        fused = self.to_state(latents) * self.to_gate(self.token_norm(encoded_token_ids)).sigmoid()
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
        dynamic_rollout_loss_weight = False,
        dynamic_loss_decay = 0.5,
        dynamic_loss_threshold = 1.,
    ):
        super().__init__()
        assert isinstance(net.attn_layers, Decoder), 'must be a decoder'
        assert exists(net.attn_layers.residual_scale), 'depth scaling must be turned on'
        assert callable(net.to_logits) and not isinstance(net.to_logits, nn.Module), 'weight tying must be turned on'
        assert not isinstance(net.post_emb_norm, Identity), 'post embedding norm must be turned on'

        self.net = net
        self.temporal_parallel_passes = temporal_parallel_passes
        self.ignore_index = ignore_index
        self.jitter_noise_delta = jitter_noise_delta
        self.feedback_pass_weight = feedback_pass_weight
        self.dynamic_rollout_loss_weight = dynamic_rollout_loss_weight
        self.dynamic_loss_decay = dynamic_loss_decay
        self.dynamic_loss_threshold = dynamic_loss_threshold

        dim = net.attn_layers.dim
        self.transition = default(transition, GLUCrossTransition(dim))

    # generate

    @torch.no_grad()
    @temp_eval
    def generate(
        self,
        prompts: Tensor | list[Tensor],
        seq_len: int,
        temperature = 1.,
        filter_logits_fn: Callable | str = top_k,
        filter_kwargs: dict = dict(),
        should_fuse_latent: Callable | None = None,
        transition: Callable | None = None,
        temporal_parallel_passes: int | None = None,
        process_prompt = True,
        eos_token = None,
        **kwargs
    ):
        net = self.net

        assert net.can_cache_kv, 'FullBandwidth generation requires key/value caching'

        transition = default(transition, self.transition)
        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        should_fuse_latent = default(should_fuse_latent, lambda step: True)

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # pack maybe no batch

        prompts = cast_tensor(prompts)
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')
        prompt_len = prompts.shape[-1]

        out = prompts
        cache = None

        # prefill, with optional temporal parallel passes

        logits, intermediates = net(out, return_intermediates = True, **kwargs)

        if process_prompt and temporal_parallel_passes > 1:
            latents = intermediates.last_hidden

            for _ in range(temporal_parallel_passes - 1):
                shifted_latents = pad_left_at_dim(latents[:, :-1], 1, dim = -2)

                def transform_token_embeds(token_embeds, shifted_latents = shifted_latents):
                    fused = transition(shifted_latents, token_embeds)
                    # first position has no previous latent, fall back to plain token embedding (eq. 8)
                    return torch.cat((token_embeds[:, :1], fused[:, 1:]), dim = -2)

                logits, intermediates = net(
                    out,
                    transform_token_embeds = transform_token_embeds,
                    return_intermediates = True,
                    **kwargs
                )
                latents = intermediates.last_hidden

        cache = intermediates

        # autoregressive decoding, with latent feedback from the previous step

        for step in range(seq_len):
            logits = logits[:, -1]

            if temperature == 0.:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token) and (sample == eos_token).all():
                break

            if step == (seq_len - 1):
                break

            # fuse the last top layer latent with the just sampled token

            transform_token_embeds = None

            if temporal_parallel_passes > 1 and should_fuse_latent(step):
                last_latent = intermediates.last_hidden[:, -1:]

                def transform_token_embeds(token_embeds, last_latent = last_latent):
                    fused = transition(last_latent, token_embeds[:, -1:])
                    return torch.cat((token_embeds[:, :-1], fused), dim = -2)

            logits, intermediates = net(
                out,
                transform_token_embeds = transform_token_embeds,
                cache = cache,
                return_intermediates = True,
                **kwargs
            )

            cache = intermediates

        # unpack output

        return inverse_pack(out[:, prompt_len:])

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
        transition: Callable | None = None,
        dynamic_rollout_loss_weight: bool | None = None,
        dynamic_loss_decay: float | None = None,
        dynamic_loss_threshold: float | None = None,
        **kwargs
    ):
        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        assert temporal_parallel_passes >= 1, 'must have at least 1 pass'

        transition = default(transition, self.transition)
        dynamic_rollout_loss_weight = default(dynamic_rollout_loss_weight, self.dynamic_rollout_loss_weight)
        dynamic_loss_decay = default(dynamic_loss_decay, self.dynamic_loss_decay)
        dynamic_loss_threshold = default(dynamic_loss_threshold, self.dynamic_loss_threshold)

        # split for next token prediction

        if return_loss:
            inp, target = token_ids[:, :-1], token_ids[:, 1:]
        else:
            inp, target = token_ids, None

        # first pass (standard)

        logits, intermediates = self.net(inp, return_intermediates = True, **kwargs)

        latents = intermediates.last_hidden
        all_logits = [logits]

        first_pass_loss = None
        feedback_pass_losses = []

        if return_loss:
            first_pass_loss = F.cross_entropy(
                rearrange(logits, 'b n l -> (b n) l'),
                rearrange(target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

        # feedback passes (temporal parallel)

        for _ in range(temporal_parallel_passes - 1):
            if self.training and self.jitter_noise_delta > 0.:
                latents = latents + uniform_like(latents, -self.jitter_noise_delta, self.jitter_noise_delta)

            # shift previous top-layer latents rightward by one position, then fuse with the token embeddings

            shifted_latents = pad_left_at_dim(latents[:, :-1], 1, dim = -2)

            def transform_token_embeds(token_embeds, shifted_latents = shifted_latents):
                fused = transition(shifted_latents, token_embeds)
                # first position has no previous latent, fall back to plain token embedding (eq. 8)
                return torch.cat((token_embeds[:, :1], fused[:, 1:]), dim = -2)

            # feed fused state back into the stack in place of token embeddings

            logits, intermediates = self.net(
                inp,
                transform_token_embeds = transform_token_embeds,
                return_intermediates = True,
                **kwargs
            )

            latents = intermediates.last_hidden
            all_logits.append(logits)

            if not return_loss:
                continue

            # only calculate feedback loss on positions that received latent states (1:)

            feedback_logits = logits[:, 1:] if logits.shape[1] > 1 else logits
            feedback_target = target[:, 1:] if target.shape[1] > 1 else target

            feedback_pass_losses.append(F.cross_entropy(
                rearrange(feedback_logits, 'b n l -> (b n) l'),
                rearrange(feedback_target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            ))

        # returns

        if not return_loss:
            return all_logits[-1], dict(all_pass_logits = all_logits)

        # combine first pass loss with the feedback pass losses
        # optionally applying dynamic rollout loss weighting as a continuous curriculum across passes

        all_pass_losses = [first_pass_loss, *feedback_pass_losses]

        loss = first_pass_loss

        if len(feedback_pass_losses) > 0:
            all_losses = torch.stack(all_pass_losses)
            feedback_losses = all_losses[1:]

            if dynamic_rollout_loss_weight:
                weights = dynamic_rollout_loss_weights(
                    all_losses.detach(),
                    decay = dynamic_loss_decay,
                    threshold = dynamic_loss_threshold
                )[1:]

                feedback_pass_loss = (feedback_losses * weights).sum() / weights.sum().clamp(min = 1e-8)
            else:
                feedback_pass_loss = feedback_losses.mean()

            loss = loss + feedback_pass_loss * self.feedback_pass_weight

        loss_breakdown = LossBreakdown(
            first_pass_loss = first_pass_loss,
            feedback_pass_losses = feedback_pass_losses,
            all_pass_losses = all_pass_losses
        )

        return loss, dict(
            loss_breakdown = loss_breakdown,
            all_pass_logits = all_logits
        )
