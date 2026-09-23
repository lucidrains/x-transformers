from __future__ import annotations

# Full-bandwidth Transformer - Xi Wang et al. https://arxiv.org/abs/2608.08888

from collections import namedtuple
from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleDict
import torch.nn.functional as F

from einops import rearrange
from torch_einops_utils import (
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
    Identity,
    LayerIntermediates,
    exists,
    first
)

from x_transformers.autoregressive_wrapper import top_k, FILTER_LOGITS_FN

# helper functions

def default(val, d):
    return val if exists(val) else d

def uniform_like(t, low = -1., high = 1.):
    return torch.empty_like(t).uniform_(low, high)

# shift and fuse latent feedback into stream

def shift_and_fuse_latents(latents, stream, transition):
    if latents.shape[-2] <= 1:
        return stream

    fused = transition(latents[:, :-1], stream[:, 1:])
    return torch.cat((stream[:, :1], fused), dim = -2)

# fuse last latent during autoregressive generation

def fuse_last_latent(latents, stream, transition):
    fused = transition(latents[:, -1:], stream[:, -1:])
    return torch.cat((stream[:, :-1], fused), dim = -2)

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

# transitions

# 1. GLU cross transition from Full-bandwidth Transformer (Eq. 4)

class GLUCrossTransition(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_state = LinearNoBias(dim, dim)
        self.to_gate = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        state = self.to_state(self.latent_norm(latents))
        gate = self.to_gate(self.stream_norm(stream)).sigmoid()
        return self.norm(state * gate)

# 2. Residual transition

class ResidualTransition(Module):
    def __init__(self, dim, scale_init = 0.1):
        super().__init__()
        self.proj = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        residual = self.proj(self.latent_norm(latents)) * self.scale
        return self.norm(stream + residual)

# 3. GRU gating transition

class GRUGatingTransition(Module):
    def __init__(self, dim):
        super().__init__()
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.gate = nn.Linear(dim * 2, dim, bias = True)
        self.to_candidate = LinearNoBias(dim * 2, dim)
        self.norm = RMSNorm(dim)
        nn.init.constant_(self.gate.bias, -1.)

    def forward(self, latents, stream):
        packed = torch.cat((self.latent_norm(latents), self.stream_norm(stream)), dim = -1)
        z = self.gate(packed).sigmoid()
        candidate = self.to_candidate(packed).tanh()
        return self.norm(stream.lerp(candidate, z))

# 4. Normalized linear combination

class LinearTransition(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_state = LinearNoBias(dim, dim)
        self.to_stream = LinearNoBias(dim, dim)
        self.latent_norm = RMSNorm(dim)
        self.stream_norm = RMSNorm(dim)
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        state = self.to_state(self.latent_norm(latents))
        stream_proj = self.to_stream(self.stream_norm(stream))
        return self.norm(state + stream_proj)

# 5. Residual GLU transition

class ResidualGLUTransition(Module):
    def __init__(self, dim, scale_init = 1.0):
        super().__init__()
        self.glu = GLUCrossTransition(dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.norm = RMSNorm(dim)

    def forward(self, latents, stream):
        return self.norm(stream + self.glu(latents, stream) * self.scale)

# registry of transition types

TRANSITIONS = dict(
    glu = GLUCrossTransition,
    residual = ResidualTransition,
    gru = GRUGatingTransition,
    linear = LinearTransition,
    residual_glu = ResidualGLUTransition
)

def resolve_transition(spec, dim):
    if isinstance(spec, Module):
        return spec

    if isinstance(spec, str):
        fn = TRANSITIONS.get(spec)
        assert exists(fn), f"unknown transition '{spec}', available: {list(TRANSITIONS.keys())}"
        return fn(dim)

    if callable(spec):
        return spec(dim)

    raise ValueError(f'invalid transition specification: {spec}')

# recirculation pattern shortcuts

RECIRC_SHORTCUTS = dict(
    paper = lambda d: ((d, 1),),
    prev = lambda d: tuple((i, i - 1) for i in range(2, d + 1)),
    top_and_prev = lambda d: tuple(dict.fromkeys(((d, 1), *[(i, i - 1) for i in range(2, d + 1)]))),
    prev_with_self = lambda d: tuple((i, max(i - 1, 1)) for i in range(1, d + 1)),
    self = lambda d: tuple((i, i) for i in range(1, d + 1)),
    full = lambda d: tuple((i, 1) for i in range(1, d + 1)),
    dense = lambda d: tuple((src, dst) for src in range(1, d + 1) for dst in range(1, d + 1)),
)

def resolve_recirc_pairs(depth: int, recirc_pairs: tuple | list | str | None):
    if not exists(recirc_pairs):
        return ((depth, 1),)

    if isinstance(recirc_pairs, str):
        fn = RECIRC_SHORTCUTS.get(recirc_pairs)
        assert exists(fn), f"unknown recirc_pairs shortcut '{recirc_pairs}', available: {list(RECIRC_SHORTCUTS.keys())}"
        return fn(depth)

    return tuple(recirc_pairs)

# helper to extract block latent

def get_block_latent(intermediates: LayerIntermediates, depth: int) -> Tensor:
    block_hiddens = intermediates.block_hiddens

    if exists(block_hiddens) and 0 <= (depth - 1) < len(block_hiddens):
        return block_hiddens[depth - 1]

    assert exists(intermediates.last_hidden), f'cannot find latent for depth {depth}'
    return intermediates.last_hidden

# main class

class FullBandwidth(Module):
    def __init__(
        self,
        net: TransformerWrapper,
        temporal_parallel_passes = 2,
        recirc_pairs: tuple | list | str | None = None,
        transition: Module | str | Callable | None = None,
        transitions: dict | None = None,
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
        depth = net.attn_layers.depth
        self.depth = depth

        raw_pairs = resolve_recirc_pairs(depth, recirc_pairs)

        parsed_pairs = []
        transitions_dict = dict()
        routes_by_in_depth = dict()

        for item in raw_pairs:
            out_d, in_d, *maybe_trans = item
            assert 1 <= out_d <= depth and 1 <= in_d <= depth, f'pair ({out_d}, {in_d}) out of range (1 to {depth})'

            trans_spec = first(maybe_trans)

            if not exists(trans_spec) and exists(transitions):
                trans_spec = transitions.get((out_d, in_d), transitions.get(in_d))

            trans_spec = default(trans_spec, default(transition, 'glu'))

            parsed_pairs.append((out_d, in_d))
            transitions_dict[f'{out_d}->{in_d}'] = resolve_transition(trans_spec, dim)
            routes_by_in_depth.setdefault(in_d, []).append(out_d)

        self.transitions = ModuleDict(transitions_dict)
        self.recirc_pairs = tuple(parsed_pairs)
        self.routes_by_in_depth = routes_by_in_depth

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    # transform block inputs helper

    def make_transforms(
        self,
        intermediates: LayerIntermediates,
        training = False,
        override_transition = None,
        generate = False
    ) -> dict[int, Callable]:
        fuse_fn = fuse_last_latent if generate else shift_and_fuse_latents
        transforms = dict()

        for in_d, out_depths in self.routes_by_in_depth.items():
            def transform(stream, in_d = in_d, out_depths = out_depths):
                for out_d in out_depths:
                    trans = default(override_transition, self.transitions[f'{out_d}->{in_d}'])
                    latent = get_block_latent(intermediates, out_d)

                    if training and self.jitter_noise_delta > 0.:
                        latent = latent + uniform_like(latent, -self.jitter_noise_delta, self.jitter_noise_delta)

                    stream = fuse_fn(latent, stream, trans)

                return stream

            transforms[in_d] = transform

        return transforms

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

        temporal_parallel_passes = default(temporal_parallel_passes, self.temporal_parallel_passes)
        should_fuse_latent = default(should_fuse_latent, lambda step: True)

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN
            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        prompts = cast_tensor(prompts)
        prompts, inverse_pack = pack_with_inverse(prompts, '* n')
        prompt_len = prompts.shape[-1]

        out = prompts

        # prefill, with optional temporal parallel passes

        logits, intermediates = net(out, return_intermediates = True, **kwargs)

        if process_prompt and temporal_parallel_passes > 1:
            for _ in range(temporal_parallel_passes - 1):
                transforms = self.make_transforms(intermediates, override_transition = transition)
                logits, intermediates = net(out, transform_block_inputs = transforms, return_intermediates = True, **kwargs)

        cache = intermediates

        # autoregressive decoding, with latent feedback from preceding step

        for step in range(seq_len):
            logits = logits[:, -1]

            if temperature == 0.:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token) and (sample == eos_token).all():
                break

            if step == (seq_len - 1):
                break

            # fuse latents onto just sampled token

            transforms = self.make_transforms(intermediates, override_transition = transition, generate = True) if (temporal_parallel_passes > 1 and should_fuse_latent(step)) else None

            logits, intermediates = net(
                out,
                transform_block_inputs = transforms,
                cache = cache,
                return_intermediates = True,
                **kwargs
            )

            cache = intermediates

        return inverse_pack(out[:, prompt_len:])

    # forward

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

        dynamic_rollout_loss_weight = default(dynamic_rollout_loss_weight, self.dynamic_rollout_loss_weight)
        dynamic_loss_decay = default(dynamic_loss_decay, self.dynamic_loss_decay)
        dynamic_loss_threshold = default(dynamic_loss_threshold, self.dynamic_loss_threshold)

        # cross entropy helper

        def ce_loss(logits, target):
            return F.cross_entropy(
                rearrange(logits, 'b n l -> (b n) l'),
                rearrange(target, 'b n -> (b n)'),
                ignore_index = self.ignore_index
            )

        inp, target = (token_ids[:, :-1], token_ids[:, 1:]) if return_loss else (token_ids, None)

        # first pass (standard)

        logits, intermediates = self.net(inp, return_intermediates = True, **kwargs)

        all_logits = [logits]
        first_pass_loss = ce_loss(logits, target) if return_loss else None
        feedback_pass_losses = []

        # feedback passes (temporal parallel)

        for _ in range(temporal_parallel_passes - 1):
            transforms = self.make_transforms(
                intermediates,
                training = self.training,
                override_transition = transition
            )

            logits, intermediates = self.net(
                inp,
                transform_block_inputs = transforms,
                return_intermediates = True,
                **kwargs
            )

            all_logits.append(logits)

            if not return_loss:
                continue

            # feedback loss only on positions that received latent states (1:)

            pred, labels = (logits[:, 1:], target[:, 1:]) if logits.shape[1] > 1 else (logits, target)
            feedback_pass_losses.append(ce_loss(pred, labels))

        if not return_loss:
            return all_logits[-1], dict(all_pass_logits = all_logits)

        # combine first pass loss with feedback pass losses

        all_pass_losses = [first_pass_loss, *feedback_pass_losses]
        all_losses = torch.stack(all_pass_losses)

        feedback_loss = self.zero

        if len(feedback_pass_losses) > 0:
            feedback_losses = all_losses[1:]

            if dynamic_rollout_loss_weight:
                weights = dynamic_rollout_loss_weights(
                    all_losses.detach(),
                    decay = dynamic_loss_decay,
                    threshold = dynamic_loss_threshold
                )[1:]
                feedback_loss = (feedback_losses * weights).sum() / weights.sum().clamp(min = 1e-8)
            else:
                feedback_loss = feedback_losses.mean()

        loss = first_pass_loss + feedback_loss * self.feedback_pass_weight

        loss_breakdown = LossBreakdown(
            first_pass_loss = first_pass_loss,
            feedback_pass_losses = feedback_pass_losses,
            all_pass_losses = all_pass_losses
        )

        return loss, dict(
            loss_breakdown = loss_breakdown,
            all_pass_logits = all_logits
        )
