# recirculation - https://arxiv.org/abs/2608.17981
# full-bandwidth transformer - https://arxiv.org/abs/2608.08888v1

from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from torch.nn import LayerNorm, Linear, GELU, Sequential
import torch.nn.functional as F

from torch.optim import AdamW

from einops import rearrange

from x_transformers.x_transformers import LayerIntermediates
from torch_einops_utils import cross_entropy

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def logit(p):
    return torch.log(torch.tensor(p) / (1 - p)).item()

# leak residual from a deep layer (source) down to a shallow one (dest), alpha-convex mixture
# multiple routes as tuples, e.g. (7, 8) -> (2, 3); or learn per-token coefficients with a small mlp (adaptive, section 4.6)

def norm_matched(z_source: Tensor, z_dest: Tensor, eps = 1e-8):
    # rescale source to match destination L2 norm (eq. 2)

    source_norm = z_source.norm(dim = -1, keepdim = True)
    dest_norm = z_dest.norm(dim = -1, keepdim = True)

    scale = dest_norm / source_norm.clamp(min = eps)
    scale = scale.masked_fill(source_norm <= eps, 0.)

    return z_source * scale

# the small mlp that learns the mixture coefficients
# 2 hidden layer gelu mlp, layer norm at the input, sigmoid at the output, initialized at alpha = 0.1, beta = 0.9

class RecirculationMixer(Module):
    def __init__(
        self,
        dim,
        hidden = None,
        init_alpha = 0.1,
        init_beta = 0.9
    ):
        super().__init__()
        hidden = default(hidden, dim)

        self.input_norm = LayerNorm(dim * 2)

        self.mlp = Sequential(
            Linear(dim * 2, hidden),
            GELU(),
            Linear(hidden, hidden),
            GELU(),
            Linear(hidden, dim * 2)
        )

        out_layer = self.mlp[-1]

        # init outputs at the desired alpha / beta

        nn.init.normal_(out_layer.weight, std = 0.02)
        nn.init.zeros_(out_layer.bias)

        with torch.no_grad():
            out_layer.bias.copy_(torch.cat((
                torch.full((dim,), logit(init_alpha)),
                torch.full((dim,), logit(init_beta))
            )))

    def forward(self, z_source: Tensor, z_dest: Tensor):
        concat = torch.cat((z_source, z_dest), dim = -1)

        coefficients = self.mlp(self.input_norm(concat)).sigmoid()
        alpha, beta = coefficients.chunk(2, dim = -1)

        return alpha, beta

# recirculation wrapper
# drop in net for the autoregressive wrapper

class Recirculation(Module):
    def __init__(
        self,
        net,
        source_layer,
        destination_layer,
        alpha = 0.15,
        beta = None,
        ramp_steps = 10,            # ramp up alpha over the first 10 tokens (appendix b.3), 0 to disable
        use_learned = False,        # learn per-token alpha / beta with a small mlp (section 4.6)
        learned_hidden = None
    ):
        super().__init__()

        self.net = net

        assert self.net.can_cache_kv, 'recirculation requires the wrapped transformer to be able to cache key / values during decoding'
        assert not self.net.looped, 'recirculation is not compatible with a looped transformer'

        source_layers = cast_tuple(source_layer)
        destination_layers = cast_tuple(destination_layer)

        assert len(source_layers) == len(destination_layers), 'number of source and destination layers must match'
        assert len(set(destination_layers)) == len(destination_layers), 'destination layers must be unique'

        num_routes = len(source_layers)

        assert all(dest < src for src, dest in zip(source_layers, destination_layers)), 'each destination layer must be shallower than its source layer'

        self.source_layers = source_layers
        self.destination_layers = destination_layers

        self.alphas = cast_tuple(alpha, num_routes)
        self.betas = cast_tuple(beta, num_routes) if exists(beta) else (None,) * num_routes

        attn_layers = getattr(net, 'attn_layers', net)
        dim = attn_layers.dim

        self.learned_mixers = ModuleList([RecirculationMixer(dim, learned_hidden) for _ in range(num_routes)]) if use_learned else None

        # ramping only for fixed coefficients; learned mixer sets the mixture per token
        self.ramp_steps = ramp_steps if not exists(self.learned_mixers) else 0

        self.attn_layers = attn_layers
        self.attn_layer_indices = [ind for ind in attn_layers.layers_execute_order if attn_layers.layer_types[ind] == 'a']

        num_attn_layers = len(self.attn_layer_indices)

        for source, destination in zip(source_layers, destination_layers):
            assert source < num_attn_layers, f'source layer {source} is out of range for a model with {num_attn_layers} attention layers'
            assert destination < num_attn_layers, f'destination layer {destination} is out of range for a model with {num_attn_layers} attention layers'

    # attributes needed by the autoregressive wrapper

    @property
    def max_seq_len(self):
        return self.net.max_seq_len

    @property
    def add_continuous_pred_head(self):
        return self.net.add_continuous_pred_head

    @property
    def can_cache_kv(self):
        return self.net.can_cache_kv

    @property
    def can_cache_kv_outside_max_seq_len(self):
        return self.net.can_cache_kv_outside_max_seq_len

    @property
    def looped(self):
        return self.net.looped

    @property
    def output_is_log_prob(self):
        return self.net.output_is_log_prob

    # mixture coefficients for a route, fixed or learned

    def mix_coefficients(self, route, z_source: Tensor, z_dest: Tensor, step):
        if exists(self.learned_mixers):
            return self.learned_mixers[route](z_source, z_dest)

        alpha = self.alphas[route]

        if self.ramp_steps > 0:
            alpha = alpha * min(step / self.ramp_steps, 1.)

        beta = default(self.betas[route], 1. - alpha)

        return alpha, beta

    # the core recirculation mixture (eq. 1)
    # returns the mixed residual stream for each destination layer

    def mix(self, hiddens: list[Tensor], step = 0):
        z_mixed = {}

        for route, (source, destination) in enumerate(zip(self.source_layers, self.destination_layers)):
            z_source = hiddens[source][..., -1:, :]
            z_dest = hiddens[destination][..., -1:, :]

            alpha, beta = self.mix_coefficients(route, z_source, z_dest, step)

            z_mixed[destination] = alpha * norm_matched(z_source, z_dest) + beta * z_dest

        return z_mixed

    # strip the latest token's kv from the cache

    def _strip_latest_kv(self, cache: LayerIntermediates):
        for attn_intermediate in cache.attn_intermediates:
            if attn_intermediate.layer_type != 'a':
                continue

            attn_intermediate.cached_kv = tuple(kv[..., :-1, :] for kv in attn_intermediate.cached_kv)

        cache.cache_length -= 1

    # recirculate the latest token
    # mix source / destination residuals (eq. 1), replay through the upper stack, replacing its kv

    def recirculate(
        self,
        x: Tensor,
        cache: LayerIntermediates,
        seq_start_pos = None,
        kwargs: dict = dict(),
        step = 0
    ):
        hiddens = cache.hiddens

        assert exists(hiddens), 'cache with hiddens required - the net must be called with `return_intermediates = True`'

        position = cache.cache_length - 1

        z_mixed_by_destination = self.mix(hiddens, step = step)

        # strip the first pass kv so the recirculated one replaces it on replay

        self._strip_latest_kv(cache)

        # inject the mixed residual at each destination layer before replay

        hook_handles = []

        for destination, z_mixed in z_mixed_by_destination.items():
            block_index = self.attn_layer_indices[destination]
            norms, block, _ = self.attn_layers.layers[block_index]

            pre_branch_norm = norms[0]
            hook_target = pre_branch_norm if exists(pre_branch_norm) else block

            def inject_z_mixed(module, input, z_mixed = z_mixed):
                (residual,) = input

                residual = torch.cat((residual[..., :-1, :], z_mixed), dim = -2)
                return (residual,)

            hook_handles.append(hook_target.register_forward_pre_hook(inject_z_mixed))

        try:
            # replay only the latest token, at its original position

            device = x.device

            replay_pos = torch.tensor([position], device = device, dtype = torch.long)
            replay_kwargs = dict(pos = replay_pos)

            rotary_pos_emb_module = self.attn_layers.rotary_pos_emb

            if exists(rotary_pos_emb_module):
                rotary_replay_pos = torch.cat((
                    torch.arange(position, device = device),
                    torch.tensor([position], device = device, dtype = torch.long)
                ))

                replay_kwargs.update(rotary_pos_emb = rotary_pos_emb_module(rotary_replay_pos))

            _, replay_cache = self.net(
                x[:, -1:],
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs,
                **replay_kwargs
            )
        finally:
            for hook_handle in hook_handles:
                hook_handle.remove()

        return replay_cache

    # prefill runs token by token (cannot be parallelized per paper)
    # decoding is a single pass on the latest token, followed by recirculation

    def forward(
        self,
        x: Tensor,
        cache: LayerIntermediates | None = None,
        return_intermediates = False,
        seq_start_pos = None,
        **kwargs
    ):
        assert exists(self.net.to_logits), 'the wrapped network needs a to_logits head'

        if not exists(cache):
            return self._prefill(x, return_intermediates, seq_start_pos, kwargs)

        # decoding

        logits, new_cache = self.net(
            x,
            return_intermediates = True,
            cache = cache,
            seq_start_pos = seq_start_pos,
            **kwargs
        )

        new_cache = self.recirculate(x, new_cache, seq_start_pos, kwargs, step = new_cache.cache_length - 1)

        return (logits, new_cache) if return_intermediates else logits

    def _prefill(self, x: Tensor, return_intermediates, seq_start_pos, kwargs: dict):
        net = self.net

        logits = []
        cache = None

        for t in range(x.shape[1]):
            token = x[:, t:(t + 1)]

            _, cache = net(
                token,
                return_intermediates = True,
                cache = cache,
                input_not_include_cache = True,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            # first pass readout (per paper), then recirculate

            hidden = cache.last_hidden[:, -1:]
            logits.append(net.to_logits(hidden))

            cache = self.recirculate(token, cache, seq_start_pos, kwargs, step = t)

        logits = torch.cat(logits, dim = -2)

        return (logits, cache) if return_intermediates else logits

    # train learned mixer with bptt on next token prediction loss, model weights frozen (section 4.6, appendix d.5)

    def learn_parameters(
        self,
        x: Tensor,
        steps = 100,
        optimizer_cls = AdamW,
        optimizer_kwargs = dict(lr = 3e-4, weight_decay = 1e-4),
        batch_size = 32
    ):
        assert exists(self.learned_mixers), 'learned mixer must be turned on for `learn_parameters`'

        x = x.contiguous()

        batch, device = x.shape[0], x.device

        labels = x[:, 1:]

        optim = optimizer_cls(self.learned_mixers.parameters(), **optimizer_kwargs)

        was_training = self.net.training
        self.net.eval()

        for _ in range(steps):
            rand_indices = torch.randint(0, batch, (batch_size,), device = device)
            batch_x = x[rand_indices]

            logits = self.forward(batch_x)[:, :-1]

            loss = cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                labels[rand_indices],
                ignore_index = 0
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

        self.net.train(was_training)

        return loss.item()
