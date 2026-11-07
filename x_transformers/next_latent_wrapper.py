from __future__ import annotations
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, stack, cat, tensor
from torch.nn import Module

import einx
from einops import rearrange, reduce
from torch_einops_utils import (
    pad_right_at_dim,
    pad_at_dim,
    masked_mean,
    and_masks,
    exclusive_cumsum,
    pack_with_inverse,
    zero_with_grad,
    cross_entropy
)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

# dynamic rollout loss weighting
# weight each rollout step by exp(-decay * cumulative loss of preceding steps), as errors compound through the dynamics model

def dynamic_rollout_loss_weights(
    step_losses, # (r b n)
    decay = 1.
):
    cum_step_losses = exclusive_cumsum(step_losses, dim = 0)
    return (-decay * cum_step_losses).exp()

# constants

Losses = namedtuple('Losses', [
    'ce',
    'next_latent',
    'kl',
    'sigreg',
    'light_mtp',
    'light_mtp_var_reg',
    'mtp'
])

# classes

def MLP(
    dim_in,
    dim_out,
    dim_hidden,
    depth = 3
):
    layers = []

    for i in range(depth):
        is_last = i == (depth - 1)

        in_d = dim_in if i == 0 else dim_hidden
        out_d = dim_out if is_last else dim_hidden

        layers.append(nn.Linear(in_d, out_d))

        if not is_last:
            layers.append(nn.GELU())

    return nn.Sequential(*layers)

class ResidualDynamics(Module):
    def __init__(
        self,
        net: Module
    ):
        super().__init__()
        self.net = net

    def forward(
        self,
        rollout_next_token_embeds, # (r b n d)
        curr_latent                # (b n d)
    ):
        next_embeds, unpack_next_embeds = pack_with_inverse(rollout_next_token_embeds, 'r * d')
        curr_latent, _ = pack_with_inverse(curr_latent, '* d')

        next_latents = []

        for next_embed in next_embeds:
            dynamics_input = cat((curr_latent, next_embed), dim = -1)
            curr_latent = curr_latent + self.net(dynamics_input)

            next_latents.append(curr_latent)

        out = stack(next_latents)
        return unpack_next_embeds(out, 'r * d')

class RolloutGRU(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.gru = nn.GRU(input_size = dim, hidden_size = dim, batch_first = False)

    def forward(
        self,
        rollout_next_token_embeds, # (r b n d)
        curr_latent                # (b n d)
    ):

        # pack batch and sequence dimensions

        next_embeds, unpack_next_embeds = pack_with_inverse(rollout_next_token_embeds, 'r * d')

        first_hidden, _ = pack_with_inverse(curr_latent, '* d')
        first_hidden = rearrange(first_hidden, '... -> 1 ...')

        # run gru over the rollout sequence length

        out, _ = self.gru(next_embeds, first_hidden)

        # unpack

        return unpack_next_embeds(out, 'r * d')

# loss modules

class MSECosineSimLoss(Module):
    def __init__(
        self,
        weight = 0.9
    ):
        super().__init__()
        self.weight = weight

    def forward(self, pred, target):
        mse_loss = F.mse_loss(pred, target, reduction = 'none')
        cos_loss = 1. - F.cosine_similarity(pred, target, dim = -1)
        cos_loss = rearrange(cos_loss, '... -> ... 1')

        return mse_loss.lerp(cos_loss, self.weight)

# lightmtp - https://arxiv.org/abs/2610.06031
# predict future hidden states from the current hidden state alone, in one parallel head call

class LightMTP(Module):
    def __init__(
        self,
        dim,
        num_steps = 4,
        variant = 'concat',
        head_hidden_dim = None,
        has_var_reg = False
    ):
        super().__init__()
        assert variant in ('single', 'concat')
        assert num_steps > 0

        self.num_steps = num_steps
        self.variant = variant
        self.has_var_reg = has_var_reg

        out_dim = dim * num_steps if variant == 'concat' else dim

        # two layer mlp with gelu, as in the paper

        self.head = MLP(
            dim_in = dim,
            dim_out = out_dim,
            dim_hidden = default(head_hidden_dim, dim),
            depth = 2
        )

    def variance_regularization(self, target_hiddens, mask):
        # penalize a shared mean direction across positions
        normed = l2norm(target_hiddens)
        mean_hidden = masked_mean(normed, mask, dim = 1)
        return mean_hidden.norm(dim = -1).mean()

    def forward(
        self,
        hiddens,          # (b n d) final layer
        target_hiddens,   # (b n d) intermediate layer targets
        mask              # (b n) valid tokens
    ):
        n = hiddens.shape[1]
        preds = self.head(hiddens)

        # varreg shapes the trunk, so keep its gradient

        if self.has_var_reg:
            var_reg_loss = self.variance_regularization(target_hiddens, mask)
        else:
            var_reg_loss = hiddens.new_zeros(())

        target_hiddens = target_hiddens.detach()

        if self.variant == 'single':
            preds = rearrange(preds, 'b n d -> b n 1 d')
            steps = (self.num_steps,)
        else:
            preds = rearrange(preds, 'b n (k d) -> b n k d', k = self.num_steps)
            steps = tuple(range(1, self.num_steps + 1))

        step_losses = []

        for step in steps:
            if step >= n:
                continue

            step_index = step - 1 if self.variant == 'concat' else 0
            pred = preds[:, :n - step, step_index]

            # input and target positions both valid

            step_mask = and_masks((mask[:, :n - step], mask[:, step:]))
            cos_loss = 1. - F.cosine_similarity(pred, target_hiddens[:, step:], dim = -1)
            step_losses.append(masked_mean(cos_loss, step_mask))

        if not step_losses:
            zero = hiddens.new_zeros(())
            return zero, zero

        return sum(step_losses), var_reg_loss

# token-level mtp
# standard (hard) and soft-mtp from https://arxiv.org/abs/2610.38331

class TokenMTP(Module):
    def __init__(
        self,
        dim,
        num_steps = 3,
        soft = False,
        head_type = 'linear',
        soft_loss_weight = 1.0
    ):
        super().__init__()
        assert num_steps > 0
        assert head_type in ('linear', 'mlp')

        self.num_steps = num_steps
        self.soft = soft
        self.soft_loss_weight = soft_loss_weight

        if head_type == 'linear':
            heads = [nn.Linear(dim, dim) for _ in range(num_steps)]
        else:
            heads = [MLP(dim, dim, dim, depth = 2) for _ in range(num_steps)]

        self.heads = nn.ModuleList(heads)

    def forward(
        self,
        hiddens,      # (b n d)
        tokens,       # (b n) token ids
        to_logits,
        ignore_index = -100
    ):
        n = hiddens.shape[1]
        steps = tuple(range(1, min(self.num_steps, n - 2) + 1))

        if not steps:
            return hiddens.new_zeros(()), None

        mtp_losses = []
        straight_through = []

        # number of valid ntp targets, for undoing the ce mean

        n0 = (tokens[:, 1:] != ignore_index).sum().clamp(min = 1)

        for step in steps:
            head = self.heads[step - 1]

            num_pred = n - step - 1
            hk = head(hiddens[:, :num_pred])
            targets = tokens[:, step + 1:]

            logits = to_logits(hk)
            mtp_losses.append(cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                targets,
                ignore_index = ignore_index
            ))

            if self.soft:
                # s_k = lambda * N0 / (K * N_k), as in appendix b.5 of the paper
                nk = (targets != ignore_index).sum().clamp(min = 1)
                s_k = (self.soft_loss_weight * n0 / (len(steps) * nk)).detach()

                # h0[t + step] += s_k * (hk[t].detach() - hk[t]), numerically zero
                straight_through.append((step, -s_k * zero_with_grad(hk)))

        mtp_loss = sum(mtp_losses) / len(mtp_losses)

        if not self.soft:
            return mtp_loss, None

        h0 = hiddens

        for step, delta in straight_through:
            h0 = h0 + pad_at_dim(delta, (step, 1), dim = 1)

        logits = to_logits(h0[:, :-1])
        ce_loss = cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            tokens[:, 1:],
            ignore_index = ignore_index
        )

        return mtp_loss, ce_loss

# main next latent wrapper

class NextLatentWrapper(Module):
    @staticmethod
    def sigreg_loss(
        x,
        num_slices = 1024,
        domain = (-5, 5),
        num_knots = 17
    ):
        dim, device = x.shape[-1], x.device

        rand_projs = torch.randn((num_slices, dim), device = device)
        rand_projs = l2norm(rand_projs)

        t = torch.linspace(*domain, num_knots, device = device)

        exp_f = (-0.5 * t.square()).exp()

        x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
        x_t = rearrange(x_t, '... m -> (...) m')

        x_t = rearrange(x_t, 'n m -> n m 1') * t
        ecf = (1j * x_t).exp().mean(dim = 0)

        err = ecf.sub(exp_f).abs().square().mul(exp_f)

        return torch.trapezoid(err, t, dim = -1).mean()

    def __init__(
        self,
        net,
        *,
        dim,
        num_rollouts = 1,
        loss_type: str | Module = 'smooth_l1',
        dynamics_type: str = 'residual',
        dynamics_network: Module | None = None,
        dynamics_hidden_dim = None,
        dynamics_num_layers = 3,
        next_latent_loss_weight = 1.0,
        kl_loss_weight = 1.0,
        ignore_index = -100,
        pad_value = 0,
        rollout_weights: tuple[float, ...] | None = None,
        sigreg_loss_weight = 0.,
        sigreg_loss_kwargs: dict = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        dynamic_rollout_loss_weight = True,
        dynamic_loss_decay = 1.0,
        light_mtp_loss_weight = 0.,
        light_mtp_variant = 'concat',
        light_mtp_num_steps = 4,
        light_mtp_target_layer: int | None = None,
        light_mtp_head_hidden_dim: int | None = None,
        light_mtp_var_reg_weight = 0.,
        mtp_loss_weight = 0.,
        mtp_num_steps = 3,
        mtp_soft = False,
        mtp_head_type = 'linear'
    ):
        super().__init__()
        self.net = net
        self.ignore_index = ignore_index
        self.pad_value = pad_value

        assert num_rollouts > 0, 'num_rollouts must be greater than 0'
        self.num_rollouts = num_rollouts

        self.loss_type = loss_type

        if isinstance(loss_type, Module):
            self.loss_fn = loss_type
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction = 'none')
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction = 'none')
        elif loss_type == 'mse_and_cosine_sim':
            # https://arxiv.org/abs/2606.30534
            self.loss_fn = MSECosineSimLoss()
        else:
            raise ValueError(f'unknown loss type {loss_type}')

        self.next_latent_loss_weight = next_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

        self.has_next_latent_loss = next_latent_loss_weight > 0.
        self.has_kl_loss = kl_loss_weight > 0.
        self.has_sigreg = sigreg_loss_weight > 0.

        self.dynamic_rollout_loss_weight = dynamic_rollout_loss_weight
        self.dynamic_loss_decay = dynamic_loss_decay

        # rollout weights

        if not exists(rollout_weights):
            rollout_weights = (1.,) * num_rollouts

        assert len(rollout_weights) == num_rollouts
        rollout_weights = tensor(rollout_weights)
        self.register_buffer('rollout_loss_weights', rollout_weights / rollout_weights.sum(), persistent = False)

        # token embedding for next-token input to dynamics model

        self.token_emb = net.token_emb

        # latent dynamics model

        assert dynamics_type in ('residual', 'gru', 'custom')

        if dynamics_type == 'residual':
            if not exists(dynamics_network):
                dynamics_hidden_dim = default(dynamics_hidden_dim, dim)
                dynamics_network = nn.Sequential(
                    nn.LayerNorm(dim * 2),
                    MLP(
                        dim_in = dim * 2,
                        dim_out = dim,
                        dim_hidden = dynamics_hidden_dim,
                        depth = dynamics_num_layers
                    )
                )

                nn.init.zeros_(dynamics_network[-1][-1].weight)
                nn.init.zeros_(dynamics_network[-1][-1].bias)

            self.dynamics_model = ResidualDynamics(dynamics_network)

        elif dynamics_type == 'gru':
            self.dynamics_model = RolloutGRU(dim = dim)

        elif dynamics_type == 'custom':
            assert exists(dynamics_network)
            self.dynamics_model = dynamics_network

        # lightmtp auxiliary loss

        self.has_light_mtp = light_mtp_loss_weight > 0. or light_mtp_var_reg_weight > 0.
        self.light_mtp_loss_weight = light_mtp_loss_weight
        self.light_mtp_var_reg_weight = light_mtp_var_reg_weight
        self.light_mtp_target_layer = light_mtp_target_layer

        if self.has_light_mtp:
            self.light_mtp = LightMTP(
                dim = dim,
                num_steps = light_mtp_num_steps,
                variant = light_mtp_variant,
                head_hidden_dim = light_mtp_head_hidden_dim,
                has_var_reg = light_mtp_var_reg_weight > 0.
            )

        # token-level mtp, optionally soft-mtp

        self.has_mtp = mtp_loss_weight > 0.
        self.mtp_loss_weight = mtp_loss_weight

        if self.has_mtp:
            self.mtp = TokenMTP(
                dim = dim,
                num_steps = mtp_num_steps,
                soft = mtp_soft,
                head_type = mtp_head_type,
                soft_loss_weight = mtp_loss_weight
            )

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        **kwargs
    ):
        seq, ignore_index, num_rollouts = x.shape[1], self.ignore_index, self.num_rollouts

        assert seq > (num_rollouts + 1), f'sequence length ({seq}) must be greater than num_rollouts ({num_rollouts}) + 1'

        # prepare input and target

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # get hidden states from transformer

        hidden_states, intermediates = self.net(
            inp,
            return_embeddings_and_intermediates = True,
            **kwargs
        )

        mask = target != ignore_index

        # token-level mtp, optionally replacing ce with its soft-mtp version

        ce_loss = None
        mtp_loss = self.zero

        if self.has_mtp:
            mtp_loss, ce_loss = self.mtp(hidden_states, x, self.net.to_logits, ignore_index)

        if not exists(ce_loss):
            logits = self.net.to_logits(hidden_states[:, :-1])
            ce_loss = cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                target,
                ignore_index = ignore_index
            )

        # lightmtp auxiliary loss

        light_mtp_loss = self.zero
        light_mtp_var_reg = self.zero

        if self.has_light_mtp:
            layer_hiddens = intermediates.layer_hiddens
            target_layer = default(self.light_mtp_target_layer, round((2 / 3) * (len(layer_hiddens) - 1)))

            assert 0 <= target_layer < len(layer_hiddens), f'target layer {target_layer} is out of range for {len(layer_hiddens)} layer hiddens'

            light_mtp_loss, light_mtp_var_reg = self.light_mtp(
                hidden_states,
                layer_hiddens[target_layer],
                x != ignore_index
            )

        # latent dynamics losses

        token_embeds = self.token_emb(inp)

        pad_amt = num_rollouts - 1

        padded_target_hiddens = pad_right_at_dim(hidden_states, pad_amt, dim = 1, value = 0.)
        padded_mask = pad_right_at_dim(mask, pad_amt, dim = 1, value = False)
        padded_token_embeds = pad_right_at_dim(token_embeds, pad_amt, dim = 1, value = 0.)

        rollout_target_hiddens = padded_target_hiddens[:, 1:].unfold(1, seq - 1, 1)
        rollout_target_hiddens = rearrange(rollout_target_hiddens, 'b r d n -> r b n d')

        rollout_masks = padded_mask.unfold(1, seq - 1, 1)
        rollout_masks = rearrange(rollout_masks, 'b r n -> r b n')

        rollout_next_token_embeds = padded_token_embeds[:, 1:].unfold(1, seq - 1, 1)
        rollout_next_token_embeds = rearrange(rollout_next_token_embeds, 'b r d n -> r b n d')

        # frozen copy of output head for kl loss

        if self.has_kl_loss:
            frozen_to_logits = deepcopy(self.net.to_logits)
            for p in frozen_to_logits.parameters():
                p.requires_grad_(False)

        curr_latent = hidden_states[:, :-1]

        dynamics_out = self.dynamics_model(
            rollout_next_token_embeds.detach(),
            curr_latent
        )

        next_latent_loss = self.zero
        kl_loss = self.zero

        weights = self.rollout_loss_weights[:num_rollouts]
        step_masks_expanded = rearrange(rollout_masks, '... -> ... 1')

        # compute latent loss

        step_latent_loss_unreduced = self.loss_fn(
            dynamics_out,
            rollout_target_hiddens.detach()
        )

        # dynamic rollout loss weight

        dynamic_weights = 1.

        if self.dynamic_rollout_loss_weight:
            step_latent_loss = reduce(step_latent_loss_unreduced.detach(), 'r b n d -> r b n', 'mean')
            dynamic_weights = dynamic_rollout_loss_weights(step_latent_loss, self.dynamic_loss_decay)

        # smooth l1 with stop-gradient on target

        if self.has_next_latent_loss:
            step_latent_loss_unreduced = einx.multiply('r b n d, r -> r b n d', step_latent_loss_unreduced, weights)

            if self.dynamic_rollout_loss_weight:
                step_latent_loss_unreduced = einx.multiply('r b n d, r b n -> r b n d', step_latent_loss_unreduced, dynamic_weights)

            next_latent_loss = masked_mean(step_latent_loss_unreduced, step_masks_expanded, dim = (1, 2, 3)).sum()

        # kl divergence

        if self.has_kl_loss:
            with torch.no_grad():
                target_log_probs = F.log_softmax(frozen_to_logits(rollout_target_hiddens.detach()), dim = -1)

            pred_log_probs = F.log_softmax(frozen_to_logits(dynamics_out), dim = -1)

            step_kl = F.kl_div(
                pred_log_probs,
                target_log_probs,
                log_target = True,
                reduction = 'none'
            )

            step_kl = einx.sum('... c -> ...', step_kl)
            step_kl = einx.multiply('r b n, r -> r b n', step_kl, weights)

            if self.dynamic_rollout_loss_weight:
                step_kl = step_kl * dynamic_weights

            kl_loss = masked_mean(step_kl, rollout_masks, dim = (1, 2)).sum()

        # sigreg regularization on embeddings

        sigreg_loss = self.zero

        if self.has_sigreg:
            sigreg_loss = self.sigreg_loss(hidden_states[mask], **self.sigreg_loss_kwargs)

        # total

        total_loss = (
            ce_loss +
            self.next_latent_loss_weight * next_latent_loss +
            self.kl_loss_weight * kl_loss +
            self.sigreg_loss_weight * sigreg_loss +
            self.light_mtp_loss_weight * light_mtp_loss +
            self.light_mtp_var_reg_weight * light_mtp_var_reg +
            self.mtp_loss_weight * mtp_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, Losses(
            ce_loss,
            next_latent_loss,
            kl_loss,
            sigreg_loss,
            light_mtp_loss,
            light_mtp_var_reg,
            mtp_loss
        )

# quick test

if __name__ == '__main__':
    from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8
        )
    )

    wrapper = NextLatentWrapper(
        model,
        dim = 512,
        num_rollouts = 3
    )

    token_ids = torch.randint(0, 256, (2, 1024))

    # forward and backwards on data as usual

    loss = wrapper(token_ids)
    loss.backward()

    # after much extract the original transformer
    # wrap it in the usual autoregressive wrapper and generate with the improved representation space

    autoregressive_model = AutoregressiveWrapper(model)
    generated = autoregressive_model.generate(token_ids[:, :1], 10)

    print(generated.shape) # (2, 11)
