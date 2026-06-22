from __future__ import annotations
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, stack, cat, tensor
from torch.nn import Module

import einx
from einops import rearrange, reduce
from torch_einops_utils import pad_right_at_dim, masked_mean, exclusive_cumsum, pack_with_inverse

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

# constants

Losses = namedtuple('Losses', ['ce', 'next_latent', 'kl', 'sigreg'])

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
        dynamic_loss_threshold = 0.5
    ):
        super().__init__()
        self.net = net
        self.ignore_index = ignore_index
        self.pad_value = pad_value

        assert num_rollouts > 0, 'num_rollouts must be greater than 0'
        self.num_rollouts = num_rollouts

        # loss weights

        self.next_latent_loss_weight = next_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs

        self.has_next_latent_loss = next_latent_loss_weight > 0.
        self.has_kl_loss = kl_loss_weight > 0.
        self.has_sigreg = sigreg_loss_weight > 0.

        self.dynamic_rollout_loss_weight = dynamic_rollout_loss_weight
        self.dynamic_loss_decay = dynamic_loss_decay
        self.dynamic_loss_threshold = dynamic_loss_threshold

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

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        **kwargs
    ):
        seq, ignore_index, num_rollouts = x.shape[1], self.ignore_index, self.num_rollouts

        # prepare input and target

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # get hidden states from transformer

        hidden_states, intermediates = self.net(
            inp,
            return_embeddings_and_intermediates = True,
            **kwargs
        )

        # ce loss

        logits = self.net.to_logits(hidden_states[:, :-1])

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        mask = target != ignore_index

        # latent dynamics losses

        assert seq > (num_rollouts + 1), f'sequence length ({seq}) must be greater than num_rollouts ({num_rollouts}) + 1'

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

        # compute smooth l1 loss

        step_smooth_l1 = F.smooth_l1_loss(
            dynamics_out,
            rollout_target_hiddens.detach(),
            reduction = 'none'
        )

        # dynamic rollout loss weight

        dynamic_weights = 1.

        if self.dynamic_rollout_loss_weight:
            step_latent_loss = reduce(step_smooth_l1.detach(), 'r b n d -> r b n', 'mean')

            cum_step_latent_loss = exclusive_cumsum(step_latent_loss, dim = 0)
            dynamic_weights = torch.sigmoid(-self.dynamic_loss_decay * (cum_step_latent_loss - self.dynamic_loss_threshold))

        # smooth l1 with stop-gradient on target

        if self.has_next_latent_loss:
            step_smooth_l1 = einx.multiply('r b n d, r -> r b n d', step_smooth_l1, weights)

            if self.dynamic_rollout_loss_weight:
                step_smooth_l1 = einx.multiply('r b n d, r b n -> r b n d', step_smooth_l1, dynamic_weights)

            next_latent_loss = masked_mean(step_smooth_l1, step_masks_expanded, dim = (1, 2, 3)).sum()

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
            self.sigreg_loss_weight * sigreg_loss
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, Losses(ce_loss, next_latent_loss, kl_loss, sigreg_loss)

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
