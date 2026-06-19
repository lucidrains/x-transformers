from __future__ import annotations
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Module

import einx
from einops import rearrange
from torch_einops_utils import pad_right_at_dim, masked_mean

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
        next_latent_loss_weight = 1.0,
        kl_loss_weight = 1.0,
        dynamics_hidden_dim = None,
        dynamics_num_layers = 3,
        ignore_index = -100,
        pad_value = 0,
        rollout_weights: tuple[float, ...] | None = None,
        sigreg_loss_weight = 0.,
        sigreg_loss_kwargs: dict = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        )
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

        # rollout weights

        if not exists(rollout_weights):
            rollout_weights = tuple([1.] * num_rollouts)

        assert len(rollout_weights) == num_rollouts
        rollout_weights = torch.tensor(rollout_weights)
        self.register_buffer('rollout_loss_weights', rollout_weights / rollout_weights.sum(), persistent = False)

        dynamics_hidden_dim = default(dynamics_hidden_dim, dim)

        # token embedding for next-token input to dynamics model

        self.token_emb = net.token_emb

        # latent dynamics model

        self.dynamics_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            MLP(
                dim_in = dim * 2,
                dim_out = dim,
                dim_hidden = dynamics_hidden_dim,
                depth = dynamics_num_layers
            )
        )

        # init last layer to zero for residual update

        nn.init.zeros_(self.dynamics_mlp[-1][-1].weight)
        nn.init.zeros_(self.dynamics_mlp[-1][-1].bias)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

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

        step_targets = padded_target_hiddens[:, 1:].unfold(1, seq - 1, 1)
        step_targets = rearrange(step_targets, 'b r d n -> r b n d')

        step_masks = padded_mask.unfold(1, seq - 1, 1)
        step_masks = rearrange(step_masks, 'b r n -> r b n')

        step_next_tokens = padded_token_embeds[:, 1:].unfold(1, seq - 1, 1)
        step_next_tokens = rearrange(step_next_tokens, 'b r d n -> r b n d')

        # frozen copy of output head for kl loss

        if self.has_kl_loss:
            frozen_to_logits = deepcopy(self.net.to_logits)
            for p in frozen_to_logits.parameters():
                p.requires_grad_(False)

        pred_loss_inputs = []
        pred = hidden_states[:, :-1]

        for step in range(num_rollouts):

            step_next_embeds = step_next_tokens[step].detach()

            dynamics_input = torch.cat((pred, step_next_embeds), dim = -1)

            # they learn the residual with the dynamics mlp

            delta = self.dynamics_mlp(dynamics_input)
            pred = pred + delta

            pred_loss_inputs.append(pred)

        pred_loss_inputs = torch.stack(pred_loss_inputs)

        next_latent_loss = self.zero
        kl_loss = self.zero

        weights = self.rollout_loss_weights[:num_rollouts]
        step_masks_expanded = rearrange(step_masks, '... -> ... 1')

        # smooth l1 with stop-gradient on target

        if self.has_next_latent_loss:
            step_smooth_l1 = F.smooth_l1_loss(
                pred_loss_inputs,
                step_targets.detach(),
                reduction = 'none'
            )

            step_smooth_l1 = einx.multiply('r b n d, r -> r b n d', step_smooth_l1, weights)
            next_latent_loss = masked_mean(step_smooth_l1, step_masks_expanded, dim = (1, 2, 3)).sum()

        # kl divergence

        if self.has_kl_loss:
            with torch.no_grad():
                target_log_probs = F.log_softmax(frozen_to_logits(step_targets.detach()), dim = -1)

            pred_log_probs = F.log_softmax(frozen_to_logits(pred_loss_inputs), dim = -1)

            step_kl = F.kl_div(
                pred_log_probs,
                target_log_probs,
                log_target = True,
                reduction = 'none'
            )

            step_kl = einx.sum('... c -> ...', step_kl)
            step_kl = einx.multiply('r b n, r -> r b n', step_kl, weights)

            kl_loss = masked_mean(step_kl, step_masks, dim = (1, 2)).sum()

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
