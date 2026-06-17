"""
Randall Balestriero, Yann LeCun - https://arxiv.org/abs/2511.08544
Hai Huang - https://arxiv.org/abs/2509.14252v2
"""

from __future__ import annotations

import torch
from torch import stack
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

import einx
from einops import rearrange

from x_transformers import Decoder, TransformerWrapper, RMSNorm
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from torch_einops_utils import lens_to_mask, pad_right_at_dim, masked_mean
from torch_einops_utils.nn import Sequential

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t, eps = 1e-6):
    return F.normalize(t, dim = -1, eps = eps)

def scale_grad(t, scale):
    if scale == 1.:
        return t
    if scale == 0.:
        return t.detach()
    return t * scale + t.detach() * (1. - scale)

# classes

class LatentAutoregressive(Module):
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
        sigreg_loss_weight = 0.05,
        l2_loss_weight = 1.,
        num_rollouts = 1,
        rollout_loss_weights: tuple[float, ...] | None = None,
        sigreg_loss_kwargs = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        frac_gradient = 0.,
        predict_next_cosine_sim = True,
        predictor_input_hiddens_index = -1,
        predict_next_embed_with_action = True,
        predict_next_embed_no_action = False,
        detach_target = True,
        ce_probe_module = None, # i.e. extra transformer blocks
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.net = net

        self.ignore_index = ignore_index
        self.pad_value = pad_value

        self.sigreg_loss_weight = sigreg_loss_weight
        self.l2_loss_weight = l2_loss_weight

        self.num_rollouts = num_rollouts
        rollout_loss_weights = default(rollout_loss_weights, (1.,) * num_rollouts)

        assert len(rollout_loss_weights) == num_rollouts, f'rollout_loss_weights must be of length {num_rollouts}'

        rollout_weights = torch.tensor(rollout_loss_weights)
        self.register_buffer('rollout_loss_weights', rollout_weights / rollout_weights.sum(), persistent = False)

        self.sigreg_loss_kwargs = sigreg_loss_kwargs
        self.frac_gradient = frac_gradient
        self.predict_next_cosine_sim = predict_next_cosine_sim
        self.predictor_input_hiddens_index = predictor_input_hiddens_index

        assert predict_next_embed_with_action or predict_next_embed_no_action, 'at least one prediction head must be turned on'

        self.predict_next_embed_with_action = predict_next_embed_with_action
        self.predict_next_embed_no_action = predict_next_embed_no_action
        self.detach_target = detach_target

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        num_tokens = net.num_tokens

        # cross entropy linear probe - mlp on detached embeddings

        net.to_logits = Sequential(
            ce_probe_module,
            RMSNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim),
            net.to_logits
        )

        # predict next embed given (embed, action)

        if predict_next_embed_with_action:
            self.action_emb = nn.Embedding(num_tokens, dim)

            self.to_next_embed_pred = Sequential(
                RMSNorm(dim * 2),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim)
            )

        # predict next embed without sampled token id for that step

        if predict_next_embed_no_action:
            self.to_next_embed_no_action_pred = Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim)
            )

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        **kwargs
    ):
        seq, ignore_index = x.shape[1], self.ignore_index

        # prepare input and target

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        # get embeddings from network

        embed, cache = self.net(
            inp,
            return_embeddings = True,
            return_intermediates = True,
            **kwargs
        )

        embed_prev = embed[:, :-1]

        if exists(self.predictor_input_hiddens_index):
            embed_prev_lower = cache.hiddens[self.predictor_input_hiddens_index][:, :-1]
        else:
            embed_prev_lower = embed_prev

        mask = target != ignore_index

        # cross entropy loss on detached embeddings (linear probe)

        embed_prev_ce = scale_grad(embed_prev, self.frac_gradient)
        logits = self.net.to_logits(embed_prev_ce)

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        num_rollouts = min(self.num_rollouts, seq - 1)

        target_embeds = embed
        if self.detach_target:
            target_embeds = target_embeds.detach()

        if self.predict_next_cosine_sim:
            target_embeds = l2norm(target_embeds)

        # default l2 losses

        l2_loss = self.zero
        l2_no_action_loss = self.zero

        has_predictors = self.predict_next_embed_with_action or self.predict_next_embed_no_action

        if num_rollouts > 0 and has_predictors:

            # pad sequences on the right to handle future predictions smoothly
            # we pad exactly by num_rollouts - 1 to get exactly num_rollouts windows

            pad_amt = num_rollouts - 1

            padded_target_embeds = pad_right_at_dim(target_embeds, pad_amt, dim = 1, value = 0.)
            padded_mask = pad_right_at_dim(mask, pad_amt, dim = 1, value = False)

            # pre-compute targets and masks across all rollout steps using unfold

            step_targets = rearrange(padded_target_embeds[:, 1:].unfold(1, seq - 1, 1), 'b r d n -> r b n d')
            step_masks = rearrange(padded_mask.unfold(1, seq - 1, 1), 'b r n -> r b n')
            step_masks_expanded = rearrange(step_masks, '... -> ... 1')

            # collect predictors

            predictors = []

            if self.predict_next_embed_with_action:
                padded_x = pad_right_at_dim(inp, pad_amt, dim = 1, value = self.pad_value)
                actions = rearrange(padded_x[:, 1:].unfold(1, seq - 1, 1), 'b r n -> r b n')

                predictors.append((True, self.to_next_embed_pred))

            if self.predict_next_embed_no_action:
                predictors.append((False, self.to_next_embed_no_action_pred))

            # iterate over predictors and accumulate rollout losses

            for has_action, predictor in predictors:
                pred = embed_prev_lower
                pred_loss_inputs = []

                for step in range(num_rollouts):
                    pred_input = pred

                    # optionally condition on action

                    if has_action:
                        action_embed = self.action_emb(actions[step])
                        pred_input = torch.cat((pred, action_embed), dim = -1)

                    pred = predictor(pred_input)
                    pred_loss_input = l2norm(pred) if self.predict_next_cosine_sim else pred
                    pred_loss_inputs.append(pred_loss_input)

                # batched mse loss computation across all rollouts

                pred_loss_inputs = stack(pred_loss_inputs)

                step_losses = F.mse_loss(pred_loss_inputs, step_targets, reduction = 'none')

                # apply rollout weights

                weights = self.rollout_loss_weights[:num_rollouts]
                step_losses = einx.multiply('r b n d, r -> r b n d', step_losses, weights)

                # mean over valid tokens

                loss = masked_mean(step_losses, step_masks_expanded, dim = (1, 2, 3)).sum()

                if has_action:
                    l2_loss = loss
                else:
                    l2_no_action_loss = loss

        # sigreg regularization on embeddings

        sreg_loss = self.sigreg_loss(embed_prev[mask], **self.sigreg_loss_kwargs)

        if exists(self.predictor_input_hiddens_index):
            sreg_loss_lower = self.sigreg_loss(embed_prev_lower[mask], **self.sigreg_loss_kwargs)
            sreg_loss = (sreg_loss + sreg_loss_lower) / 2

        # total

        lam = self.sigreg_loss_weight

        total_loss = (
            ce_loss +
            (1. - lam) * l2_loss * self.l2_loss_weight +
            (1. - lam) * l2_no_action_loss * self.l2_loss_weight +
            lam * sreg_loss
        )

        if return_loss_breakdown:
            return total_loss, (ce_loss, l2_loss, l2_no_action_loss, sreg_loss)

        return total_loss

if __name__ == '__main__':
    from x_transformers import Decoder, TransformerWrapper

    transformer = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = 1024,
        attn_layers = Decoder(
            dim = 512,
            depth = 1,
            heads = 8,
            rotary_pos_emb = True,
            pre_norm_has_final_norm = False
        )
    )

    model = LatentAutoregressive(
        transformer,
        dim = 512,
        num_rollouts = 4,
        rollout_loss_weights = [1., 0.5, 0.25, 0.125],
        predict_next_embed_no_action = True
    )

    x = torch.randint(0, 256, (2, 1024))

    loss = model(x)

    assert loss.ndim == 0, 'loss must be a scalar'
    loss.backward()
