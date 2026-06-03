"""
LeJePA: Lean Joint-Embedding Predictive Architecture
Randall Balestriero, Yann LeCun - https://arxiv.org/abs/2511.08544
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module

from einops import rearrange

from x_transformers import Decoder, TransformerWrapper, RMSNorm
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from torch_einops_utils import lens_to_mask
from torch_einops_utils.nn import Sequential

# helpers

def exists(val):
    return val is not None

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
        sigreg_loss_kwargs = dict(
            num_slices = 1024,
            domain = (-5, 5),
            num_knots = 17
        ),
        frac_gradient = 0.,
        predict_next_cosine_sim = True,
        predict_next_embed_with_action = True,
        predict_next_embed_no_action = True,
        ce_probe_module = None, # i.e. extra transformer blocks
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.net = net

        self.ignore_index = ignore_index
        self.pad_value = pad_value

        self.sigreg_loss_weight = sigreg_loss_weight
        self.sigreg_loss_kwargs = sigreg_loss_kwargs
        self.frac_gradient = frac_gradient
        self.predict_next_cosine_sim = predict_next_cosine_sim

        assert predict_next_embed_with_action or predict_next_embed_no_action, 'at least one prediction head must be turned on'

        self.predict_next_embed_with_action = predict_next_embed_with_action
        self.predict_next_embed_no_action = predict_next_embed_no_action

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        num_tokens = net.num_tokens

        # cross entropy linear probe - mlp on detached embeddings

        net.to_logits = Sequential(
            ce_probe_module,
            RMSNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.LeakyReLU(),
            nn.Linear(dim * 2, dim),
            net.to_logits
        )

        # predict next embed given (embed, action)

        if predict_next_embed_with_action:
            self.to_next_embed_pred = Sequential(
                RMSNorm(dim * 2),
                nn.Linear(dim * 2, dim * 2),
                nn.LeakyReLU(),
                nn.Linear(dim * 2, dim)
            )

        # predict next embed given ONLY embed (no action)

        if predict_next_embed_no_action:
            self.to_next_embed_no_action_pred = Sequential(
                RMSNorm(dim),
                nn.Linear(dim, dim * 2),
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
        mask = target != ignore_index

        # cross entropy loss on detached embeddings (linear probe)

        embed_prev_ce = scale_grad(embed_prev, self.frac_gradient)
        logits = self.net.to_logits(embed_prev_ce)

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        # l2 loss - predict next representation from current representation and action token

        l2_loss = self.zero
        l2_no_action_loss = self.zero

        target_embed = embed[:, 1:]

        if self.predict_next_cosine_sim:
            target_embed = l2norm(target_embed)

        if self.predict_next_embed_with_action:
            with torch.no_grad():
                action_embed = self.net.token_emb(target)

            pred_input = torch.cat((embed_prev, action_embed), dim = -1)
            pred_next_embed = self.to_next_embed_pred(pred_input)

            if self.predict_next_cosine_sim:
                pred_next_embed = l2norm(pred_next_embed)

            l2_loss = F.mse_loss(pred_next_embed, target_embed, reduction = 'none')
            l2_loss = l2_loss[mask].mean()

        if self.predict_next_embed_no_action:
            pred_next_embed_no_action = self.to_next_embed_no_action_pred(embed_prev)

            if self.predict_next_cosine_sim:
                pred_next_embed_no_action = l2norm(pred_next_embed_no_action)

            l2_no_action_loss = F.mse_loss(pred_next_embed_no_action, target_embed, reduction = 'none')
            l2_no_action_loss = l2_no_action_loss[mask].mean()

        # sigreg regularization on embeddings

        sreg_loss = self.sigreg_loss(embed_prev[mask], **self.sigreg_loss_kwargs)

        # total

        lam = self.sigreg_loss_weight

        total_loss = (
            ce_loss +
            (1. - lam) * l2_loss +
            (1. - lam) * l2_no_action_loss +
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
        dim = 512
    )

    x = torch.randint(0, 256, (2, 1024))

    loss = model(x)

    assert loss.ndim == 0, 'loss must be a scalar'
    loss.backward()
