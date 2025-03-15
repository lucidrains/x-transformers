
# Belief State Transformer

# Hu et al. https://arxiv.org/abs/2410.23506
# https://www.youtube.com/watch?v=aqhbRtB2Fyg

from __future__ import annotations
from random import random

import torch
from torch.autograd import Function
from torch.nn import Module, ModuleList
from torch import nn, cat, stack, tensor, Tensor, arange, cartesian_prod
import torch.nn.functional as F

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    min_p,
)

from x_transformers.x_transformers import (
    Decoder,
    TransformerWrapper
)

import einx
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# a custom flip that can handle variable lengths across batch

def flip(x, dim = 1, lens = None):
    if not exists(lens):
        return x.flip(dim)

    batch, seq_len, device = *x.shape[:2], x.device
    seq = arange(seq_len, device = device)

    mask = einx.less('j, i -> i j', seq, lens)
    masked_seq = einx.where('i j, j,', mask, seq, -1)

    flip_indices = masked_seq.argsort(dim = -1, descending = True)

    if x.ndim == 3:
        flip_indices = repeat(flip_indices, '... -> ... d', d = x.shape[-1])

    return x.gather(dim, flip_indices)

# detach multiple tensors and backward the gradients once

class DetachMultiple(Function):

    @classmethod
    def forward(self, ctx, *tensors):
        detached_tensors = tuple(t.detach() for t in tensors)

        for detached_tensor in detached_tensors:
            detached_tensor.requires_grad_()

        return detached_tensors

    @classmethod
    def backward(self, ctx, *grads):

        return grads

detach_multiple = DetachMultiple.apply

# wrappers

class BeliefStateWrapper(Module):
    """
    Figure 13. in https://arxiv.org/abs/2410.23506
    """

    def __init__(
        self,
        forward_decoder: TransformerWrapper,
        backward_decoder: TransformerWrapper | None = None,
        train_frac_forward_backward_pairs: float = 1.,
        text_head: Module | None = None,
        backward_ar_loss_weight: float = 1., # can weigh the training of the backwards decoder differently, perhaps fwd/bwd have a shared backbone etc etc
        pred_distance = False,
        pred_distance_loss_weight: float = 1.,
        cond_on_distance = False,
        cond_on_distance_prob = 0.5,
        max_pred_distance = None
    ):
        super().__init__()
        backward_decoder = default(backward_decoder, forward_decoder) # if backward decoder not set, use the same transformer, assume it knows how to switch gears based on suffix token

        assert forward_decoder.emb_dim == backward_decoder.emb_dim, 'forward and backwards model must have the same embedding dimension'
        assert forward_decoder.num_tokens == backward_decoder.num_tokens, 'forward and backwards model must have the same number of tokens'

        dim = forward_decoder.emb_dim
        num_tokens = forward_decoder.num_tokens
        max_seq_len = forward_decoder.max_seq_len

        self.num_tokens = num_tokens

        # the suffix token

        self.suffix_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.suffix_token, std = 0.02)

        # the text prediction head, which predicts for the combinations of prefix and suffix the next and previous token for forwards and backward sequences

        if not exists(text_head):
            text_head = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, num_tokens * 2),
            )

        self.text_head = text_head

        # predicting terminal state (when suffix and prefix predict the same token)

        self.max_pred_distance = default(max_pred_distance, max_seq_len)

        self.to_distance_logits = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, self.max_pred_distance),
        ) if pred_distance else None

        self.pred_distance_loss_weight = pred_distance_loss_weight

        # conditioning on distance

        assert 0. < cond_on_distance_prob < 1.

        self.cond_on_distance = cond_on_distance
        self.cond_on_distance_prob = cond_on_distance_prob

        if cond_on_distance:
            self.to_distance_cond = nn.Sequential(
                Rearrange('... -> ... 1'),
                nn.Linear(1, dim),
                nn.LeakyReLU(),
                nn.Linear(dim, dim * 2),
            )

        # the two decoders, one which is causal forward, the other causal backwards

        self.forward_decoder = forward_decoder
        self.backward_decoder = backward_decoder

        # what fraction of forward backward pairs to train on
        # for further memory efficiency

        assert 0 < train_frac_forward_backward_pairs <= 1.
        self.train_frac_fb_pairs = train_frac_forward_backward_pairs
        self.needs_subsample_fb_pairs = train_frac_forward_backward_pairs < 1.

        # loss weighting

        self.backward_ar_loss_weight = backward_ar_loss_weight
        self.needs_loss_weight = backward_ar_loss_weight != 1.

        self.register_buffer('loss_weights', tensor([1., self.backward_ar_loss_weight]))

        # sampling

        self.max_seq_len = self.forward_decoder.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate_with_suffix_cond(
        self,
        prompts,
        seq_len,
        temperature = 1.25,
        cache_kv = False,
        suffix: Tensor | None = None, # the goal conditioning
        filter_logits_fn = min_p,
        filter_kwargs = dict(
            min_p = 0.1
        ),
        decode_backwards = False,
        **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        prompts, batch_ps = pack([prompts], '* d')

        batch, orig_seq_len = prompts.shape

        # allow for decoding backwards, to make sure it is working

        main_decoder = self.forward_decoder

        if decode_backwards:
            prompts = prompts.flip(1)
            main_decoder = self.backward_decoder

        out = prompts

        # kv caches

        cache = None

        # get the encoded suffix token once

        suffix_sos_tokens = rearrange(self.suffix_token, 'd -> 1 1 d')

        suffix_sos_tokens = repeat(suffix_sos_tokens, '1 1 d -> b 1 d', b = batch)

        if not decode_backwards:
            if exists(suffix):
                if suffix.ndim == 1:
                    suffix = repeat(suffix, 'n -> b n', b = batch)

                suffix = suffix.flip(1) # reverse autoregressive

            suffix_embed = self.backward_decoder(
                suffix,
                prepend_embeds = suffix_sos_tokens,
                return_embeddings = True
            )

            # pick out the last embedding for fill in the middle

            suffix_embed = suffix_embed[:, -1:]

        else:
            # just grab a random token for now for prefix

            prefix_embed = torch.randint(0, self.num_tokens, (batch, 1), device = device)

            prefix_embed = self.forward_decoder(prefix_embed, return_embeddings = True)

        # sampling up to seq_len

        for _ in range(seq_len):

            embeds, new_cache = main_decoder(
                out,
                prepend_embeds = suffix_sos_tokens if decode_backwards else None,
                return_intermediates = True,
                return_embeddings = True,
                cache = cache,
                **kwargs
            )

            last_embeds = embeds[:, -1:]

            if not decode_backwards:
                embeds = cat((last_embeds, suffix_embed), dim = -1)
            else:
                embeds = cat((prefix_embed, last_embeds), dim = -1)

            if cache_kv and self.forward_decoder.can_cache_kv:
                cache = new_cache

            forward_logits, backward_logits = self.text_head(embeds).chunk(2, dim = -1)

            logits = forward_logits if not decode_backwards else backward_logits

            logits = logits[:, -1]

            if greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            # concat sample

            out = torch.cat((out, sample), dim = -1)

        out = out[:, orig_seq_len:]

        out, = unpack(out, batch_ps, '* n')

        return out

    def forward(
        self,
        seq,
        lens: Tensor | None = None, # Int['b']
        loss_weight_by_fb_indices: callable | None = None
    ):
        batch, seq_len, device = *seq.shape, seq.device

        # handle variable length sequences

        seq_for_labels = seq

        if exists(lens):
            mask = einx.less('j, i -> i j', arange(seq_len, device = device), lens)
            seq_for_labels = torch.where(mask, seq, -1)

        # forward autoregressive

        forward_embeds = self.forward_decoder(seq, return_embeddings = True)

        # backward autoregressive

        backward_seq = flip(seq, lens = lens)

        suffix_tokens = repeat(self.suffix_token, 'd -> b 1 d', b = batch)

        backward_embeds = self.backward_decoder(
            backward_seq,
            prepend_embeds = suffix_tokens,
            return_embeddings = True
        )

        backward_embeds = flip(backward_embeds, lens = lens)

        # trick to reduce memory on backwards pass

        forward_embeds, backward_embeds = detach_multiple(forward_embeds, backward_embeds)

        # belief state objective

        seq_arange = arange(seq_len, device = device)

        fb_pairs = cartesian_prod(seq_arange, seq_arange + 1) # plus one for suffix token

        # filter down to valid pairs, as in figure 11
        # f - forward, b - backward, i - indices

        fi, bi = fb_pairs.unbind(dim = -1)

        valid_mask = (bi - fi) >= 2

        fb_pairs = fb_pairs[valid_mask]

        # maybe subsample fb pairs

        if self.needs_subsample_fb_pairs:
            num_pairs = fb_pairs.shape[0]

            num_subsampled = max(int(num_pairs * self.train_frac_fb_pairs), 1)

            rand_subsampled_indices = torch.randperm(num_pairs, device = device)[:num_subsampled]

            fb_pairs = fb_pairs[rand_subsampled_indices]

        # get labels for both

        fi, bi = fb_pairs.unbind(dim = -1)

        labels_fi, labels_bi = (fi + 1), (bi - 1)

        forward_labels, backward_labels = seq_for_labels[:, labels_fi], seq_for_labels[:, labels_bi]

        labels = cat((forward_labels, backward_labels), dim = -1)

        # get the forward and backward embedding pairs and feed them through the text head for both forward and backward predictions

        fb_embeds = cat((
            forward_embeds[:, fi],
            backward_embeds[:, bi]
        ), dim = -1)

        logits = self.text_head(fb_embeds)

        # cross entropy loss

        loss = F.cross_entropy(
            rearrange(logits, 'b n (fb l) -> b l (fb n)', fb = 2),
            labels,
            reduction = 'none' if self.needs_loss_weight else 'mean',
            ignore_index = -1
        )

        # maybe condition on distance

        cond_on_distance = self.cond_on_distance and (random() < self.cond_on_distance_prob)

        if cond_on_distance:
            distance = (bi - fi).float()
            distance_cond = self.to_distance_cond(distance)

            fb_embeds = fb_embeds * distance_cond

        # maybe predict distance

        if exists(self.to_distance_logits) and not cond_on_distance:
            distance_logits = self.to_distance_logits(fb_embeds)

            distance_labels = (bi - fi).clamp(max = self.max_pred_distance - 1)
            distance_labels = repeat(distance_labels, 'n -> b n', b = batch)

            pred_dist_loss = F.cross_entropy(
                rearrange(distance_logits, 'b n l -> b l n'),
                distance_labels
            )

            loss = (
                loss +
                pred_dist_loss * self.pred_distance_loss_weight
            )

        # maybe loss weighting

        needs_loss_weight = default(self.needs_loss_weight, exists(loss_weight_by_fb_indices))

        if needs_loss_weight:
            loss = rearrange(loss, 'b (fb n) -> b fb n', fb = 2)

            if self.needs_loss_weight:
                loss = einx.multiply('b fb n, fb', loss, self.loss_weights)

            # allow researcher to pass in a function that acts on the the forward backward indices Int['n fb']
            # the reason this may be needed is because the earlier tokens will have more eligible pairs for training, and perhaps this could be normalized

            if exists(loss_weight_by_fb_indices):
                loss_weight = loss_weight_by_fb_indices(fb_pairs)

                if loss_weight.ndim == 1:
                    loss = einx.multiply('b fb n, n', loss, loss_weight)
                elif loss_weight.ndim == 2:
                    loss = einx.multiply('b fb n, n fb', loss, loss_weight)
                else:
                    raise ValueError('invalid loss weight dims')

            loss = loss.mean()

        return loss
