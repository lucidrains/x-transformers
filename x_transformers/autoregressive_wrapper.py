from __future__ import annotations

from math import ceil, log
from typing import Tuple, Callable

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# gumbel topk

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

# function for modifying all the cached key / values

def modify_cached_kv(cache, fn):
    for inter in cache.attn_intermediates:
        if inter.layer_type == 'a':
            inter.cached_kv = [fn(t) for t in inter.cached_kv]

# for variable lengthed prefixes

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape[:2], t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = pad_at_dim(t, (max_pad_len, 0), value = pad_id, dim = 1)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None], ...]
    return aligned

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# filter logits functions dict[str -> Callable]

FILTER_LOGITS_FN = dict(
    top_p = top_p,
    top_k = top_k,
    top_a = top_a,
    min_p = min_p
)

# contrastive decoding function

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# autoregressive wrapper class

class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False,
        next_embed_loss_weight = 0.1
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

        # whether to add a continuous loss
        self.add_continuous_pred_head = net.add_continuous_pred_head
        self.next_embed_loss_weight = next_embed_loss_weight

    @torch.no_grad()
    @eval_decorator
    def beam_search(
        self,
        prompts,
        seq_len,
        beams = 4,
        return_beams_and_scores = False,
        eos_token = None,
        temperature = 1.,
        stochastic = False,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = identity,
        restrict_to_max_seq_len = True,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        **kwargs
    ):
        assert not exists(eos_token), 'eos token not supported yet'

        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        prompts, packed_shape = pack([prompts], '* n')

        batch, orig_seq_len = prompts.shape

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = orig_seq_len - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        should_cache = cache_kv and self.net.can_cache_kv

        # scores for the beams

        scores = torch.zeros((batch,), device = device)

        batch_arange = torch.arange(batch, device = device)

        # sampling up to seq_len

        for i in range(seq_len):
            is_first = i == 0

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    modify_cached_kv(cache, lambda t: t[..., -(max_seq_len - 1):, :])

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if should_cache:
                cache = new_cache

            logits = logits[:, -1]

            # to add to the scores

            log_probs = logits.log_softmax(dim = -1)

            # maybe filter by top_k, top_p (nucleus) for stochastic beam search

            if stochastic and not greedy:
                logits = filter_logits_fn(logits, **filter_kwargs)
                logits = (logits / temperature) + gumbel_noise(logits)

            # (gumbel) topk

            samples = logits.topk(beams, dim = -1).indices

            # get the scores for keeping track of beams

            next_scores = log_probs.gather(-1, samples)

            # expand beam times

            scores = repeat(scores, 'b -> b beams', beams = beams)
            scores = scores + next_scores

            out = repeat(out, 'b ... -> (b beams) ...', beams = beams)
            samples = rearrange(samples, 'b beams -> (b beams) 1')

            if should_cache and is_first:
                modify_cached_kv(cache, lambda t: repeat(t, 'b ... -> (b beams) ...', beams = beams))

            # concat sample

            out = torch.cat((out, samples), dim=-1)

            # sort by score and excise
            # excise out the beams

            scores = rearrange(scores, '(b prev_beams) next_beams -> b (prev_beams next_beams)', b = batch)
            curr_num_beams = scores.shape[-1]

            if curr_num_beams > beams:
                scores, sort_indices = scores.sort(dim = -1, descending = True)

                scores = scores[:, :beams]
                top_beams_indices = sort_indices[:, :beams]

                top_beams_indices = curr_num_beams * batch_arange[:, None] + top_beams_indices

                flattened_beam_indices = rearrange(top_beams_indices, 'b beams -> (b beams)')

                out = out[flattened_beam_indices]

            scores = rearrange(scores, 'b beams -> (b beams)')

            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            if is_eos_tokens.any(dim = -1).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # select out the top beam

        out = rearrange(out, '(b beams) seq -> b beams seq', b = batch)

        out = out[..., orig_seq_len:]

        out, = unpack(out, packed_shape, '* beams n') # prompt may have no batch dimension

        if not return_beams_and_scores:
            return out[..., 0, :]

        scores = rearrange(scores, '(b beams) -> beams b', b = batch)
        out = rearrange(out, 'b beams n -> beams b n')

        return out, scores

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts: list[Tensor] | Tensor,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Module | Tuple[Module] | None = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: dict | Tuple[dict] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        **kwargs
    ):
        max_seq_len, greedy = self.max_seq_len, temperature == 0.

        # handle prompts given as list of variable lengthed token ids

        if isinstance(prompts, list):
            assert len(prompts) > 0, 'prompts cannot be empty list'
            assert not exists(prompt_lens), '`prompt_len` will be auto derived if prompts are passed in as list of Tensors'

            prompt_lens = tensor([t.shape[0] for t in prompts], device = prompts[0].device)

            prompts = pad_sequence(prompts, batch_first = True)

        # pack maybe no batch

        prompts, ps = pack([prompts], '* n')

        b, t, device = *prompts.shape, prompts.device

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len

        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == 'a':
                            inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # concat sample

            out = torch.cat((out, sample), dim=-1)

            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            if is_eos_tokens.any(dim = -1).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        return_outputs = False,
        prepend_embeds = None,
        **kwargs
    ):
        seq, ignore_index, add_attn_z_loss, add_next_embed_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss, self.add_continuous_pred_head

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        out, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            return_next_embed_pred = add_next_embed_loss,
            prepend_embeds = prepend_embeds,
            **kwargs
        )

        # destruct differently if doing continuous pred

        if add_next_embed_loss:
            logits, (next_embed_pred, init_embeds) = out
        else:
            logits = out

        # if there are prepended embeds, excise it out

        if exists(prepend_embeds):
            prepend_len = prepend_embeds.shape[1]
            logits = logits[:, prepend_len:]

        # take all tokens but the last

        logits = logits[:, :-1]

        # loss function

        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # cross entropy loss

        loss = loss_fn(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if add_next_embed_loss:
            mask = target != ignore_index
            embed_pred = next_embed_pred[:, :-1]
            cont_targets = init_embeds[:, 1:].detach()

            cont_loss = F.l1_loss(embed_pred, cont_targets, reduction = 'none')
            cont_loss = cont_loss[mask].mean()

            loss = loss + cont_loss * self.next_embed_loss_weight

        if not return_outputs:
            return loss

        return loss, (logits, cache)
