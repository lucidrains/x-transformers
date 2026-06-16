from __future__ import annotations
import math
from itertools import zip_longest

import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

import einx
from einops import repeat, rearrange, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def calc_entropy_from_logits(logits):
    prob = logits.softmax(dim = -1)
    return -(prob * log(prob)).sum(dim = -1)

def get_accumulated_threshold_mask(
    entropies,
    threshold,
    ignore_entropy_below = 0.
):
    # https://www.nature.com/articles/s41467-026-73345-0

    batch, seq_len, device = *entropies.shape, entropies.device

    if ignore_entropy_below > 0.:
        entropies = entropies.masked_fill(entropies < ignore_entropy_below, 0.)

    cum_padded = F.pad(entropies.cumsum(dim = -1), (1, 0), value = 0.)

    # find next threshold boundary

    next_idx = torch.searchsorted(cum_padded, cum_padded + threshold)

    # ensure strict progression

    next_idx = torch.maximum(next_idx, torch.arange(seq_len + 1, device = device) + 1)

    # absorbing state

    next_idx = F.pad(next_idx, (0, 1), value = seq_len + 1)

    reachable = torch.zeros((batch, seq_len + 2), dtype = torch.int32, device = device)
    reachable[:, 0] = 1

    # parallel pointer doubling

    for _ in range(seq_len.bit_length()):
        reachable.scatter_add_(1, next_idx, reachable.clone()).clamp_max_(1)
        next_idx = next_idx.gather(1, next_idx)

    return reachable[:, 1:-1].bool()

# entropy based tokenizer applied in byte-latent transformer paper
# they use a simple entropy threshold for segmenting a string into variable sized tokens

# https://arxiv.org/abs/2412.09871

class EntropyBasedTokenizer(Module):
    def __init__(
        self,
        decoder: Module,
        entropy_threshold: float,
        accumulate_entropy: bool = False,
        ignore_entropy_below: float = 0.,
        max_token_size: int | None = None
    ):
        super().__init__()
        self.decoder = decoder
        self.entropy_threshold = entropy_threshold
        self.accumulate_entropy = accumulate_entropy

        # when set to be equal to the entropy threshold (with accumulating entropy) it will recover the original non-accumulating behavior
        assert ignore_entropy_below <= entropy_threshold, 'ignore_entropy_below must be lower than or equal to the entropy threshold'
        self.ignore_entropy_below = ignore_entropy_below

        self.max_token_size = max_token_size

    @torch.no_grad()
    def forward(
        self,
        seq,            # Float['b n'] | Float['n']
        lens = None,    # Int['b']
        return_segmented_seq = False,
        decoder_forward_kwargs: dict = dict()
    ):
        no_batch_dim = seq.ndim == 1
        seq, maybe_batch_ps = pack((seq,), '* n')

        self.decoder.eval()

        is_var_length = exists(lens)
        batch, seq_len, device, max_token_size = *seq.shape, seq.device, self.max_token_size

        arange = torch.arange(seq_len, device = device)

        # forward through a small trained decoder and get the entropies of the logits

        logits = self.decoder(seq, **decoder_forward_kwargs)

        entropies = calc_entropy_from_logits(logits)

        # get length mask for boundaries

        mask = tensor(True, device = device)

        if is_var_length:
            mask = einx.less('n, b -> b n', arange, lens)

        # the mask for tokens that were of a sufficient surprise level

        if self.accumulate_entropy:
            over_thres_mask = get_accumulated_threshold_mask(
                entropies,
                self.entropy_threshold,
                ignore_entropy_below = self.ignore_entropy_below
            )
        else:
            over_thres_mask = entropies >= self.entropy_threshold

        over_thres_mask &= mask

        # needed for selecting out indices at entropy threshold mask

        arange_plus_one = arange + 1
        arange_plus_one = repeat(arange_plus_one, 'n -> b n', b = batch)

        # get a tensor of Int['b num_tokens'] with the token lengths, zero padded

        boundaries = over_thres_mask.clone()

        # set the boundary of the last token

        # if `lens` not given, assume always last token
        # but if `lens` were given, then properly set the index

        if not is_var_length:
            boundaries[..., -1] = True
        else:
            scatter_indices = rearrange(lens - 1, 'b -> b 1')
            boundaries.scatter_(-1, scatter_indices, True)

        # handle max token size - technique has the flaw that repeating subsequences are grouped into one large token

        if exists(max_token_size):
            token_ids = boundaries.cumsum(dim = -1)
            token_ids = F.pad(token_ids, (1, -1), value = 0)

            max_num_tokens = boundaries.sum(dim = -1).amax().item()
            token_ids_seq = torch.arange(max_num_tokens, device = device)

            token_mask = einx.equal('j, b i -> b j i', token_ids_seq, token_ids)

            token_sub_seq_arange = token_mask.cumsum(dim = -1)

            sub_seq_boundaries = (token_sub_seq_arange % max_token_size == 0)
            sub_seq_boundaries = (sub_seq_boundaries & token_mask).any(dim = 1)

            boundaries = boundaries | sub_seq_boundaries

            if exists(mask):
                boundaries = boundaries & mask

        # number of tokens

        num_tokens = boundaries.sum(dim = -1)

        # get number of tokens as well as derived indices

        indices = arange_plus_one[boundaries].split(num_tokens.tolist())

        # get the token lengths

        token_lengths = []

        for one_indices in indices:
            padded_indices = F.pad(one_indices, (1, 0), value = 0.)
            one_token_lengths = padded_indices[1:] - padded_indices[:-1]

            token_lengths.append(one_token_lengths)

        token_lengths = pad_sequence(token_lengths, batch_first = True)

        # early return

        if not return_segmented_seq:
            token_lengths, = unpack(token_lengths, maybe_batch_ps, '* num_tokens')

            return token_lengths

        # segment the sequence based on the token lengths

        lens = default(lens, (None,))
        segmented_seq = []

        for one_seq, one_len, one_token_length in zip_longest(seq, lens, token_lengths):

            if exists(one_len):
                one_seq = one_seq[:one_len]

            one_token_length = one_token_length[one_token_length > 0]

            splitted_seq = one_seq.split(one_token_length.tolist())
            segmented_seq.append(splitted_seq)

        if no_batch_dim:
            segmented_seq = segmented_seq[0]

        return segmented_seq
