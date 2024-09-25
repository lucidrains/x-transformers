from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator

# helper functions

def exists(val):
    return val is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0 

# xl autoregressive wrapper class

class XLAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_kwargs: dict = dict(),
        mems = None,
        **kwargs
    ):
        device, max_seq_len = start_tokens.device, self.max_seq_len

        start_tokens, ps = pack([start_tokens], '* n')

        b, t = start_tokens.shape

        *all_leading_tokens, _ = start_tokens.split(max_seq_len, dim = -1)

        # catch the memory up to the current segment

        for leading_tokens in all_leading_tokens:
            _, mems = self.net(
                leading_tokens,
                mems = mems,
                return_mems = True,
                **kwargs
            )

        # now start sampling from the current segment

        curr_pos = len(all_leading_tokens) * max_seq_len
        curr_mems = mems

        cache = None
        out = start_tokens

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits, cache = self.net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            mems = cache.mems

            logits = logits[:, -1]
            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        **kwargs
    ):
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]

        seq_len = x.shape[1]

        # prepare chunks

        split_x = x.split(max_seq_len, dim = -1)
        split_labels = labels.split(max_seq_len, dim = -1)
        loss_weights = tuple((t.shape[-1] / seq_len) for t in split_x)

        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # go through each chunk and derive weighted losses

        total_loss = 0.        

        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):

            logits, mems = self.net(
                chunk,
                mems = mems,
                return_mems = True,
                **kwargs
            )

            loss = loss_fn(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = ignore_index
            )

            total_loss = total_loss + loss * loss_weight

        return total_loss
