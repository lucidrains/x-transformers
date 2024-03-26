from math import ceil

import torch
from torch import nn, Tensor
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
            pad_value=0,
            ignore_index=-100,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        if exists(net.token_emb):
            net.token_emb.padding_idx = pad_value
            net.token_emb.emb.padding_idx = pad_value
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            prompts,
            seq_len,
            eos_token=None,
            temperature=1.,
            filter_logits_fn=top_k,
            filter_thres=0.9,
            mems=None,
            filter_kwargs: dict = dict(),
            **kwargs
    ):
        device, greedy, max_seq_len = prompts.device, temperature == 0., self.max_seq_len

        #prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        *all_leading_tokens, _ = prompts.split(max_seq_len, dim=-1)

        # catch the memory up to the current segment

        for leading_tokens in all_leading_tokens:
            _, mems = self.net(
                leading_tokens,
                mems=mems,
                return_mems=True,
                **kwargs
            )

        # now start sampling from the current segment

        curr_pos = len(all_leading_tokens) * max_seq_len
        curr_mems = mems

        cache = None
        out = prompts

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits, cache = self.net(
                x,
                mems=curr_mems,
                cache=cache,
                return_mems=True,
                return_intermediates=True,
                **kwargs
            )

            mems = cache.mems

            logits = logits[:, -1]
            if greedy:
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered_logits_i = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits_i / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems

            out = torch.cat((out, sample), dim=-1)
            if exists(eos_token):
                is_eos_tokens = (out == eos_token)
                if is_eos_tokens.any(dim=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                    out = torch.where(mask.unsqueeze(-1), self.pad_value, out)
                    break

        out = out[:, t:]

        return out

    def forward(
            self,
            x,
            mems=None,
            return_outputs=False,
            **kwargs
    ):
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]

        seq_len = x.shape[1]

        # prepare chunks

        split_x = x.split(max_seq_len, dim=-1)
        split_labels = labels.split(max_seq_len, dim=-1)
        loss_weights = tuple(map(lambda t: t.shape[-1] / seq_len, split_x))

        # go through each chunk and derive weighted losses

        total_loss = 0.
        logits_total = None
        mems_total = []
        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):
            chunk = torch.where(chunk == ignore_index, self.pad_value, chunk)
            mask = chunk != self.pad_value
            if torch.all(chunk_labels == self.pad_value):
                break
                # padding chunk_labels cause errors
            logits, mems = self.net(
                chunk,
                mems=mems,
                return_mems=True,
                mask=mask,
                **kwargs
            )

            if logits_total is None:
                logits_total = logits
            else:
                logits_total = torch.cat((logits_total, logits), dim=1)

            mems_total.append(mems)

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index=self.pad_value
            )

            total_loss = total_loss + loss * loss_weight
        if not return_outputs:
            return total_loss

        return total_loss, (logits_total, mems_total)
