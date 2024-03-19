from x_transformers.multi_IO import MultiIOTransformerWrapper
from x_transformers.xl_autoregressive_wrapper import *


class MultiOXLAutoregressiveWrapper(nn.Module):
    def __init__(
            self,
            net,
            outputs: int,
            ignore_index=-100,
            pad_value=0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.outputs = outputs
        self.net = net
        if type(net) == MultiIOTransformerWrapper:
            net.autoregressive = True
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            start_tokens,
            seq_len,
            eos_token=None,
            temperature=1.,
            filter_logits_fn=top_k,
            filter_thres=0.9,
            mems=None,
            filter_kwargs: dict = dict(),
            **kwargs
    ):
        device, greedy, max_seq_len = start_tokens.device, temperature==0, self.max_seq_len

        start_tokens, ps = pack([start_tokens], '* n')

        b, t = start_tokens.shape

        *all_leading_tokens, _ = start_tokens.split(max_seq_len, dim=-1)

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
        out = start_tokens

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits_ = self.net(
                x,
                mems=curr_mems,
                cache=cache,
                return_mems=True,
                return_intermediates=True,
                **kwargs
            )
            logits = logits_[0]
            cache = logits_[1]

            sample = torch.Tensor([])
            for i in range(self.outputs):
                logits_i = logits[i][:, -1]
                if greedy:
                    sample_i = logits_i.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits_i = filter_logits_fn(logits_i, **filter_kwargs)
                    probs_i = F.softmax(filtered_logits_i / temperature, dim=-1)
                    sample_i = torch.multinomial(probs_i, 1)

                sample = torch.cat((sample, sample_i), dim=-1)
                out = torch.cat((out, sample), dim=-1)
                if is_last_segment_tokens:
                    curr_pos = curr_segment_len
                    curr_mems = cache

                if exists(eos_token):
                    is_eos_tokens = torch.all(torch.eq(out[:, :, :], eos_token), dim=-1)
                    if torch.any(is_eos_tokens, dim=-1):
                        break

            if exists(eos_token):
                # mask out everything after the eos tokens
                shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = torch.where(mask.unsqueeze(-1), self.pad_value, out)

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
            self,
            x,
            mems=None,
            **kwargs
    ):
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]
        seq_len = x.shape[1]

        # prepare chunks

        split_x = x.split(max_seq_len, dim=1)
        split_labels = labels.split(max_seq_len, dim=1)
        loss_weights = tuple(map(lambda t: t.shape[-1] / seq_len, split_x))

        # go through each chunk and derive weighted losses

        total_loss = 0.

        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):
            #print(chunk_labels)
            logits_ = self.net(
                chunk,
                mems=mems,
                return_mems=True,
                return_intermediates=True,
                **kwargs
            )

            logits = logits_[0]
            mems = logits_[1]
            loss = None
            #print(logits.shape)
            for i in range(self.outputs):
                #print(logits[i])
                loss_i = F.cross_entropy(
                    rearrange(logits[i], 'b n c -> b c n'),
                    chunk_labels[:, :, i].long(),
                    ignore_index=ignore_index
                )
                if loss is None:
                    loss = loss_i
                else:
                    loss = loss + loss_i

            total_loss = total_loss + loss * loss_weight
        return total_loss
