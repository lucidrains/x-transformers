from x_transformers.multi_IO.IO_wrapper import MultiIOTransformerWrapper
from x_transformers.xl_autoregressive_wrapper import *
from torch import Tensor

class MultiOXLAutoregressiveWrapper(nn.Module):
    def __init__(
            self,
            net,
            pad_value: Tensor,
            outputs: int,
            ignore_index=-100,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.outputs = outputs
        self.net = net
        if type(net) == MultiIOTransformerWrapper:
            net.autoregressive = True
            for i, token_emb in enumerate(net.token_emb):
                token_emb.padding_idx = int(pad_value[i])
                token_emb.emb.padding_idx = int(pad_value[i])
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            prompts,
            seq_len,
            eos_token=None,
            temperature=1.,
            index_eos_token:dict[int,int]=None,
            filter_logits_fn=top_k,
            filter_thres=0.9,
            mems=None,
            filter_kwargs: dict = dict(),
            **kwargs
    ):
        device, greedy, max_seq_len = prompts.device, temperature == 0, self.max_seq_len


        #prompts, ps = pack([prompts], '* n')

        b, t, _ = prompts.shape

        *all_leading_tokens, _ = prompts.split(max_seq_len, dim=1)

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
            curr_segment_len = out.shape[1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits_ = self.net(
                x,
                mems=curr_mems,
                cache=cache,
                return_mems=True,
                return_intermediates=True,
                mask=torch.zeros((b, x.shape[1])).bool().to(device),
                **kwargs
            )
            logits = logits_[0]
            cache = logits_[1]
            sample = torch.Tensor([]).to(device)
            for i in range(self.outputs):
                logits_i = logits[i][:, -1]
                if greedy:
                    sample_i = logits_i.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits_i = filter_logits_fn(logits_i, **filter_kwargs)
                    probs_i = F.softmax(filtered_logits_i / temperature, dim=-1)
                    sample_i = torch.multinomial(probs_i, 1)
                sample = torch.cat((sample, sample_i), dim=1)
            out = torch.cat((out, sample[None, :, :]), dim=1)
            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = cache

            continue_generation = True
            if exists(eos_token):
                is_eos_tokens = torch.all(torch.eq(out[:, :, :], eos_token), dim=-1)
                if torch.any(is_eos_tokens, dim=-1):
                    continue_generation = False

            if exists(index_eos_token):
                for index, eos_token in index_eos_token.items():
                    if (out[:, :, index] == eos_token).any(dim=-1):
                        continue_generation = False
            if not continue_generation:
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = torch.where(mask.unsqueeze(-1), self.pad_value, out)
        if exists(index_eos_token):
            for index, eos_token in index_eos_token.items():
                shifted_is_eos_tokens = F.pad(out[:, :, index] == eos_token, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = torch.where(mask.unsqueeze(-1), self.pad_value, out)

        out = out[:, t:]

        #out, = unpack(out, ps, '* n')

        return out

    def forward(
            self,
            x,
            mems=None,
            return_outputs=False,
            **kwargs
    ):
        self.pad_value = self.pad_value.to(x.device)
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]
        seq_len = x.shape[1]

        # prepare chunks

        split_x = x.split(max_seq_len, dim=1)
        split_labels = labels.split(max_seq_len, dim=1)
        loss_weights = tuple(map(lambda t: t.shape[-2] / seq_len, split_x))

        # go through each chunk and derive weighted losses

        total_loss = 0.
        logits_total = None
        mems_total = []
        padding_adjustment = 0
        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):
            mask = torch.all(chunk == self.pad_value, dim=2)
            if torch.all(mask, dim=1).all():
                padding_adjustment += loss_weight
                continue
                # essentially just breaking before the last chunk if the labels are all pad values
            logits_ = self.net(
                chunk,
                mems=mems,
                return_mems=True,
                return_intermediates=True,
                mask=mask,
                **kwargs
            )
            logits = logits_[0]
            if logits_total is None:
                logits_total = logits
            else:
                for i in range(len(logits_total)):
                    logits_total[i] = torch.cat((logits_total[i], logits[i]), dim=1)

            mems = logits_[1]
            mems_total.append(mems)

            loss = None
            for i in range(self.outputs):
                if torch.all(chunk_labels[:, :, i].long()==self.pad_value[i]):
                    continue
                # get first indexes before padding starts
                loss_i = F.cross_entropy(
                    rearrange(logits[i], 'b n c -> b c n'),
                    chunk_labels[:, :, i].long(),
                    ignore_index=int(self.pad_value[i])
                )
                if not loss_i.isnan():
                    if loss is None:
                        loss = loss_i
                    else:
                        loss = loss + loss_i
            if loss is None:
                padding_adjustment += loss_weight
                continue
            total_loss = total_loss + loss * loss_weight
        if 0 < padding_adjustment < 1:
            total_loss = total_loss / (1 - padding_adjustment)
        if padding_adjustment == 1:
            total_loss = torch.Tensor([0]).to(x.device)
        if not return_outputs:
            return total_loss

        return total_loss, (logits_total, mems_total)
