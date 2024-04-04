from x_transformers.autoregressive_wrapper import *
from x_transformers.multi_IO.IO_wrapper import MultiIOTransformerWrapper


class MultiOAutoregressiveWrapper(Module):
    def __init__(
            self,
            net,
            pad_value: Tensor,
            outputs: int = None,
            ignore_index=-100,
            mask_prob=0.,
            add_attn_z_loss=False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.outputs = outputs
        self.max_seq_len = net.max_seq_len
        self.net = net
        if type(net) == MultiIOTransformerWrapper:
            self.outputs = len(net.logits_dim)
            net.autoregressive = True
            for i, token_emb in enumerate(net.token_emb):
                token_emb.padding_idx = int(pad_value[i])
                token_emb.emb.padding_idx = int(pad_value[i])
        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            prompts,
            seq_len,
            eos_token: Tensor = None,
            index_eos_token: dict[int, int] = None,
            temperature=1.,
            prompt_lens: Optional[Tensor] = None,
            filter_logits_fn: Callable = top_k,
            restrict_to_max_seq_len=True,
            # amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
            filter_kwargs: dict = dict(),
            # contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            #    beta=0.5,
            #    alpha=0.1
            # ),
            cache_kv=True,
            **kwargs
    ):
        # assumes it is multi-output
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        # prompts, ps = pack([prompts], '* n')

        b, t, _ = prompts.shape

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id=self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        # sampling up to seq_len

        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (
                        cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeddings. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits_ = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs
            )
            logits = logits_[0]
            new_cache = logits_[1:]
            # if cache_kv and self.net.can_cache_kv:
            #    cache = new_cache

            # logits is a tuple of outputs
            # assert len(logits) == self.outputs
            sample = torch.Tensor([])
            for i in range(self.outputs):

                # print(logits[0])
                logits_i = logits[i][:, -1]
                # handle contrastive decoding, Li et al.
                # https://arxiv.org/abs/2210.15097
                # filter by top_k, top_p (nucleus), top_a, or custom

                if greedy:
                    sample_i = logits_i.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits_i = filter_logits_fn(logits_i, **filter_kwargs)
                    probs = F.softmax(filtered_logits_i / temperature, dim=-1)
                    sample_i = torch.multinomial(probs, 1)

                sample = torch.cat((sample, sample_i), dim=-1)
                # concat sample
            out = torch.cat((out, sample[None, :, :]), dim=1)

            if not exists(eos_token) and not exists(index_eos_token):
                continue

            continue_generation = True
            if exists(index_eos_token):
                for index, eos_token in index_eos_token.items():
                    if (out[:, :, index] == eos_token).any(dim=-1):
                        continue_generation = False
            if exists(eos_token):
                is_eos_tokens = torch.all(torch.eq(out[:, :, :], eos_token), dim=-1)
                if torch.any(is_eos_tokens, dim=-1):
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

        # out, = unpack(out, ps, '* n')

        return out

    def forward(self, x, return_outputs=False, **kwargs):
        self.pad_value = self.pad_value.to(x.device)
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss
        inp, target = x[:, :-1], x[:, 1:]
        # print(inp.shape,target.shape)
        # inp = torch.where(inp == ignore_index, self.pad_value, inp)
        # inp is 2d tensor, pad value is 1d
        mask = torch.all(inp == self.pad_value, dim=2)
        logits_ = self.net(
            inp,
            return_intermediates=True,
            return_attn_z_loss=add_attn_z_loss,
            mask=mask,
            **kwargs
        )
        logits = logits_[0]
        # print(logits)
        cache = logits_[1]
        # print(len(cache))
        loss = None
        for i in range(self.outputs):
            # print(logits[i].shape)
            # print(target[:,:,i].shape)
            loss_i = F.cross_entropy(
                rearrange(logits[i], 'b n c -> b c n'),
                target[:, :, i].long(),
                ignore_index=int(self.pad_value[i])
            )

            if add_attn_z_loss:
                if self.net.post_attn_layers is not None:
                    loss_i = loss_i + cache[len(cache) - 1][i].attn_z_loss
            if loss is None:
                loss = loss_i
            else:
                loss = loss + loss_i
        if add_attn_z_loss and self.net.post_attn_layers is None:
            loss = loss + cache[len(cache) - 1].attn_z_loss
        if not return_outputs:
            return loss

        return loss, (logits, cache)
