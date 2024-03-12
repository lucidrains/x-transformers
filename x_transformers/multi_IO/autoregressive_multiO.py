from x_transformers.autoregressive_wrapper import *



class MultiOAutoregressiveWrapper(Module):
    def __init__(
            self,
            net,
            pad_value: Tensor,
            outputs: int,
            ignore_index=-100,
            mask_prob=0.,
            add_attn_z_loss=False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.outputs = outputs

        self.net = net

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

        # if doing contrastive decoding, turn off filter automatically
        """
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
        """
        # sampling up to seq_len

        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (
                        cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

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
            logits = logits_[:-1]
            new_cache = logits_[-1]
            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            # logits is a tuple of outputs
            assert len(logits) == self.outputs
            sample = torch.Tensor([])

            for i in range(self.outputs):
                logits_i = logits[i][:, -1]
                # handle contrastive decoding, Li et al.
                # https://arxiv.org/abs/2210.15097
                """
                if exists(amateur_model):
                    for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(
                            zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                        amateur_logits, next_amateur_cache = amateur(
                            x,
                            return_intermediates=True,
                            cache=amateur_cache,
                            seq_start_pos=seq_start_pos,
                            **kwargs
                        )
    
                        amateur_logits = amateur_logits[:, -1]
    
                        assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                        logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)
    
                        if cache_kv and amateur.can_cache_kv:
                            amateur_caches[i] = next_amateur_cache
                        """
                # filter by top_k, top_p (nucleus), top_a, or custom

                if greedy:
                    sample_i = logits_i.argmax(dim=-1, keepdim=True)
                else:
                    filtered_logits_i = filter_logits_fn(logits_i, **filter_kwargs)
                    probs = F.softmax(filtered_logits_i / temperature, dim=-1)
                    sample_i = torch.multinomial(probs, 1)

                sample = torch.cat((sample, sample_i), dim=-1)
                # concat sample

            out = torch.cat((out, sample), dim=-1)

            if not exists(eos_token):
                continue

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

        # out, = unpack(out, ps, '* n')

        return out

    def forward(self, x, return_outputs=False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss
        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        """ 
        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device=x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max  # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask=mask)
        """
        logits_ = self.net(
            inp,
            return_intermediates=True,
            return_attn_z_loss=add_attn_z_loss,
            **kwargs
        )
        logits = logits_[:-1]
        cache = logits_[-1]
        loss = None
        for i in range(self.outputs):
            loss_i = F.cross_entropy(
                rearrange(logits[0][i], 'b n c -> b c n'),
                target[:, :, i].long(),
                ignore_index=ignore_index
            )

            if add_attn_z_loss:
                loss_i = loss_i + cache.attn_z_loss[i]
            if loss is None:
                loss = loss_i
            else:
                loss = loss + loss_i
        if not return_outputs:
            return loss

        return loss, (logits, cache)
