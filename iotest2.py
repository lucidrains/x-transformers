from x_transformers.x_transformers import *
from x_transformers.autoregressive_wrapper import *
from x_transformers.multi_IO.autoregressive_multiO import MultiOAutoregressiveWrapper
from x_transformers.multi_IO.IO_wrapper import MultiIOTransformerWrapper


# based on x_transformers
# three token types = values,time, instruments
class AutoregressiveWrapper(Module):
    def __init__(
            self,
            net,
            ignore_index=-100,
            pad_value=0,
            mask_prob=0.,
            add_attn_z_loss=False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big
        # improvements https://arxiv.org/abs/2210.13432
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
            eos_token=None,
            eos_first_token=None,
            temperature=1.,
            prompt_lens: Optional[Tensor] = None,
            filter_logits_fn: Callable = top_k,
            restrict_to_max_seq_len=True,
            filter_kwargs: dict = dict(),
            cache_kv=True,
            **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        # prompts, ps = pack([prompts], '* n')
        # print(prompts.shape)
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
        is_eos_tokens = None
        for _ in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[1] > max_seq_len

                assert not (
                        cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:, :]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            (logits_values, logits_times, logits_instruments), new_cache = self.net(
                x,
                return_intermediates=True,
                cache=cache,
                seq_start_pos=seq_start_pos,
                **kwargs
            )

            logits_values = logits_values[:, -1]
            logits_times = logits_times[:, -1]
            logits_instruments = logits_instruments[:, -1]
            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = torch.Tensor([logits_values.argmax(dim=-1, keepdim=True),
                                       logits_times.argmax(dim=-1, keepdim=True),
                                       logits_instruments.argmax(dim=-1, keepdim=True)])
            else:
                filtered_logits_values = filter_logits_fn(logits_values, **filter_kwargs)
                filtered_logits_times = filter_logits_fn(logits_times, **filter_kwargs)
                filtered_logits_instruments = filter_logits_fn(logits_instruments, **filter_kwargs)
                probs_values = F.softmax(filtered_logits_values / temperature, dim=-1)
                probs_times = F.softmax(filtered_logits_times / temperature, dim=-1)
                probs_instruments = F.softmax(filtered_logits_instruments / temperature, dim=-1)
                sample = torch.Tensor([[[torch.multinomial(probs_values, 1),
                                         torch.multinomial(probs_times, 1),
                                         torch.multinomial(probs_instruments, 1)]]])

            # concat sample

            out = torch.cat((out, sample), dim=1)

            if not exists(eos_token) and not exists(eos_first_token):
                continue

            if exists(eos_token):
                is_eos_tokens = torch.all(torch.eq(out[:, :, :], eos_token), dim=-1)
                if torch.any(is_eos_tokens, dim=-1):
                    break

            if exists(eos_first_token):
                is_eos_tokens = (out[:, :, 0] == eos_first_token)
                if torch.any(is_eos_tokens, dim=-1):
                    break
        if exists(eos_token) or exists(eos_first_token) or (is_eos_tokens is not None):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
            out = torch.where(mask.unsqueeze(-1), Tensor([1, 0, 0]), out)
        out = out[:, t:]

        # out, = unpack(out, ps, '* n')

        return out

    def forward(self, x, return_outputs=False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        target_values = target[:, :, 0]
        target_times = target[:, :, 1]
        target_instruments = target[:, :, 2]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device=x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max  # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim=-1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask=mask)

        logits = self.net(
            inp,
            #return_intermediates=True,
            #return_attn_z_loss=add_attn_z_loss,
            **kwargs
        )
        logits_values, logits_times, logits_instruments = logits
        loss_values = F.cross_entropy(
            rearrange(logits_values, 'b n c -> b c n'),
            target_values,
            ignore_index=ignore_index
        )
        loss_times = F.cross_entropy(
            rearrange(logits_times, 'b n c -> b c n'),
            target_times,
            ignore_index=ignore_index
        )
        loss_instruments = F.cross_entropy(
            rearrange(logits_instruments, 'b n c -> b c n'),
            target_instruments,
            ignore_index=ignore_index
        )
        loss = loss_values + loss_times + loss_instruments
        if add_attn_z_loss:
            loss = loss# + cache.attn_z_loss

        if not return_outputs:
            return loss

        return loss, (logits_values, logits_times, logits_instruments)#, cache)


# based on x_transformers
# three token types = values,time, instruments
class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens_values,
            num_tokens_times,
            num_tokens_instruments,
            max_seq_len,
            pre_attn_layers: list[AttentionLayers],
            attn_layers: AttentionLayers,
            emb_dropout=0.,
            post_emb_norm=False,
            use_abs_pos_emb=True,
            scaled_sinu_pos_emb=False,
            l2norm_embed=False,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = dim
        self.emb_dim = emb_dim
        self.num_tokens_values = num_tokens_values
        self.num_tokens_times = num_tokens_times
        self.num_tokens_instruments = num_tokens_instruments

        self.pre_attn_layers = pre_attn_layers

        self.values_emb_dim = pre_attn_layers[0].dim
        self.times_emb_dim = pre_attn_layers[1].dim
        self.instruments_emb_dim = pre_attn_layers[2].dim

        self.max_seq_len = max_seq_len

        self.l2norm_embed = l2norm_embed

        self.token_emb_values = TokenEmbedding(self.values_emb_dim, num_tokens_values, l2norm_embed=l2norm_embed)
        self.token_emb_times = TokenEmbedding(self.times_emb_dim, num_tokens_times, l2norm_embed=l2norm_embed)
        self.token_emb_instruments = TokenEmbedding(self.instruments_emb_dim, num_tokens_instruments,
                                                    l2norm_embed=l2norm_embed)

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.has_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        # ill make it generalized later
        # elif scaled_sinu_pos_emb:
        #    self.pos_emb = ScaledSinusoidalEmbedding(pre_attn_token_emb_dim)
        # else:
        #    self.pos_emb = AbsolutePositionalEmbedding(pre_attn_token_emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        self.post_emb_norm_values = nn.LayerNorm(self.values_emb_dim) if post_emb_norm else nn.Identity()
        self.post_emb_norm_times = nn.LayerNorm(self.times_emb_dim) if post_emb_norm else nn.Identity()
        self.post_emb_norm_instruments = nn.LayerNorm(self.instruments_emb_dim) if post_emb_norm else nn.Identity()

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.init_()

        self.to_logits_values = nn.Linear(dim, num_tokens_values)
        self.to_logits_times = nn.Linear(dim, num_tokens_times)
        self.to_logits_instruments = nn.Linear(dim, num_tokens_instruments)

        # whether can do cached kv decoding

        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb_times.emb.weight, std=1e-5)
            nn.init.normal_(self.token_emb_values.emb.weight, std=1e-5)
            nn.init.normal_(self.token_emb_instruments.emb.weight, std=1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb_times.emb.weight, std=1e-5)
                nn.init.normal_(self.pos_emb_values.emb.weight, std=1e-5)
                nn.init.normal_(self.pos_emb_instruments.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb_times.emb.weight)
        nn.init.kaiming_normal_(self.token_emb_values.emb.weight)
        nn.init.kaiming_normal_(self.token_emb_instruments.emb.weight)

    def forward(
            self,
            x,
            return_embeddings=False,
            return_logits_and_embeddings=False,
            return_intermediates=False,
            mask=None,
            return_attn=False,
            mems=None,
            return_attn_z_loss=False,
            attn_z_loss_weight=1e-4,
            seq_start_pos=None,
            cache: Optional[LayerIntermediates] = None,
            **kwargs
    ):

        # absolute positional embedding

        # external_pos_emb = exists(pos) and pos.dtype != torch.long
        # pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x_values = x[:, :, 0]
        x_times = x[:, :, 1]
        x_instruments = x[:, :, 2]
        x_values = self.token_emb_values(x_values)
        x_times = self.token_emb_times(x_times)
        x_instruments = self.token_emb_instruments(x_instruments)

        # post embedding norm, purportedly leads to greater stabilization

        x_values = self.post_emb_norm_values(x_values)
        x_times = self.post_emb_norm_times(x_times)
        x_instruments = self.post_emb_norm_instruments(x_instruments)

        # embedding dropout

        x_values = self.emb_dropout(x_values)
        x_times = self.emb_dropout(x_times)
        x_instruments = self.emb_dropout(x_instruments)

        # pre attention layers
        x_values = self.pre_attn_layers[0](x_values, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                           seq_start_pos=seq_start_pos, **kwargs)
        x_times = self.pre_attn_layers[1](x_times, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                          seq_start_pos=seq_start_pos, **kwargs)
        x_instruments = self.pre_attn_layers[2](x_instruments, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                                seq_start_pos=seq_start_pos, **kwargs)

        x = torch.cat([x_values, x_times, x_instruments], dim=-1)
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, cache=cache, return_hiddens=True,
                                            seq_start_pos=seq_start_pos, **kwargs)

        if return_logits_and_embeddings:
            out = (self.to_logits_values(x),
                   self.to_logits_times(x),
                   self.to_logits_instruments(x), x)
        elif return_embeddings:
            out = x
        else:
            out = (self.to_logits_values(x),
                   self.to_logits_times(x),
                   self.to_logits_instruments(x))

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out


model = MultiIOTransformerWrapper(
    num_tokens=[4, 4, 4],
    autoregressive=True,
    max_seq_len=4,
    use_abs_pos_emb=False,
    input_attn_layers=[AttentionLayers(dim=4, depth=1, heads=1, causal=True),
                       AttentionLayers(dim=4, depth=1, heads=1, causal=True),
                       AttentionLayers(dim=4, depth=1, heads=1, causal=True), ],
    attn_layers=AttentionLayers(
        dim=12,
        depth=2,
        heads=4,
        # rotary_pos_emb=True,
        attn_flash=True,
        # use_scalenorm=True,
        # ff_glu=True,
        causal=True
    )
)

model = MultiOAutoregressiveWrapper(model, outputs=3, pad_value=torch.Tensor([0, 0, 0]))

print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# x = torch.Tensor(torch.randint(1, 3, (1, 10, 2))).float()
# print(x)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.Tensor([[[0, 1, 2], [0, 1, 2]]]).long()
for i in range(10000):
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print(model(x, return_outputs=True)[1])
