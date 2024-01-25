from x_transformers.x_transformers import *

import math
from typing import Dict
from typing import Optional

import torch
from einops import rearrange, repeat, pack, unpack
from torch import nn, Tensor

from x_transformers import *
from x_transformers.x_transformers import AttentionLayers


class MultiIOTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens: list[int] or int,
            max_seq_len,
            pre_attn_layers: list[AttentionLayers] = None,
            concat_emb_dim: bool = True,
            attn_layers: AttentionLayers,
            embed_num_tokens: Dict[str, int] = dict(),
            emb_dim: list[int] or int = None,
            max_mem_len=0,
            shift_mem_down=0,
            emb_dropout=0.,
            post_emb_norm=False,
            post_attn_layers: list[AttentionLayers] = None,
            num_memory_tokens=None,
            memory_tokens_interspersed_every=None,
            tie_embedding=False,
            logits_dim: list[int] or int = None,
            use_abs_pos_emb=True,
            scaled_sinu_pos_emb=False,
            l2norm_embed=False,
            emb_frac_gradient=1.,  # GLM-130B and Cogview successfully used this, set at 0.1
            attn_z_loss_weight=1e-4,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        if type(emb_dim) == list and type(num_tokens) == list:
            assert len(emb_dim) == len(num_tokens), 'number of embeddings must match number of inputs'

        self.multi_input = ((pre_attn_layers is not None) or (type(num_tokens) == list) or (type(emb_dim) == list))
        self.multi_output = (post_attn_layers is not None) or (type(logits_dim) == list)
        if not self.multi_input:
            self.model = TransformerWrapper(
                num_tokens=num_tokens,
                logits_dim=logits_dim if not self.multi_output else dim,
                max_seq_len=max_seq_len,
                attn_layers=attn_layers,
                embed_num_tokens=embed_num_tokens,
                emb_dim=emb_dim,
                max_mem_len=max_mem_len,
                shift_mem_down=shift_mem_down,
                emb_dropout=emb_dropout,
                post_emb_norm=post_emb_norm,
                num_memory_tokens=num_memory_tokens,
                memory_tokens_interspersed_every=memory_tokens_interspersed_every,
                tie_embedding=tie_embedding,
                use_abs_pos_emb=use_abs_pos_emb,
                scaled_sinu_pos_emb=scaled_sinu_pos_emb,
                l2norm_embed=l2norm_embed,
                emb_frac_gradient=emb_frac_gradient,
                attn_z_loss_weight=attn_z_loss_weight,
            )
        else:
            self.emb_dim = emb_dim if (pre_attn_layers is None) else [layer.dim for layer in pre_attn_layers]
            self.num_tokens = num_tokens

            self.max_seq_len = max_seq_len

            self.max_mem_len = max_mem_len
            self.shift_mem_down = shift_mem_down

            self.pre_attn_layers = pre_attn_layers
            if pre_attn_layers is not None:
                assert type(num_tokens) == list, 'num_tokens must be a list of number of tokens for each input'
                assert len(pre_attn_layers) == len(num_tokens), 'number of pre_attn_layers must match number of inputs'
                if concat_emb_dim:
                    self.pre_attn_layers_map = nn.Linear(sum(self.emb_dim), dim) if sum(self.emb_dim) != dim else \
                        nn.Identity()
                    if sum(self.emb_dim) != dim:
                        print('Note: Since the embedding dimensions of the pre_attention layers are concatenated, '
                              'the dimensions are added. As your model dimension is not equal to the sum of the '
                              'embedding dimensions, a linear layer is added to project the concatenated embedding. '
                              'If this is not desired, please change the model dimensions.')
                else:
                    # assert that all embedding dimensions are equal to the model dimension
                    assert all(dim == d for d in self.emb_dim), 'all embedding dimensions must be equal to the model ' \
                                                                'dimension since having concat_emb_dim means that ' \
                                                                'the model dimension is added to the embedding '

            self.concat_emb_dim = concat_emb_dim if pre_attn_layers else False

            self.l2norm_embed = l2norm_embed

            self.token_emb = [TokenEmbedding(self.emb_dim[i], num_tokens[i], l2norm_embed=l2norm_embed) for i in
                              range(len(num_tokens))]

            no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

            if no_abs_pos_emb:
                self.pos_emb = always(0)
            elif scaled_sinu_pos_emb:
                self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
            else:
                self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

            # additional embeddings - say type embedding from BERT

            self.embeds = None

            if len(embed_num_tokens) > 0:
                if self.pre_attn_layers is not None:
                    self.embeds = [nn.ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, self.emb_dim[i])
                                                  for name, num_tokens in embed_num_tokens.items()}) for i in
                                   range(len(self.emb_dim))]
                else:
                    self.embeds = nn.ModuleDict(
                        {f'{name}_embed': nn.Embedding(num_tokens, self.emb_dim) for name, num_tokens in
                         embed_num_tokens.items()})

            # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

            self.emb_frac_gradient = emb_frac_gradient
            if self.pre_attn_layers is not None:
                self.post_emb_norm = [LayerNorm(self.emb_dim[i]) if post_emb_norm else nn.Identity() for i in
                                      range(len(self.emb_dim))]
                self.emb_dropout = [nn.Dropout(emb_dropout) for _ in range(len(self.emb_dim))]
                self.project_emb = [
                    nn.Linear(self.emb_dim[i], dim) if self.emb_dim[i] != self.pre_attn_layers[i].dim else nn.Identity()
                    for i in
                    range(len(self.emb_dim))]
            else:
                self.post_emb_norm = LayerNorm(self.emb_dim) if post_emb_norm else nn.Identity()
                self.emb_dropout = nn.Dropout(emb_dropout)
                self.project_emb = nn.Linear(self.emb_dim, dim) if self.emb_dim != dim else nn.Identity()

            self.attn_layers = attn_layers

            self.init_()

            # memory tokens (like [cls]) from Memory Transformers paper

            self.num_memory_tokens = num_memory_tokens
            if type(self.num_memory_tokens) == list:
                assert len(num_memory_tokens) == len(
                    self.emb_dim), 'number of memory tokens must match number of inputs'
                self.memory_tokens = nn.ParameterList(
                    [nn.Parameter(torch.randn(num_memory_tokens[i], self.emb_dim[i])) for i in
                     range(len(self.emb_dim))])

            self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

            # whether can do cached kv decoding

            self.can_cache_kv = self.num_memory_tokens == 0
            self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

        if self.multi_output:
            self.post_attn_layers = post_attn_layers
            if post_attn_layers is not None:
                if logits_dim is not None:
                    assert len(post_attn_layers) == len(
                        logits_dim), 'number of post_attn_layers must match number of outputs'

            if self.post_attn_layers is not None:
                self.post_mapping = [nn.Linear(dim, self.post_attn_layers[i].dim)
                                     if dim != self.post_attn_layers[i].dim else nn.Identity() for i in
                                     range(len(self.post_attn_layers))]
                if any(dim != self.post_attn_layers[i].dim for i in range(len(self.post_attn_layers))):
                    print('Note: Since the model dimension is not equal to the post_attn_layers dimension, '
                          'a linear layer is added to project the model dimension to the post_attn_layers dimension. '
                          'If this is not desired, please change the model dimensions.')
                if logits_dim is not None:
                    if tie_embedding:
                        assert all(self.post_attn_layers[i].dim == self.post_attn_layers[i].dim for i in range(
                            len(self.post_attn_layers))), 'if tie_embedding is True, the dimensions of the input and output attn layers must be equal'
                    self.to_logits = [nn.Linear(dim, d, bias=False) for d in logits_dim] if not tie_embedding else \
                        lambda t: [t @ self.token_emb[i].emb.weight.t() for i in range(len(logits_dim))]
                else:
                    self.to_logits = [nn.Identity() for _ in range(len(self.post_attn_layers))]
            else:
                if logits_dim is not None:
                    if tie_embedding:
                        self.logits = [lambda t: t @ self.token_emb[i].emb.weight.t() if self.multi_input else lambda t: t @ self.model.token_emb.emb.weight.t() for i in range(len(logits_dim)) ]
                    else:
                        self.to_logits = [nn.Linear(dim, d, bias=False) for d in logits_dim]
                else:
                    self.to_logits = nn.Identity()

    def init_(self):
        if self.l2norm_embed:
            for i in range(len(self.token_emb)):
                nn.init.normal_(self.token_emb[i].emb.weight, std=1e-5)
                if not isinstance(self.pos_emb, always):
                    nn.init.normal_(self.pos_emb[i].emb.weight, std=1e-5)
        for i in range(len(self.token_emb)):
            nn.init.kaiming_normal_(self.token_emb[i].emb.weight)

    def forward(
            self,
            x,
            return_embeddings=False,
            return_logits_and_embeddings=False,
            return_intermediates=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            mem_masks=None,
            pos=None,
            prepend_embeds=None,
            prepend_mask=None,
            embed_ids: list[Dict[str, Tensor]] or Dict[str, Tensor] = dict(),
            sum_embeds=None,
            return_attn_z_loss=False,
            attn_z_loss_weight=1e-4,
            seq_start_pos=None,
            cache=None,
            **kwargs
    ):
        global intermediates_model, cache_pre_attn_layers, cache_model, cache_post_attn_layers, intermediates_pre_attn_layers, intermediates_pre_attn_layer, mem_packed_shape, mem_every

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        if not self.multi_input and not self.multi_output:
            return self.model(x, return_embeddings, return_logits_and_embeddings, return_intermediates, mask,
                              return_mems, return_attn, mems, mem_masks, pos, prepend_embeds, prepend_mask, embed_ids,
                              sum_embeds, return_attn_z_loss, attn_z_loss_weight, seq_start_pos, cache)
        if cache is not None:
            if self.pre_attn_layers is not None and self.post_attn_layers is not None:
                cache_pre_attn_layers, cache_model, cache_post_attn_layers = cache
            elif self.pre_attn_layers is not None:
                cache_pre_attn_layers, cache_model = cache
                cache_post_attn_layers = None
            elif self.post_attn_layers is not None:
                cache_model, cache_post_attn_layers = cache
                cache_pre_attn_layers = None
            else:
                cache_model = cache
                cache_pre_attn_layers = None
                cache_post_attn_layers = None
        else:
            cache_pre_attn_layers = None
            cache_model = None
            cache_post_attn_layers = None

        if self.multi_input:
            b, n, device, emb_frac_gradient = x.shape[0], x.shape[1], x.device, self.emb_frac_gradient
            if self.num_memory_tokens is not None:
                num_mems, has_memory_tokens = self.num_memory_tokens, True
            else:
                num_mems, has_memory_tokens = 0, False
            assert x.shape[-1] == len(self.pre_attn_layers), 'number of inputs must match number of ' \
                                                             'pre_attn_layers'
            assert x.shape[-1] == len(self.num_tokens), 'number of inputs must match number of num_tokens'
            assert x.shape[-1] == len(self.emb_dim), 'number of inputs must match number of emb_dim'
            external_pos_emb = exists(pos) and pos.dtype != torch.long
            pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
            intermediates_pre_attn_layers = []
            outx = []
            for i in range(x.shape[-1]):
                x_i = x[:, :, i]
                x_i = self.token_emb[i](x_i) + pos_emb
                if exists(self.embeds):
                    assert len(embed_ids[i]) == len(self.embeds)

                    for name, embed_id in embed_ids[i].items():
                        embed_key = f'{name}_embed'

                        assert embed_key in self.embeds
                        embed = self.embeds[embed_key](embed_id)

                        x_i = x_i + embed
                x_i = self.post_emb_norm[i](x_i)

                if exists(prepend_embeds):
                    prepend_seq, prepend_dim = prepend_embeds[i].shape[1:]
                    assert prepend_dim == x_i.shape[
                        -1], 'prepended embeddings need to have same dimensions as text model dimensions'
                    x_i = torch.cat((prepend_embeds[i], x_i), dim=-2)
                    if exists(prepend_mask) or exists(mask):
                        mask = default(mask, lambda: torch.ones((b, n), device=device, dtype=torch.bool))
                        prepend_mask = default(prepend_mask[i],
                                               lambda: torch.ones((b, prepend_seq), device=device, dtype=torch.bool))
                        mask = torch.cat((prepend_mask, mask), dim=-1)

                if emb_frac_gradient < 1:
                    assert emb_frac_gradient > 0
                    x_i = x_i * emb_frac_gradient + x_i.detach() * (1 - emb_frac_gradient)
                x_i = self.emb_dropout[i](x_i)
                x_i = self.project_emb[i](x_i)

                if has_memory_tokens:
                    mem_every = self.memory_tokens_interspersed_every[i]

                    if exists(mem_every):
                        assert mem_every > 0
                        assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                        next_seq_len = math.ceil(n / mem_every) * mem_every

                        x_i = pad_at_dim(x_i, (0, next_seq_len - n), dim=-2, value=0.)
                        x_i = rearrange(x_i, 'b (n m) d -> (b n) m d', m=mem_every)

                    mem = repeat(self.memory_tokens[i], 'n d -> b n d', b=x_i.shape[0])
                    x_i, mem_packed_shape = pack((mem, x_i), 'b * d')

                    # auto-handle masking after appending memory tokens
                    if not exists(mem_every) and exists(mask):
                        mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)

                    if exists(mem_every):
                        x_i = rearrange(x_i, '(b n) m d -> b (n m) d', b=b)

                if self.shift_mem_down and exists(mems):
                    mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
                    mems = [*mems_r, *mems_l]

                if self.pre_attn_layers is not None:
                    x_i, intermediates_pre_attn_layer = self.pre_attn_layers[i](x_i, mask=mask, mems=mems,
                                                                                cache=cache_pre_attn_layers[
                                                                                    i] if cache_pre_attn_layers is not None else None,
                                                                                return_hiddens=True,
                                                                                seq_start_pos=seq_start_pos,
                                                                                **kwargs)
                    intermediates_pre_attn_layers.append(intermediates_pre_attn_layer)
                if has_memory_tokens:
                    if exists(mem_every):
                        x_i = rearrange(x_i, 'b (n m) d -> (b n) m d', m=(mem_every + self.num_memory_tokens[i]))

                    mem, x_i = unpack(x_i, mem_packed_shape, 'b * d')

                    intermediates_pre_attn_layer.memory_tokens = mem

                    if exists(mem_every):
                        x_i = rearrange(x_i, '(b n) m d -> b (n m) d', b=b)

                    x_i = x_i[:, :n]  # probably cause of issue currently
                if i == 0:
                    outx = x_i
                else:
                    if self.concat_emb_dim:
                        outx = torch.cat((x, x_i), dim=-1)
                    else:
                        outx = x + x_i
            x = outx
            if self.concat_emb_dim:
                x = self.pre_attn_layers_map(x)
        else:
            if return_hiddens:
                x, intermediates_model = self.model(x, return_embeddings, return_logits_and_embeddings,
                                                    return_intermediates, mask,
                                                    return_mems, return_attn, mems, mem_masks, pos, prepend_embeds,
                                                    prepend_mask, embed_ids,
                                                    sum_embeds, return_attn_z_loss, attn_z_loss_weight, seq_start_pos,
                                                    cache)
            else:
                x = self.model(x, False, False, False, mask,
                               False, False, mems, mem_masks, pos, prepend_embeds, prepend_mask, embed_ids,
                               sum_embeds, False, attn_z_loss_weight, seq_start_pos, cache)

        if self.multi_output:
            if self.post_attn_layers is not None:
                outputs = []
                intermediates_post = []
                x_values = []
                for i, layer in enumerate(self.post_attn_layers):
                    post_x = self.post_mapping[i](x)
                    if return_hiddens:
                        post_x, inter = layer(post_x, mask=mask, mems=mems, mem_masks=mem_masks,
                                              cache=cache_post_attn_layers[i],
                                              return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs)
                        intermediates_post.append(inter)
                        x_values.append(post_x)
                    else:
                        x_values.append(
                            layer(post_x, mask=mask, mems=mems, mem_masks=mem_masks,
                                  cache=cache_post_attn_layers[i] if cache_post_attn_layers is not None else None,
                                  return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs))
                    outputs.append(self.to_logits[i](x_values[i]))
                if return_logits_and_embeddings:
                    out = (outputs, x_values)
                elif return_embeddings:
                    out = x_values
                else:
                    out = outputs

                if return_attn_z_loss:
                    pre_softmax_attns = list(list(map(lambda t: t.pre_softmax_attn, intermediate.attn_intermediates))
                                             for intermediate in intermediates_post)
                    for i in range(len(intermediates_post)):
                        intermediates_post[i].attn_z_loss = calc_z_loss(pre_softmax_attns[i], weight=attn_z_loss_weight)
                    return_intermediates = True

                if return_mems:
                    for i in range(len(intermediates_post)):
                        hiddens = intermediates_post[i].hiddens
                        new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(
                            mems) else hiddens
                        new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

                        if not return_intermediates:
                            return out, new_mems

                        intermediates_post[i].mems = new_mems
                        intermediates_post[i].mems = hiddens

                if return_intermediates:
                    if self.pre_attn_layers is not None:
                        return out, (intermediates_pre_attn_layers, intermediates_model, intermediates_post)
                    else:
                        return out, (intermediates_model, intermediates_post)

                if return_attn:
                    attn_maps = list(list(map(lambda t: t.post_softmax_attn, intermediate.attn_intermediates))
                                     for intermediate in intermediates_post)
                    return out, (intermediates_model, attn_maps)

                return out
            else:
                x_values = []
                for i in self.to_logits:
                    x_values.append(i(x))
                if return_logits_and_embeddings:
                    out = (x_values, x)
                elif return_embeddings:
                    out = x
                else:
                    out = x_values

                if return_intermediates:
                    if self.pre_attn_layers is not None:
                        return out, (intermediates_pre_attn_layers, intermediates_model)
                    else:
                        return out, intermediates_model
                return out
        else:
            if self.num_memory_tokens is not None:
                n = x.shape[1]
                mem_every = self.memory_tokens_interspersed_every
                if exists(mem_every):
                    x = rearrange(x, 'b (n m) d -> (b n) m d', m=(mem_every + self.num_memory_tokens))

                mem, x = unpack(x, mem_packed_shape, 'b * d')

                intermediates_model.memory_tokens = mem

                if exists(mem_every):
                    x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

                x = x[:, :n]

            if return_logits_and_embeddings:
                if type(self.to_logits) == list:
                    out = (list(self.to_logits[i](x) for i in range(len(self.to_logits))), x)
                else:
                    out = (self.to_logits(x), x)
            elif return_embeddings:
                out = x
            else:
                out = list(self.to_logits[i](x) for i in range(len(self.to_logits)))

            if return_attn_z_loss:
                pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates_model.attn_intermediates))
                intermediates_model.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
                return_intermediates = True

            if return_mems:
                hiddens = intermediates_model.hiddens
                new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(
                    mems) else hiddens
                new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

                if not return_intermediates:
                    return out, new_mems

                intermediates_model.mems = new_mems

            if return_intermediates:
                if self.pre_attn_layers is not None:
                    return out, (intermediates_pre_attn_layers, intermediates_model)

            if return_attn:
                attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates_model.attn_intermediates))
                return out, attn_maps

            return out

