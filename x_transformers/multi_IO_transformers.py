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
            embed_num_tokens: Dict[str, int] = dict(),  # not sure how to utilize
            emb_dim: list[int] or int = None,
            max_mem_len=0,  # not sure how to utilize
            shift_mem_down=0,  # not sure how to utilize
            emb_dropout=0.,
            post_emb_norm=False,
            post_attn_layers: list[AttentionLayers] = None,
            num_memory_tokens=None,  # not sure how to utilize
            memory_tokens_interspersed_every=None,  # not sure how to utilize
            tie_embedding=False,
            logits_dim: list[int] or int = None,
            use_abs_pos_emb=True,
            scaled_sinu_pos_emb=False,
            l2norm_embed=False,
            emb_frac_gradient=1.,  # GLM-130B and Cogview successfully used this, set at 0.1
            attn_z_loss_weight=1e-4,
    ):
        """
        num_tokens: list of number of tokens for each input
        max_seq_len: maximum sequence length
        pre_attn_layers: list of AttentionLayers for each input
        concat_emb_dim: whether to concat the embedding dimensions for each input or add them
        attn_layers: AttentionLayers for the whole model (not compatible with singular num_tokens)
        embed_num_tokens: dictionary of number of tokens for each embedding
        emb_dim: list of embedding dimensions for each input (not applicable with pre_attn_layers)
        max_mem_len: maximum memory length
        shift_mem_down: shift memory down
        emb_dropout: embedding dropout
        post_emb_norm: whether to use post embedding norm
        post_attn_layers: list of AttentionLayers for each input after the main AttentionLayers
        num_memory_tokens: number of memory tokens
        memory_tokens_interspersed_every: intersperse memory tokens every x tokens
        tie_embedding: whether to tie the embedding and the logits layer
        logits_dim: logits dimension (not applicable with tie_embedding or post_attn_layers)
        use_abs_pos_emb: whether to use absolute positional embedding
        scaled_sinu_pos_emb: whether to use scaled sinusoidal positional embedding
        l2norm_embed: whether to use l2 normalization on the embedding
        emb_frac_gradient: fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290
        attn_z_loss_weight: weight for the attention regularization loss
        """
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
        self.emb_dim = emb_dim if not pre_attn_layers else [layer.dim for layer in pre_attn_layers]
        self.num_tokens = num_tokens if not pre_attn_layers else [layer.num_tokens for layer in pre_attn_layers]

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.pre_attn_layers = pre_attn_layers
        if pre_attn_layers is not None:
            assert type(num_tokens) == list, 'num_tokens must be a list of number of tokens for each input'
            assert len(pre_attn_layers) == len(num_tokens), 'number of pre_attn_layers must match number of inputs'
            if concat_emb_dim:
                # assert that sum of embedding dimensions is equal to the model dimension
                assert sum(emb_dim) == dim, 'sum of embedding dimensions must be equal to the model dimension'

        self.concat_emb_dim = concat_emb_dim if pre_attn_layers else False

        self.post_attn_layers = post_attn_layers

        if post_attn_layers is not None:
            assert type(logits_dim) == list, 'logits_dim must be a list of logits dimension for each output'
            assert len(post_attn_layers) == len(logits_dim), 'number of post_attn_layers must match number of outputs'

        self.l2norm_embed = l2norm_embed

        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed) if not self.multi_input \
            else [TokenEmbedding(emb_dim[i], num_tokens[i], l2norm_embed=l2norm_embed) for i in \
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
            self.embeds = nn.ModuleDict(
                {f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        logits_dim = default(logits_dim, num_tokens)
        if type(logits_dim) == list:
            self.to_logits = [(nn.Linear(dim, d, bias=False) for d in logits_dim) if not tie_embedding else lambda \
                    t: t @ self.token_emb.emb.weight.t()]
        else:
            self.to_logits = nn.Linear(dim, logits_dim, bias=False) if not tie_embedding else lambda \
                    t: t @ self.token_emb.emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std=1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

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
        global intermediates_model
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient = x.shape[0], x.shape[
            1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        if not self.multi_input and not self.multi_output:
            return self.model(x, return_embeddings, return_logits_and_embeddings, return_intermediates, mask,
                              return_mems, return_attn, mems, mem_masks, pos, prepend_mask, embed_ids, sum_embeds,
                              return_attn_z_loss, attn_z_loss_weight, seq_start_pos, cache)
        elif not self.multi_input:
            # multi_output system
            cache_model, cache_post_attn_layers = cache
            if return_mems or return_attn or return_intermediates or return_attn_z_loss:
                x, intermediates_model = self.model(x, True, False, return_intermediates, mask,
                                                    return_mems, return_attn, mems, mem_masks, pos, prepend_mask,
                                                    embed_ids, sum_embeds,
                                                    return_attn_z_loss, attn_z_loss_weight, seq_start_pos, cache_model)
            else:
                x = self.model(x, True, False, return_intermediates, mask,
                               return_mems, return_attn, mems, mem_masks, pos, prepend_mask, embed_ids, sum_embeds,
                               return_attn_z_loss, attn_z_loss_weight, seq_start_pos, cache_model)
            if self.post_attn_layers is not None:
                outputs = []
                intermediates = []
                x_values = []
                for i, layer in enumerate(self.post_attn_layers):
                    if return_mems or return_attn or return_intermediates or return_attn_z_loss:
                        x, inter = layer(x, mask=mask, mems=mems, mem_masks=mem_masks, cache=cache_post_attn_layers[i],
                                         return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs)
                        intermediates.append(inter)
                        x_values.append(x)
                    else:
                        x_values.append(
                            layer(x, mask=mask, mems=mems, mem_masks=mem_masks, cache=cache_post_attn_layers[i],
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
                                             for intermediate in intermediates)
                    for i in range(len(intermediates)):
                        intermediates[i].attn_z_loss = calc_z_loss(pre_softmax_attns[i], weight=attn_z_loss_weight)
                    return_intermediates = True

                if return_mems:
                    for i in range(len(intermediates)):
                        hiddens = intermediates[i].hiddens
                        new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(
                            mems) else hiddens
                        new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

                        if not return_intermediates:
                            return out, new_mems

                        intermediates[i].mems = new_mems
                        intermediates[i].mems = hiddens

                if return_intermediates:
                    return out, (intermediates_model, intermediates)

                if return_attn:
                    attn_maps = list(list(map(lambda t: t.post_softmax_attn, intermediate.attn_intermediates))
                                     for intermediate in intermediates)
                    return out, (intermediates_model, attn_maps)

                return out
        elif not self.multi_output:
            # only multi_input, I'll finish layer
            pass
        else:
            # multi_input and multi_output
            # multi_input = [..., x inputs]
            if self.pre_attn_layers is not None:
                assert len(x[:, :, -1]) == len(self.pre_attn_layers), 'number of inputs must match number of ' \
                                                                      'pre_attn_layers'
            assert len(x[:, :, -1]) == len(self.num_tokens), 'number of inputs must match number of num_tokens'
            assert len(x[:, :, -1]) == len(self.emb_dim), 'number of inputs must match number of emb_dim'
            external_pos_emb = exists(pos) and pos.dtype != torch.long
            pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
            for i in range(len(x[:, :, -1])):
                x_i = x[:, :, i]
                x_i = self.token_emb(x_i) + pos_emb
                if exists(self.embeds):
                    assert len(embed_ids) == len(self.embeds)

                    for name, embed_id in embed_ids.items():
                        embed_key = f'{name}_embed'

                        assert embed_key in self.embeds
                        embed = self.embeds[embed_key](embed_id)

                        x_i = x_i + embed
                # following have to be done for each input
                #x_i = self.post_emb_norm(x_i)
                #x_i = self.emb_dropout(x_i)
                #x_i = self.project_emb(x_i)
                if self.pre_attn_layers is not None:
                    x_i = self.pre_attn_layers[i](x_i, mask=mask, mems=mems, cache=cache, return_hiddens=False,
                                                  seq_start_pos=seq_start_pos, **kwargs)
                if i == 0:
                    x = x_i
                else:
                    if self.concat_emb_dim:
                        x = torch.cat((x, x_i), dim=-1)
                    else:
                        x = x + x_i

        # absolute positional embedding
        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb
        # add additional embeddings
        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed
        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training
        if exists(sum_embeds):
            x = x + sum_embeds
        # post embedding norm, purportedly leads to greater stabilization
        x = self.post_emb_norm(x)
        # whether to append embeds, as in PaLI, for image embeddings
        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[
                -1], 'prepended embeddings need to have same dimensions as text model dimensions'
            x = torch.cat((prepend_embeds, x), dim=-2)
            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device=device, dtype=torch.bool))
                prepend_mask = default(prepend_mask,
                                       lambda: torch.ones((b, prepend_seq), device=device, dtype=torch.bool))
                mask = torch.cat((prepend_mask, mask), dim=-1)
        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model
        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)
        # embedding dropout
        x = self.emb_dropout(x)
        x = self.project_emb(x)
        if has_memory_tokens:
            mem_every = self.memory_tokens_interspersed_every
            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every
                x = pad_at_dim(x, (0, next_seq_len - n), dim=-2, value=0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=mem_every)
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')
            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)
            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)
        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]
        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, mem_masks=mem_masks, cache=cache,
                                            return_hiddens=True, seq_start_pos=seq_start_pos, **kwargs)

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=(mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

            x = x[:, :n]

        if return_logits_and_embeddings:
            out = (self.to_logits(x), x)
        elif return_embeddings:
            out = x
        else:
            out = self.to_logits(x)

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
