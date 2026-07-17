from __future__ import annotations
from math import ceil
from functools import reduce
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.func import functional_call, vmap

from einops import rearrange, repeat, pack, unpack
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator, cast_tuple

from torch_einops_utils import masked_mean, mask_after, maybe, tree_map_tensor
from torch_einops_utils.device import module_device

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_empty(arr):
    return len(arr) == 0

def first(it, default = None):
    return next(iter(it), default)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def get_module_by_path(model, path):
    return reduce(getattr, path.split('.'), model)

def set_module_by_path(model, path, new_mod):
    *parts, last = path.split('.')
    mod = reduce(getattr, parts, model)
    setattr(mod, last, new_mod)

# tensor functions

def repeat_batch(t, batch_size):
    return repeat(t, '... -> b ...', b = batch_size)

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(t_in):
        return unpack(t_in, ps, pattern)[0]
    return packed, inverse

def transpose(t):
    return rearrange(t, '... i j -> ... j i')

# Muon update rule (Newton-Schulz) - Keller Jordan et al.

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim < 3:
        return t, False

    rows, cols = t.shape[-2:]
    should_transpose = rows > cols

    if should_transpose:
        t = transpose(t)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ transpose(t)
        B = b * A + c * A @ A
        t = a * t + B @ t

    t = inv_pack(t)

    if should_transpose:
        t = transpose(t)

    return t, True

# episodic memories

class EpisodicMemories(Module):
    def __init__(
        self,
        depth,
        heads,
        seq_len,
        dim_head
    ):
        super().__init__()
        self.mem_kv = nn.Parameter(torch.randn(2, depth, heads, seq_len, dim_head))
        nn.init.normal_(self.mem_kv, std = 0.02)

    def forward(self):
        return self.mem_kv

class EpisodicMemoryWrapper(Module):
    def __init__(
        self,
        net,
        episodic_mem_len
    ):
        super().__init__()
        self.net = net

        with torch.no_grad():
            mock_input = torch.zeros((1, 1), dtype = torch.long, device = module_device(net))
            _, intermediates = net(mock_input, return_intermediates = True)

        attn_intermediates = [interm for interm in intermediates.attn_intermediates if interm.layer_type == 'a']

        assert not is_empty(attn_intermediates), 'no attention layers found for episodic memory wrapper'

        first_cached_kv = first(attn_intermediates).cached_kv[0]
        _, kv_heads, _, dim_head = first_cached_kv.shape
        depth = len(attn_intermediates)

        self.episodic_mems = EpisodicMemories(
            depth = depth,
            heads = kv_heads,
            seq_len = episodic_mem_len,
            dim_head = dim_head
        )

    def get_additional_kv(self, batch_size):
        mem_kv = self.episodic_mems()

        if mem_kv.ndim == 5:
            mem_kv = repeat_batch(mem_kv, batch_size)

        mem_k, mem_v = mem_kv.unbind(dim = 1)
        return [(k, v) for k, v in zip(mem_k.unbind(dim = 1), mem_v.unbind(dim = 1))]

    def forward(self, x, **kwargs):
        batch = x.shape[0]

        kwargs.update(
            self_attn_additional_kv = self.get_additional_kv(batch)
        )

        return self.net(x, **kwargs)

# ttt module wrapper

class TTTModuleWrapper(Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.batch_params = None

    def reset(self):
        self.batch_params = None

    @staticmethod
    def reset_all(module):
        for mod in module.modules():
            if isinstance(mod, TTTModuleWrapper):
                mod.reset()

    def forward(self, *args, **kwargs):
        if not exists(self.batch_params):
            return self.module(*args, **kwargs)

        def call_single(params, *args_single):
            return functional_call(self.module, params, args_single, kwargs)

        return vmap(call_single)(self.batch_params, *args)

# custom TTT loss modules

class TTTMetaLearningTargetKLLoss(Module):
    def __init__(
        self,
        *,
        dim,
        num_classes,
        depth = 1,
        heads = 4,
        dim_head = 64,
        **kwargs
    ):
        super().__init__()
        from x_transformers.x_transformers import Encoder

        self.encoder = Encoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **kwargs
        )

        self.to_prediction = nn.Linear(dim, num_classes)
        self.to_meta_prediction = nn.Linear(dim, num_classes)

    def forward(self, intermediates, mask = None):
        hiddens = intermediates.last_layer_hiddens

        pred = self.to_prediction(hiddens)

        encoded = self.encoder(hiddens, mask = mask)
        meta_pred = self.to_meta_prediction(encoded)

        prob_earlier = pred.softmax(dim = -1)
        log_prob_encoded = meta_pred.log_softmax(dim = -1)

        loss = F.kl_div(log_prob_encoded, prob_earlier, reduction = 'none').sum(dim = -1)
        return loss

# xl autoregressive wrapper class

class XLAutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        tbptt_steps = 1,
        ttt_module_paths = tuple(),
        ttt_lr = 1e-3,
        ttt_wd = 0.01,
        ttt_use_muon = False,
        ttt_muon_steps = 5,
        ttt_muon_lr = 1e-2,
        ttt_custom_loss_module: Module | None = None,
        episodic_mem_len = 0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.tbptt_steps = tbptt_steps

        self.net = net
        self.max_seq_len = net.max_seq_len
        self.output_is_log_prob = net.output_is_log_prob

        self.ttt_module_paths = ttt_module_paths
        self.ttt_lr = ttt_lr
        self.ttt_wd = ttt_wd
        self.ttt_use_muon = ttt_use_muon
        self.ttt_muon_steps = ttt_muon_steps
        self.ttt_muon_lr = ttt_muon_lr
        self.ttt_custom_loss_module = ttt_custom_loss_module

        self.has_episodic_mem = episodic_mem_len > 0

        # maybe wrap network with episodic memory wrapper

        if self.has_episodic_mem:
            net = EpisodicMemoryWrapper(net, episodic_mem_len)

        self.net = net

        # format ttt module paths to always be (source_path, target_path)

        self.ttt_paths_map = tuple(cast_tuple(item, 2) for item in ttt_module_paths)

        # modify paths to have 'net.' prepended if wrapped

        if self.has_episodic_mem:
            prepended_paths_map = []
            for src, tgt in self.ttt_paths_map:
                src = f'net.{src}' if src != 'episodic_mems' else src
                tgt = f'net.{tgt}' if tgt != 'episodic_mems' else tgt
                prepended_paths_map.append((src, tgt))

            self.ttt_paths_map = tuple(prepended_paths_map)

        # gather all unique wrappers

        self.ttt_wrappers = dict()

        for source_path, target_path in self.ttt_paths_map:
            for path in (source_path, target_path):
                if path in self.ttt_wrappers:
                    continue

                mod = get_module_by_path(net, path)
                wrapper = TTTModuleWrapper(mod)
                set_module_by_path(net, path, wrapper)

                self.ttt_wrappers[path] = wrapper

        if self.has_ttt:
            assert tbptt_steps > 1, 'tbptt_steps must be greater than 1 if ttt is turned on'

    @property
    def has_ttt(self):
        return len(self.ttt_wrappers) > 0

    def init_ttt(self, batch):
        if not self.has_ttt:
            return

        for wrapper in self.ttt_wrappers.values():
            wrapper.batch_params = tree_map_tensor(
                lambda t: repeat_batch(t, batch),
                dict(wrapper.module.named_parameters())
            )

    def update_ttt(
        self,
        logits,        # (b n c)
        chunk_labels,  # (b n)
        create_graph = False,
        intermediates = None
    ):
        if not self.has_ttt:
            return

        # compute loss

        mask = chunk_labels != self.ignore_index

        if exists(self.ttt_custom_loss_module):
            assert exists(intermediates), 'intermediates must be passed to update_ttt when using ttt_custom_loss_module'
            loss_t = self.ttt_custom_loss_module(intermediates, mask = mask)
            loss_t_per_batch = masked_mean(loss_t, mask) if loss_t.ndim > 1 else loss_t
        else:
            loss_fn = F.cross_entropy if not self.output_is_log_prob else F.nll_loss
            loss_t = loss_fn(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = self.ignore_index,
                reduction = 'none'
            )

            loss_t_per_batch = masked_mean(loss_t, mask)

        # gather all batch parameters for source modules

        all_batch_params = [
            param
            for source_path, _ in self.ttt_paths_map
            for param in self.ttt_wrappers[source_path].batch_params.values()
        ]

        # compute gradients

        grads = torch.autograd.grad(
            loss_t_per_batch.sum(),
            all_batch_params,
            create_graph = create_graph,
            retain_graph = create_graph,
            allow_unused = True
        )

        # update target parameters with source gradients

        grad_idx = 0

        for source_path, target_path in self.ttt_paths_map:
            source_wrapper = self.ttt_wrappers[source_path]
            target_wrapper = self.ttt_wrappers[target_path]

            num_params = len(source_wrapper.batch_params)
            wrapper_grads = grads[grad_idx : grad_idx + num_params]
            grad_idx += num_params

            assert len(source_wrapper.batch_params) == len(target_wrapper.batch_params), f'source module {source_path} and target module {target_path} must have the same number of parameters'

            # maybe muon

            if self.ttt_use_muon:
                muon_results = [newtonschulz5(g, steps = self.ttt_muon_steps) if exists(g) else (None, False) for g in wrapper_grads]
                wrapper_grads, is_muon_grads = tuple(zip(*muon_results))
            else:
                is_muon_grads = (False,) * len(wrapper_grads)

            # update test-time trained parameters

            new_batch_params = dict()

            for (src_name, src_param), (tgt_name, tgt_param), grad, is_muon_grad in zip(source_wrapper.batch_params.items(), target_wrapper.batch_params.items(), wrapper_grads, is_muon_grads):
                assert src_param.shape == tgt_param.shape, f'source parameter {source_path}.{src_name} (shape {src_param.shape}) and target parameter {target_path}.{tgt_name} (shape {tgt_param.shape}) must have the exact same shape'

                if exists(grad):
                    lr = self.ttt_muon_lr if is_muon_grad else self.ttt_lr
                    tgt_param = tgt_param - lr * grad - lr * self.ttt_wd * tgt_param

                if not create_graph:
                    tgt_param = tgt_param.detach().requires_grad_()

                new_batch_params[tgt_name] = tgt_param

            target_wrapper.batch_params = new_batch_params

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

        batch, t = start_tokens.shape

        *all_leading_tokens, remainder_tokens = start_tokens.split(max_seq_len, dim = -1)

        forward_context = torch.enable_grad if self.has_ttt else nullcontext

        # catch the memory up to the current segment

        with forward_context():
            self.init_ttt(batch)

            for idx, leading_tokens in enumerate(all_leading_tokens):
                logits, intermediates = self.net(
                    leading_tokens,
                    mems = mems,
                    return_mems = True,
                    return_intermediates = True,
                    **kwargs
                )
                mems = intermediates.mems

                if not self.has_ttt:
                    continue

                curr_pos = idx * max_seq_len
                chunk_labels = start_tokens.narrow(1, curr_pos + 1, max_seq_len)
                self.update_ttt(logits, chunk_labels, create_graph = False, intermediates = intermediates)

        # test-time train on the remainder of the prompt

        if self.has_ttt and remainder_tokens.shape[-1] > 1:
            with forward_context():
                remainder_x = remainder_tokens[:, :-1]
                remainder_labels = remainder_tokens[:, 1:]

                logits, intermediates = self.net(
                    remainder_x,
                    mems = mems,
                    return_mems = True,
                    return_intermediates = True,
                    **kwargs
                )
                mems = intermediates.mems

                self.update_ttt(logits, remainder_labels, create_graph = False, intermediates = intermediates)

        # now start sampling from the current segment

        curr_pos = len(all_leading_tokens) * max_seq_len
        curr_mems = mems

        cache = None
        out = start_tokens

        is_greedy = temperature == 0.

        # unwrap episodic memory wrapper to avoid evaluating it per token

        net = self.net
        if self.has_episodic_mem:
            kwargs.update(
                self_attn_additional_kv = net.get_additional_kv(batch)
            )
            net = net.net

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems
                cache = None

            x = out[:, curr_pos:]

            logits, cache = net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            mems = cache.mems

            logits = logits[:, -1]

            if is_greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim = -1)
                sample = torch.multinomial(probs, 1)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems
                cache = None

            out = torch.cat((out, sample), dim = -1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    mask = mask_after(out, eos_token, inclusive = False)
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        dropout_mems = False,
        ttt_recurrent_steps = 1,
        **kwargs
    ):
        if ttt_recurrent_steps > 1:
            # treating the ttt gradients as the fast weights
            assert self.has_ttt, 'ttt must be turned on to use ttt_recurrent_steps'

        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]

        batch, seq_len = x.shape[:2]

        self.init_ttt(batch)

        # prepare chunks

        split_x = x.split(max_seq_len, dim = -1)
        split_labels = labels.split(max_seq_len, dim = -1)
        loss_weights = tuple((t.shape[-1] / seq_len) for t in split_x)

        loss_fn = F.cross_entropy if not self.output_is_log_prob else F.nll_loss

        # go through each chunk and derive weighted losses

        total_loss = 0.
        num_chunks = len(split_x)
        forward_context = torch.enable_grad if self.has_ttt else nullcontext

        with forward_context():
            for idx, (chunk, chunk_labels, loss_weight) in enumerate(zip(split_x, split_labels, loss_weights)):
                is_last_chunk = (idx == num_chunks - 1)
                should_detach = divisible_by(idx + 1, self.tbptt_steps)

                for recurrent_step in range(ttt_recurrent_steps):
                    is_last_recurrent_step = (recurrent_step == ttt_recurrent_steps - 1)

                    passed_mems = None if dropout_mems else mems

                    logits, intermediates = self.net(
                        chunk,
                        mems = passed_mems,
                        return_mems = True,
                        return_intermediates = True,
                        detach_mems = should_detach,
                        **kwargs
                    )

                    if is_last_recurrent_step:
                        mems = intermediates.mems

                    loss = loss_fn(
                        rearrange(logits, 'b n c -> b c n'),
                        chunk_labels,
                        ignore_index = ignore_index,
                        reduction = 'none'
                    )

                    mask = chunk_labels != ignore_index
                    loss_per_batch = masked_mean(loss, mask)

                    if is_last_recurrent_step:
                        total_loss = total_loss + loss_per_batch.mean() * loss_weight

                    if (is_last_chunk and is_last_recurrent_step) or not self.has_ttt:
                        continue

                    self.update_ttt(logits, chunk_labels, create_graph = self.training, intermediates = intermediates)

                if not should_detach:
                    continue

                # detach

                for wrapper in self.ttt_wrappers.values():
                    wrapper.batch_params = tree_map_tensor(
                        lambda t: t.detach().requires_grad_(t.requires_grad),
                        wrapper.batch_params
                    )

        return total_loss
