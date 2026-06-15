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
from x_transformers.autoregressive_wrapper import top_p, top_k, eval_decorator

from torch_einops_utils import masked_mean, mask_after

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_empty(arr):
    return len(arr) == 0

def divisible_by(numer, denom):
    return (numer % denom) == 0

def get_module_by_path(model, path):
    return reduce(getattr, path.split('.'), model)

def set_module_by_path(model, path, new_mod):
    *parts, last = path.split('.')
    mod = reduce(getattr, parts, model)
    setattr(mod, last, new_mod)

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
        ttt_custom_loss_module: Module | None = None
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.tbptt_steps = tbptt_steps

        self.net = net
        self.max_seq_len = net.max_seq_len

        self.ttt_module_paths = ttt_module_paths
        self.ttt_lr = ttt_lr
        self.ttt_wd = ttt_wd

        self.ttt_custom_loss_module = ttt_custom_loss_module

        self.ttt_wrappers = []
        for path in ttt_module_paths:
            mod = get_module_by_path(net, path)
            wrapper = TTTModuleWrapper(mod)
            set_module_by_path(net, path, wrapper)
            self.ttt_wrappers.append(wrapper)

        if self.has_ttt:
            assert tbptt_steps > 1, 'tbptt_steps must be greater than 1 if ttt is turned on'

    @property
    def has_ttt(self):
        return not is_empty(self.ttt_wrappers)

    def init_ttt(self, batch):
        if not self.has_ttt:
            return

        for wrapper in self.ttt_wrappers:
            wrapper.batch_params = {
                name: repeat(param, '... -> b ...', b = batch)
                for name, param in wrapper.module.named_parameters()
            }

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
            loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss
            loss_t = loss_fn(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = self.ignore_index,
                reduction = 'none'
            )

            loss_t_per_batch = masked_mean(loss_t, mask)

        all_batch_params = [
            param
            for wrapper in self.ttt_wrappers
            for param in wrapper.batch_params.values()
        ]

        # compute gradients

        grads = torch.autograd.grad(
            loss_t_per_batch.sum(),
            all_batch_params,
            create_graph = create_graph,
            retain_graph = create_graph,
            allow_unused = True
        )

        # update test-time trained parameters

        grad_idx = 0
        for wrapper in self.ttt_wrappers:
            num_params = len(wrapper.batch_params)
            wrapper_grads = grads[grad_idx : grad_idx + num_params]
            grad_idx += num_params

            wrapper.batch_params = {
                name: (param - self.ttt_lr * grad - self.ttt_lr * self.ttt_wd * param) if exists(grad) else param
                for (name, param), grad in zip(wrapper.batch_params.items(), wrapper_grads)
            }

            if not create_graph:
                wrapper.batch_params = {
                    name: param.detach().requires_grad_()
                    for name, param in wrapper.batch_params.items()
                }

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
                chunk_labels = start_tokens[:, curr_pos + 1 : curr_pos + max_seq_len + 1]
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

        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

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

                if should_detach:
                    for wrapper in self.ttt_wrappers:
                        wrapper.batch_params = {
                            name: param.detach().requires_grad_(param.requires_grad)
                            for name, param in wrapper.batch_params.items()
                        }

        return total_loss
