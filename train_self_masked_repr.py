# /// script
# dependencies = [
#   "accelerate",
#   "ema_pytorch",
#   "fire",
#   "tqdm",
#   "x-transformers",
#   "x-mlps-pytorch"
# ]
# ///

from __future__ import annotations

import fire
import gzip
import random
from itertools import chain
from collections import namedtuple, defaultdict

import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat
from torch.utils.data import DataLoader, Dataset

from ema_pytorch import EMA
from accelerate import Accelerator

from x_transformers import TransformerWrapper, Decoder, RMSNorm, FeedForward
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

from x_mlps_pytorch import MLP

# Self-Masked Representation Training
# following the similar formula as in 'Self-Flow' from Chefer et al. at Black Forest Labs
# https://bfl.ai/research/self-flow

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def set_dropout_(model: nn.Module, prob: float):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = prob

def get_extraction_index(target_block, repeated_block, repeats):
    base_idx = 2 * (target_block + 1)
    if repeated_block <= target_block:
        return base_idx + 2 * (repeats - 1)
    return base_idx

# ssl wrapper modules

LossBreakdown = namedtuple('LossBreakdown', ['gen_loss', 'ssl_loss', 'ssl_loss_next'])

def default_rep_loss_fn(pred, target):
    cos_sim = F.cosine_similarity(pred, target, dim = -1)
    return 1. - cos_sim.mean()

class SelfDistilledTraining(nn.Module):
    def __init__(
        self,
        net: TransformerWrapper,
        mask_ratio = 0.15,
        ema_beta = 0.999,
        kl_loss_weight = 0.1,
        use_self_attn_kv_mask = True,
        use_asymmetric_dropout = False,
        student_dropout_rate = 0.,
        teacher_dropout_rate = 0.,
        teacher_extra_repeats = 1,
        student_recurrent_block_repeat_max = 3,
        recurrent_block = None
    ):
        super().__init__()
        assert use_self_attn_kv_mask or use_asymmetric_dropout, 'must use either asymmetric dropout, self attention kv masking, or both'
        assert not use_asymmetric_dropout or student_dropout_rate > teacher_dropout_rate, 'student must have greater dropout rate than teacher to ensure teacher has a better view'

        self.student = AutoregressiveWrapper(net)
        self.teacher = EMA(net, beta = ema_beta)

        self.mask_ratio = mask_ratio
        self.kl_loss_weight = kl_loss_weight
        self.has_ssl_loss = kl_loss_weight > 0

        self.use_self_attn_kv_mask = use_self_attn_kv_mask
        self.use_asymmetric_dropout = use_asymmetric_dropout
        self.student_dropout_rate = student_dropout_rate
        self.teacher_dropout_rate = teacher_dropout_rate
        self.teacher_extra_repeats = teacher_extra_repeats

        self.depth = net.attn_layers.depth
        self.recurrent_block = default(recurrent_block, self.depth // 2)
        self.student_recurrent_block_repeat_max = student_recurrent_block_repeat_max

        self.register_buffer('zero', tensor(0.))

    def parameters(self):
        return self.student.parameters()

    def update_teacher(self):
        self.teacher.update()

    def forward(
        self,
        x,
        **kwargs
    ):
        batch, seq_len, device = *x.shape, x.device

        # set block repeats
        student_repeats = random.randint(1, self.student_recurrent_block_repeat_max)
        teacher_repeats = student_repeats + self.teacher_extra_repeats

        student_repeat_blocks = (self.recurrent_block, student_repeats)
        teacher_repeat_blocks = (self.recurrent_block, teacher_repeats)

        # create mask for student
        # 1 is masked, 0 is unmasked

        mask = torch.rand(x.shape, device = device) < self.mask_ratio

        # don't mask the first token just to be safe for AR

        mask[:, 0] = False

        # student pass with masked inputs

        if self.use_asymmetric_dropout:
            set_dropout_(self.student, self.student_dropout_rate)

        student_loss, (student_logits, _) = self.student(
            x,
            return_outputs = True,
            self_attn_kv_mask = ~mask if self.use_self_attn_kv_mask else None,
            repeat_blocks = student_repeat_blocks,
            **kwargs
        )

        if not self.has_ssl_loss:
            return student_loss, LossBreakdown(student_loss, self.zero, None)

        # teacher pass

        if self.use_asymmetric_dropout:
            set_dropout_(self.teacher, self.teacher_dropout_rate)

        with torch.no_grad():
            teacher_inp = x[:, :-1]

            # ema wraps TransformerWrapper

            teacher_logits, _ = self.teacher.ema_model(
                teacher_inp,
                return_intermediates = True,
                repeat_blocks = teacher_repeat_blocks
            )

        # reverse KL divergence loss

        student_log_probs = student_logits.log_softmax(dim = -1)
        teacher_log_probs = teacher_logits.log_softmax(dim = -1)

        ssl_loss = F.kl_div(
            teacher_log_probs,
            student_log_probs,
            log_target = True,
            reduction = 'none'
        ).sum(dim = -1).mean()

        total_loss = student_loss + self.kl_loss_weight * ssl_loss

        return total_loss, LossBreakdown(student_loss, ssl_loss, None)

class SelfPredictRepresentation(Module):
    # https://arxiv.org/abs/2007.05929
    # Max Schwarzer et al.

    def __init__(
        self,
        dim,
        dim_action,
        expansion_factor = 4.,
        detach_actions = True
    ):
        super().__init__()
        self.state_norm = nn.RMSNorm(dim)

        self.mlp = MLP(
            dim + dim_action,
            int(dim * expansion_factor),
            dim
        )

        self.detach_actions = detach_actions

    def forward(
        self,
        states,
        actions
    ):
        if self.detach_actions:
            actions = actions.detach()

        states = self.state_norm(states)

        state_action = cat((states, actions), dim = -1)

        return self.mlp(state_action)

class SelfMaskedRepTraining(Module):
    def __init__(
        self,
        net: TransformerWrapper,
        mask_ratio = 0.15,
        ema_beta = 0.999,
        rep_loss_weight = 0.1,
        student_layer = 2,
        teacher_layer = 4,
        predict_next_teacher = False,
        predict_head_expansion = 4,
        loss_fn = default_rep_loss_fn,
        use_self_attn_kv_mask = True,
        use_asymmetric_dropout = False,
        student_dropout_rate = 0.,
        teacher_dropout_rate = 0.,
        teacher_extra_repeats = 1,
        student_recurrent_block_repeat_max = 3,
        recurrent_block = None
    ):
        super().__init__()
        assert use_self_attn_kv_mask or use_asymmetric_dropout, 'must use either asymmetric dropout, self attention kv masking, or both'
        assert not use_asymmetric_dropout or student_dropout_rate > teacher_dropout_rate, 'student must have greater dropout rate than teacher to ensure teacher has a better view'

        self.student = AutoregressiveWrapper(net)
        self.teacher = EMA(net, beta = ema_beta)

        self.mask_ratio = mask_ratio
        self.rep_loss_weight = rep_loss_weight
        self.has_ssl_loss = rep_loss_weight > 0

        self.use_self_attn_kv_mask = use_self_attn_kv_mask
        self.use_asymmetric_dropout = use_asymmetric_dropout
        self.student_dropout_rate = student_dropout_rate
        self.teacher_dropout_rate = teacher_dropout_rate
        self.teacher_extra_repeats = teacher_extra_repeats

        self.student_layer = student_layer
        self.teacher_layer = teacher_layer

        self.recurrent_block = default(recurrent_block, student_layer + (teacher_layer - student_layer) // 2)
        self.student_recurrent_block_repeat_max = student_recurrent_block_repeat_max

        self.predict_next_teacher = predict_next_teacher
        self.loss_fn = loss_fn

        # prediction head logic

        dim = net.attn_layers.dim

        self.student_predict_head = nn.Sequential(
            RMSNorm(dim),
            FeedForward(dim, mult = predict_head_expansion)
        )

        if self.predict_next_teacher:
            self.student_predict_next_head = SelfPredictRepresentation(dim, dim_action = dim, expansion_factor = predict_head_expansion)

        self.register_buffer('zero', tensor(0.))

    def parameters(self):
        heads = [self.student_predict_head.parameters()]

        if self.predict_next_teacher:
            heads.append(self.student_predict_next_head.parameters())

        return chain(
            self.student.parameters(),
            *heads
        )

    def update_teacher(self):
        self.teacher.update()

    def forward(
        self,
        x,
        **kwargs
    ):
        batch, seq_len, device = *x.shape, x.device

        # set block repeats

        student_repeats = random.randint(1, self.student_recurrent_block_repeat_max)
        teacher_repeats = student_repeats + self.teacher_extra_repeats

        student_repeat_blocks = (self.recurrent_block, student_repeats)
        teacher_repeat_blocks = (self.recurrent_block, teacher_repeats)

        # create mask for student
        # 1 is masked, 0 is unmasked

        mask = torch.rand(x.shape, device = device) < self.mask_ratio

        # don't mask the first token just to be safe for AR

        mask[:, 0] = False

        # student pass with masked inputs

        if self.use_asymmetric_dropout:
            set_dropout_(self.student, self.student_dropout_rate)

        student_loss, (student_logits, student_cache) = self.student(
            x,
            return_outputs = True,
            self_attn_kv_mask = ~mask if self.use_self_attn_kv_mask else None,
            repeat_blocks = student_repeat_blocks,
            **kwargs
        )

        if not self.has_ssl_loss:
            return student_loss, LossBreakdown(student_loss, self.zero, None)

        # extract student representation at layer l

        student_hiddens = student_cache.layer_hiddens
        student_idx = get_extraction_index(self.student_layer, self.recurrent_block, student_repeats)
        student_rep = student_hiddens[student_idx]

        # teacher pass with unmasked (cleaner) inputs

        if self.use_asymmetric_dropout:
            set_dropout_(self.teacher, self.teacher_dropout_rate)

        with torch.no_grad():
            teacher_inp = x if self.predict_next_teacher else x[:, :-1]

            _, teacher_cache = self.teacher.ema_model(
                teacher_inp,
                return_intermediates = True,
                repeat_blocks = teacher_repeat_blocks
            )

            teacher_hiddens = teacher_cache.layer_hiddens
            teacher_idx = get_extraction_index(self.teacher_layer, self.recurrent_block, teacher_repeats)
            teacher_rep = teacher_hiddens[teacher_idx]

        # cosine similarity representation loss

        student_rep = student_rep[:, :-1]

        # prediction head

        student_pred = self.student_predict_head(student_rep)

        # teacher_rep is length n if predict_next_teacher else n - 1

        teacher_rep_current = teacher_rep[:, :-1] if self.predict_next_teacher else teacher_rep
        ssl_loss = self.loss_fn(student_pred, teacher_rep_current)

        ssl_loss_next = None

        if self.predict_next_teacher:
            teacher_rep_next = teacher_rep[:, 1:]

            # sampled token is an 'action'

            next_sampled_token_ids = x[:, 1:]
            next_token_embed = self.student.net.token_emb(next_sampled_token_ids)

            # student predicts the teacher representation based off current representation + sampled action

            student_pred_next = self.student_predict_next_head(student_rep, next_token_embed)

            ssl_loss_next = self.loss_fn(student_pred_next, teacher_rep_next)

            ssl_loss = (ssl_loss + ssl_loss_next) / 2

        total_loss = student_loss + self.rep_loss_weight * ssl_loss

        return total_loss, LossBreakdown(student_loss, ssl_loss, ssl_loss_next)

# data helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

# training function

def train(
    num_batches = 2000,
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = 256,
    seq_len = 256,
    mask_ratio = 0.15,
    ema_beta = 0.999,
    rep_loss_weight = 0.1,
    student_layer = 3,
    teacher_layer = 5,
    predict_next_teacher = False,
    predict_head_expansion = 4,
    use_ssl = True,
    distill_type = 'cosine',
    use_self_attn_kv_mask = True,
    use_asymmetric_dropout = False,
    student_dropout_rate = 0.,
    teacher_dropout_rate = 0.,
    teacher_extra_repeats = 1,
    student_recurrent_block_repeat_max = 3,
    recurrent_block = None
):
    # accelerator

    accelerator = Accelerator()
    device = accelerator.device

    # rep loss weight

    if not use_ssl:
        rep_loss_weight = 0.

    # model

    student = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = seq_len,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = True
        )
    )

    # ssl wrapper

    assert distill_type in ('cosine', 'reverse_kl'), f'unknown distill type {distill_type}'

    if distill_type == 'cosine':
        ssl_wrapper = SelfMaskedRepTraining(
            student,
            mask_ratio = mask_ratio,
            ema_beta = ema_beta,
            rep_loss_weight = rep_loss_weight,
            student_layer = student_layer,
            teacher_layer = teacher_layer,
            predict_next_teacher = predict_next_teacher,
            predict_head_expansion = predict_head_expansion,
            use_self_attn_kv_mask = use_self_attn_kv_mask,
            use_asymmetric_dropout = use_asymmetric_dropout,
            student_dropout_rate = student_dropout_rate,
            teacher_dropout_rate = teacher_dropout_rate,
            teacher_extra_repeats = teacher_extra_repeats,
            student_recurrent_block_repeat_max = student_recurrent_block_repeat_max,
            recurrent_block = recurrent_block
        )
    elif distill_type == 'reverse_kl':
        ssl_wrapper = SelfDistilledTraining(
            student,
            mask_ratio = mask_ratio,
            ema_beta = ema_beta,
            kl_loss_weight = rep_loss_weight,
            use_self_attn_kv_mask = use_self_attn_kv_mask,
            use_asymmetric_dropout = use_asymmetric_dropout,
            student_dropout_rate = student_dropout_rate,
            teacher_dropout_rate = teacher_dropout_rate,
            teacher_extra_repeats = teacher_extra_repeats,
            student_recurrent_block_repeat_max = student_recurrent_block_repeat_max,
            recurrent_block = recurrent_block
        )

    ssl_wrapper = accelerator.prepare(ssl_wrapper)

    # enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, drop_last = True)

    # optimizer

    optim = torch.optim.Adam(ssl_wrapper.parameters(), lr = learning_rate)

    train_loader, val_loader, optim = accelerator.prepare(train_loader, val_loader, optim)

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training loop

    pbar = tqdm(range(num_batches), mininterval = 10., desc = 'training')

    for i in pbar:
        step = i + 1

        ssl_wrapper.train()

        total_loss = 0.

        total_breakdown = defaultdict(float)

        for __ in range(gradient_accumulate_every):
            seq = next(train_loader)

            loss, breakdown = ssl_wrapper(seq)

            for k, v in breakdown._asdict().items():
                if not exists(v) or not is_tensor(v):
                    continue
                total_breakdown[k] += v.item() / gradient_accumulate_every

            loss = loss / gradient_accumulate_every

            accelerator.backward(loss)

            total_loss += loss.item()

        breakdown_str = ' | '.join(f"{k}: {v:.3f}" for k, v in total_breakdown.items())
        accelerator.print(f'{i}: loss: {total_loss:.3f} | {breakdown_str}')

        accelerator.clip_grad_norm_(ssl_wrapper.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        # teacher update

        accelerator.unwrap_model(ssl_wrapper).update_teacher()

        # validation

        if divisible_by(step, validate_every):
            ssl_wrapper.eval()
            with torch.no_grad():
                seq = next(val_loader)
                loss, _ = ssl_wrapper(seq)

                accelerator.print(f'validation loss: {loss.item():.3f}')

        # generation

        if divisible_by(step, generate_every) and accelerator.is_main_process:
            ssl_wrapper.eval()

            # unwrapped student used for generation

            unwrapped_student = accelerator.unwrap_model(ssl_wrapper).student

            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp.cpu().numpy())

            accelerator.print(f'\n{prime} \n\n {"*" * 100}')

            sample = unwrapped_student.generate(
                prompts = inp.unsqueeze(0).to(device),
                seq_len = generate_length,
                cache_kv = True
            )

            output_str = decode_tokens(sample[0].cpu().numpy())
            accelerator.print(f'{output_str}\n')

if __name__ == '__main__':
    fire.Fire(train)
