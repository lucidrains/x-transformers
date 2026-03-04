# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers",
#   "ema_pytorch",
#   "accelerate",
#   "fire",
# ]
# ///

from __future__ import annotations

import gzip
import random
import numpy as np

import torch
from torch import nn, Tensor, tensor, is_tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import fire
from tqdm import tqdm

from ema_pytorch import EMA
from accelerate import Accelerator

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

# Self-Masked Representation Training
# following the similar formula as in 'Self-Flow' from Chefer et al.
# https://bfl.ai/research/self-flow

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# ssl wrapper module

class SelfMaskedRepTraining(nn.Module):
    def __init__(
        self,
        net: TransformerWrapper,
        mask_ratio = 0.15,
        ema_beta = 0.999,
        rep_loss_weight = 0.1,
        student_layer = 2,
        teacher_layer = 4
    ):
        super().__init__()
        self.student = AutoregressiveWrapper(net)
        self.teacher = EMA(net, beta = ema_beta)

        self.mask_ratio = mask_ratio
        self.rep_loss_weight = rep_loss_weight

        self.student_layer = student_layer
        self.teacher_layer = teacher_layer

        self.register_buffer('zero', tensor(0.))

    def update_teacher(self):
        self.teacher.update()

    def forward(
        self,
        x,
        **kwargs
    ):
        batch, seq_len, device = *x.shape, x.device

        # create mask for student
        # 1 is masked, 0 is unmasked
        
        mask = torch.rand(x.shape, device = device) < self.mask_ratio
        
        # don't mask the first token just to be safe for AR

        mask[:, 0] = False

        # student pass with masked inputs

        student_loss, (student_logits, student_cache) = self.student(
            x,
            return_outputs = True,
            self_attn_kv_mask = ~mask,
            **kwargs
        )

        has_ssl_loss = self.rep_loss_weight > 0

        if not has_ssl_loss:
            return student_loss, (student_loss, self.zero)

        # extract student representation at layer l
        
        student_hiddens = student_cache.layer_hiddens
        student_rep = student_hiddens[self.student_layer]

        # teacher pass with unmasked (cleaner) inputs
        
        with torch.no_grad():
            teacher_inp = x[:, :-1]
            
            # ema wraps TransformerWrapper
            
            _, teacher_cache = self.teacher.ema_model(
                teacher_inp,
                return_intermediates = True,
            )
            
            teacher_hiddens = teacher_cache.layer_hiddens
            teacher_rep = teacher_hiddens[self.teacher_layer]

        # cosine similarity representation loss
        
        student_rep = student_rep[:, :-1]

        # teacher_rep is already length n - 1

        cos_sim = F.cosine_similarity(student_rep, teacher_rep, dim = -1)
        
        # mean positive-valued cosine similarity loss going to 0
        # following the representation alignment paper (REPA) which aligns the full sequence
        # and "Self-Flow" which uses a similar self-supervised representation objective
        
        ssl_loss = 1. - cos_sim.mean()

        total_loss = student_loss + self.rep_loss_weight * ssl_loss

        return total_loss, (student_loss, ssl_loss)

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
    student_layer = 2,
    teacher_layer = 4,
    use_ssl = True
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

    ssl_wrapper = SelfMaskedRepTraining(
        student,
        mask_ratio = mask_ratio,
        ema_beta = ema_beta,
        rep_loss_weight = rep_loss_weight,
        student_layer = student_layer,
        teacher_layer = teacher_layer
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
        ssl_wrapper.train()

        total_loss = 0.
        total_gen_loss = 0.
        total_ssl_loss = 0.

        for __ in range(gradient_accumulate_every):
            seq = next(train_loader)
            
            loss, (gen_loss, ssl_loss) = ssl_wrapper(seq)
            
            loss = loss / gradient_accumulate_every
            
            accelerator.backward(loss)

            total_loss += loss.item()
            total_gen_loss += gen_loss.item() / gradient_accumulate_every
            total_ssl_loss += ssl_loss.item() / gradient_accumulate_every if is_tensor(ssl_loss) else 0.
                
        accelerator.print(f'{i}: loss: {total_loss:.3f} | gen: {total_gen_loss:.3f} | ssl: {total_ssl_loss:.3f}')

        accelerator.clip_grad_norm_(ssl_wrapper.parameters(), 0.5)
        optim.step()
        optim.zero_grad()
        
        # teacher update
        
        accelerator.unwrap_model(ssl_wrapper).update_teacher()

        # validation

        if (i + 1) % validate_every == 0:
            ssl_wrapper.eval()
            with torch.no_grad():
                seq = next(val_loader)
                loss, _ = ssl_wrapper(seq)
                
                accelerator.print(f'validation loss: {loss.item():.3f}')

        # generation

        if (i + 1) % generate_every == 0 and accelerator.is_main_process:
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
