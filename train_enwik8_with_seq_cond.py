# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers",
#   "wandb",
#   "fire",
#   "accelerate"
# ]
# ///

import gzip
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
import tqdm
import fire
import wandb

from x_transformers import (
    TransformerWrapper,
    Decoder,
    AutoregressiveWrapper,
    DSConvConditioner,
    EMAConditioner
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq.to(self.device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

# train

def train(
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = None,
    seq_len = 1024,
    dim = 512,
    depth = 6,
    heads = 8,
    seq_cond = 'dsconv',
    attn_seq_cond = None,
    ff_seq_cond = None,
    seq_cond_dim = None,
    seq_cond_kernel_size = 4,
    seq_cond_scale_groups = 4,
    seq_cond_every = 1,
    track_experiment_online = False,
    run_name = 'seq-cond',
    cpu = False
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    generate_length = default(generate_length, seq_len)

    def make_cond(spec):
        if not spec:
            return None
        if spec in (True, 'dsconv'):
            return DSConvConditioner(dim, dim_cond = seq_cond_dim, kernel_size = seq_cond_kernel_size)
        if spec == 'ema':
            return EMAConditioner(dim, dim_cond = seq_cond_dim)
        raise ValueError(f'unknown conditioner spec: {spec}')

    has_split_cond = exists(attn_seq_cond) or exists(ff_seq_cond)

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = seq_len,
        attn_layers = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            rotary_pos_emb = True,
            ff_glu = True,
            ff_swish = True,
            pre_and_post_norm = True,
            seq_cond = None if has_split_cond else make_cond(seq_cond),
            attn_seq_cond = make_cond(attn_seq_cond),
            ff_seq_cond = make_cond(ff_seq_cond),
            seq_cond_scale_groups = seq_cond_scale_groups,
            seq_cond_every = seq_cond_every
        )
    )

    model = AutoregressiveWrapper(model)

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    train_dataset = TextSamplerDataset(data_train, seq_len, device)
    val_dataset   = TextSamplerDataset(data_val, seq_len, device)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))

    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    wandb.init(project = 'enwik8', mode = 'online' if track_experiment_online else 'disabled')
    wandb.run.name = run_name

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    for i in tqdm.tqdm(range(num_batches), mininterval = 10., desc = 'training'):
        model.train()

        for _ in range(gradient_accumulate_every):
            loss = model(next(train_loader))
            accelerator.backward(loss / gradient_accumulate_every)

        print(f'training loss: {loss.item()}')
        if accelerator.is_main_process:
            wandb.log(dict(loss = loss.item()))

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))

                print(f'validation loss: {loss.item()}')
                if accelerator.is_main_process:
                    wandb.log(dict(valid_loss = loss.item()))

        if divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp.cpu().numpy())
            print(f'%s \n\n %s' % (prime, '*' * 100))

            sample = model.generate(
                prompts = inp,
                seq_len = generate_length,
                cache_kv = True
            )

            output_str = decode_tokens(sample.cpu().numpy())
            print(output_str)

if __name__ == '__main__':
    fire.Fire(train)
