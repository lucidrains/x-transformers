# /// script
# dependencies = [
#   "tqdm",
#   "x-transformers",
#   "wandb",
#   "fire",
#   "accelerate"
# ]
# ///

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import fire
import wandb
from accelerate import Accelerator

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def train(
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = None,
    seq_len = 1024,
    track_experiment_online = False,
    run_name = 'baseline',
    cpu = False
):
    accelerator = Accelerator(cpu=cpu)
    device = accelerator.device

    generate_length = default(generate_length, seq_len)

    # instantiate GPT-like decoder model

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = seq_len,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = False,
            polar_pos_emb = True,
            pre_and_post_norm = True
        )
    )

    model = AutoregressiveWrapper(model)

    # prepare enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq.to(device)

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # experiment

    wandb.init(project = 'enwik8', mode = 'online' if track_experiment_online else 'disabled')
    wandb.run.name = run_name

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval=10., desc='training'):
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

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))

                print(f'validation loss: {loss.item()}')
                if accelerator.is_main_process:
                    wandb.log(dict(valid_loss = loss.item()))

        if i % generate_every == 0:
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
