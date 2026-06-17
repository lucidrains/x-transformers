# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "x-transformers",
#     "accelerate",
#     "fire",
#     "numpy",
#     "tqdm"
# ]
# ///

from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.entropy_based_tokenizer import EntropyBasedTokenizer

import random
import fire
import tqdm
import gzip
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator

# helpers

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

def train(
    num_batches: int = int(1e5),
    batch_size: int = 4,
    gradient_accumulate_every: int = 4,
    learning_rate: float = 1e-4,
    validate_every: int = 100,
    generate_every: int = 100,
    seq_len: int = 1024,
    entropy_threshold: float = 2.5,
    accumulate_entropy: bool = False,
    ignore_entropy_below: float = 0.,
    cpu: bool = False
):
    accelerator = Accelerator(cpu=cpu)
    device = accelerator.device

    # instantiate GPT-like decoder model

    model = TransformerWrapper(
        num_tokens = 256,
        max_seq_len = seq_len,
        attn_layers = Decoder(
            dim = 512,
            depth = 6,
            heads = 8,
            rotary_pos_emb = True
        )
    )

    tokenizer = EntropyBasedTokenizer(
        model,
        entropy_threshold = entropy_threshold,
        accumulate_entropy = accumulate_entropy,
        ignore_entropy_below = ignore_entropy_below,
        max_token_size = 4
    )

    model = AutoregressiveWrapper(model)

    # prepare enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x), torch.from_numpy(valid_x)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)
    train_loader  = DataLoader(train_dataset, batch_size = batch_size, drop_last = True)
    val_loader    = DataLoader(val_dataset, batch_size = batch_size, drop_last = True)

    # optimizer

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # prepare with accelerate

    model, opt, train_loader, val_loader = accelerator.prepare(
        model, opt, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval=10., desc='training'):
        model.train()

        for __ in range(gradient_accumulate_every):
            loss = model(next(train_loader))
            accelerator.backward(loss / gradient_accumulate_every)

        print(f'training loss: {loss.item()}')
        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        opt.zero_grad()

        if i % validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')

        if i % generate_every == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1].to(device)

            with torch.no_grad():
                tokens = tokenizer(inp, return_segmented_seq = True)

            delimiter = " \u275A "
            output_str = delimiter.join([decode_tokens(token.tolist()) for token in tokens])

            print(f"{output_str}\n\n")

if __name__ == '__main__':
    fire.Fire(train)
