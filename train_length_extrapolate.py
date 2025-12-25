# /// script
# dependencies = [
#   "accelerate",
#   "tqdm",
#   "x-transformers>=2.12.0",
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

from accelerate import Accelerator

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 500
GENERATE_LENGTH = 256
SEQ_LEN = 256

VALIDATE_EVERY  = 250
VALIDATE_SEQ_LENS = (256, 512, 1024, 2048, 4096)

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

# accelerator

accelerator = Accelerator()

# instantiate GPT-like decoder model

model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    use_abs_pos_emb = False,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        polar_pos_emb = True,
        rotary_pos_emb = False,
        dynamic_pos_bias = False
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
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
train_loader  = DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True)

val_dataset_generate = TextSamplerDataset(data_val, SEQ_LEN)

# optimizer

optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# prepare

model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

train_loader = cycle(train_loader)

# validation loaders with different sequence lengths

val_loaders = dict()

for valid_seq_len in VALIDATE_SEQ_LENS:
    val_dataset   = TextSamplerDataset(data_val, valid_seq_len)
    val_loader    = DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True)
    val_loader    = cycle(val_loader)

    val_loaders[valid_seq_len] = val_loader

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        data = next(train_loader)
        loss = model(data)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 0.5)

        optim.step()
        optim.zero_grad()

    if i % 10 == 0:
        accelerator.print(f'training loss: {loss.item()}')

    if i % VALIDATE_EVERY == 0:
        accelerator.print(f'validation losses:\n')

        model.eval()
        with torch.inference_mode():
            for valid_seq_len in VALIDATE_SEQ_LENS:
                val_loader = val_loaders[valid_seq_len]

                val_data = next(val_loader).to(accelerator.device)
                loss = model(val_data)
                accelerator.print(f'[{valid_seq_len}]:\t {loss.item()}')

        accelerator.print('\n')

    if i % GENERATE_EVERY == 0:
        model.eval()
        unwrapped_model = accelerator.unwrap_model(model)
        
        inp = random.choice(val_dataset_generate)[:-1]
        inp = inp.to(accelerator.device)
        prime = decode_tokens(inp)
        accelerator.print(f'{prime} \n\n {"*" * 100}')

        sample = unwrapped_model.generate(
            prompts = inp,
            seq_len = GENERATE_LENGTH,
            cache_kv = True
        )

        output_str = decode_tokens(sample)
        accelerator.print(f'{output_str}\n\n')
