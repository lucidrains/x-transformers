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

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 500
GENERATE_LENGTH = 256
SEQ_LEN = 256

VALIDATE_EVERY  = 100
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

# instantiate GPT-like decoder model

model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    use_abs_pos_emb = False,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        dynamic_pos_bias = True,
    )
)

model = AutoregressiveWrapper(model)
model.cuda()

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
        return full_seq.cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))

val_dataset_generate = TextSamplerDataset(data_val, SEQ_LEN)

# validation loaders with different sequence lengths

val_loaders = dict()

for valid_seq_len in VALIDATE_SEQ_LENS:
    val_dataset   = TextSamplerDataset(data_val, valid_seq_len)
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

    val_loaders[valid_seq_len] = val_loader

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader))
        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print(f'training loss: {loss.item()}')

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        print(f'validation losses:\n')

        model.eval()
        with torch.no_grad():
            for valid_seq_len in VALIDATE_SEQ_LENS:
                val_loader = val_loaders[valid_seq_len]

                loss = model(next(val_loader))
                print(f'[{valid_seq_len}]:\t {loss.item()}')

        print('\n')

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset_generate)[:-1]
        prime = decode_tokens(inp)
        print(f'%s \n\n %s', (prime, '*' * 100))

        sample = model.generate(
            prompts = inp,
            seq_len = GENERATE_LENGTH,
            cache_kv = True
        )

        output_str = decode_tokens(sample)
        print(f'{output_str}\n\n')
