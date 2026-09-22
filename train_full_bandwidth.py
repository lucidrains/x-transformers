import gzip
import random
import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from x_transformers import TransformerWrapper, Decoder, FullBandwidth, default_device

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
GENERATE_EVERY = 500
GENERATE_LENGTH = 128
SEQ_LEN = 256

DEVICE = default_device()

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, int(token))))

def decode_tokens(tokens):
    if torch.is_tensor(tokens):
        tokens = tokens.tolist()
    return ''.join(list(map(decode_token, tokens)))

# instantiate full bandwidth transformer

model = TransformerWrapper(
    num_tokens = 256,
    max_seq_len = SEQ_LEN,
    tie_embedding = True,
    post_emb_norm = True,
    attn_layers = Decoder(
        dim = 512,
        depth = 6,
        heads = 8,
        depth_scale_residual = True
    )
)

model = FullBandwidth(
    model,
    temporal_parallel_passes = 2,
    dynamic_rollout_loss_weight = True
).to(DEVICE)

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
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
        return full_seq.to(DEVICE)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE, drop_last = True))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE, drop_last = True))

# optimizer

optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

if __name__ == '__main__':
    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10., desc = 'training'):
        model.train()

        for _ in range(GRADIENT_ACCUMULATE_EVERY):
            passes = random.choices([1, 2, 3], weights = [0.75, 0.22, 0.03])[0]
            loss = model(next(train_loader), temporal_parallel_passes = passes)
            (loss / GRADIENT_ACCUMULATE_EVERY).backward()

        print(f'training loss: {loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:100]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s' % (prime, '*' * 100))

            sample = model.generate(
                prompts = inp,
                seq_len = GENERATE_LENGTH
            )

            output_str = decode_tokens(sample)
            print(output_str)
