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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from tqdm import tqdm
import fire
import wandb

from x_transformers import TransformerWrapper, Decoder
from x_transformers.xm_induced_latent_decoder import XMInducedLatentDecoder

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

# latent kl metric

@torch.no_grad()
def calculate_latent_kl(
    model,
    accelerator,
    x,
    num_latents,
    num_pairs = 4
):
    unwrapped = accelerator.unwrap_model(model)
    net, latent_proj, latent_dim = unwrapped.net, unwrapped.latent_proj, unwrapped.latent_dim

    batch, seq, device = x.shape[0], x[:, :-1], x.device
    kls = []

    for _ in range(num_pairs):
        z1 = torch.randn(batch, num_latents, latent_dim, device = device)
        z2 = torch.randn(batch, num_latents, latent_dim, device = device)

        logits1 = net(seq, prepend_embeds = latent_proj(z1), excise_prepend_embeds = True)
        logits2 = net(seq, prepend_embeds = latent_proj(z2), excise_prepend_embeds = True)

        kl = F.kl_div(
            F.log_softmax(logits2, dim = -1),
            F.log_softmax(logits1, dim = -1),
            log_target = True,
            reduction = 'none'
        ).sum(dim = -1).mean()

        kls.append(kl)

    return torch.stack(kls).mean()

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,)).item()
        return self.data[rand_start: rand_start + self.seq_len + 1].clone().long()

    def __len__(self):
        return self.data.size(0) // self.seq_len

# train

def train(
    candidates = 2,
    num_latents = 4,
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    max_grad_norm = 0.5,
    validate_every = 100,
    generate_every = 500,
    generate_length = None,
    seq_len = 1024,
    track_experiment_online = False,
    project_name = 'enwik8-xm',
    run_name = None,
    cpu = False
):
    accelerator = Accelerator(cpu = cpu)

    run_name = default(run_name, f'xm-k{candidates}-n{num_latents}')
    generate_length = default(generate_length, seq_len)

    # model

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

    model = XMInducedLatentDecoder(
        net = model,
        num_latents = num_latents,
        candidates = candidates
    )

    # prepare enwik8 data

    with gzip.open('./data/enwik8.gz') as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        train_x, valid_x = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(train_x).clone(), torch.from_numpy(valid_x).clone()

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # experiment tracking

    if accelerator.is_main_process:
        wandb.init(project = project_name, name = run_name, mode = 'online' if track_experiment_online else 'disabled')
        if exists(wandb.run) and exists(wandb.run.url):
            print(f"W&B Run URL: {wandb.run.url}")

    model, optim = accelerator.prepare(model, optim)

    # training loop

    for i in tqdm(range(num_batches), mininterval = 10., desc = 'training'):
        model.train()

        for _ in range(gradient_accumulate_every):
            data = next(train_loader).to(accelerator.device)
            loss = model(data)
            accelerator.backward(loss / gradient_accumulate_every)

        print(f'step {i} | training loss: {loss.item():.4f}')
        if accelerator.is_main_process:
            wandb.log(dict(loss = loss.item()), step = i)

        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optim.step()
        optim.zero_grad()

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                val_seq = next(val_loader).to(accelerator.device)
                val_loss = model(val_seq)
                latent_kl = calculate_latent_kl(model, accelerator, val_seq, num_latents = num_latents)

                print(f'step {i} | validation loss: {val_loss.item():.4f} | latent KL div: {latent_kl.item():.6f}')
                if accelerator.is_main_process:
                    wandb.log(dict(
                        valid_loss = val_loss.item(),
                        latent_kl_div = latent_kl.item()
                    ), step = i)

        if divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:-1].to(accelerator.device)
            prime = decode_tokens(inp.cpu().numpy())
            print(f'{prime} \n\n {"*" * 100}')

            unwrapped_model = accelerator.unwrap_model(model)
            sample = unwrapped_model.generate(
                start_tokens = inp.unsqueeze(0),
                seq_len = generate_length
            )

            output_str = decode_tokens(sample[0].cpu().numpy())
            print(output_str)

if __name__ == '__main__':
    fire.Fire(train)
