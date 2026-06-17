import gzip
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import fire
import wandb
import tqdm
from accelerate import Accelerator
from x_transformers import AutoregressiveWrapper, TransformerWrapper, Decoder

from x_transformers.gpt_lejepa import LatentAutoregressive

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def cycle(loader):
    while True:
        for data in loader:
            yield data

# main

def train(
    num_batches = int(1e5),
    batch_size = 4,
    gradient_accumulate_every = 4,
    learning_rate = 1e-4,
    validate_every = 100,
    generate_every = 500,
    generate_length = None,
    seq_len = 128,
    track_experiment_online = False,
    run_name = 'gpt-lejepa',
    cpu = False,
    sigreg_loss_weight = 0.05,
    l2_loss_weight = 1.,
    frac_gradient = 0.,
    predictor_input_hiddens_index = -1,
    predict_next_embed_with_action = True,
    predict_next_embed_no_action = True,
    detach_target = False,
    num_rollouts = 1,
    rollout_loss_weights = None
):
    accelerator = Accelerator(cpu = cpu)
    device = accelerator.device

    generate_length = generate_length if exists(generate_length) else seq_len

    # instantiate gpt-lejepa

    model = LatentAutoregressive(
        TransformerWrapper(
            num_tokens = 256,
            max_seq_len = seq_len,
            attn_layers = Decoder(
                dim = 512,
                depth = 8,
                heads = 8,
                rotary_pos_emb = True,
                pre_norm_has_final_norm = False
            )
        ),
        dim = 512,
        sigreg_loss_weight = sigreg_loss_weight,
        l2_loss_weight = l2_loss_weight,
        frac_gradient = frac_gradient,
        predictor_input_hiddens_index = predictor_input_hiddens_index,
        predict_next_embed_with_action = predict_next_embed_with_action,
        predict_next_embed_no_action = predict_next_embed_no_action,
        detach_target = detach_target,
        num_rollouts = num_rollouts,
        rollout_loss_weights = rollout_loss_weights
    )

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
            return full_seq.to(device)

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset   = TextSamplerDataset(data_val, seq_len)

    train_loader  = cycle(DataLoader(train_dataset, batch_size = batch_size, drop_last = True))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = batch_size, drop_last = True))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # wandb

    wandb.init(project = 'enwik8-lejepa', mode = 'online' if track_experiment_online else 'disabled')
    wandb.run.name = run_name

    # accelerate

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    # training

    for i in tqdm.tqdm(range(num_batches), mininterval = 10., desc = 'training'):
        model.train()

        for _ in range(gradient_accumulate_every):
            loss, (ce_loss, l2_loss, l2_no_action_loss, sreg_loss) = model(next(train_loader), return_loss_breakdown = True)

        if exists(gradient_accumulate_every):
            accelerator.backward(loss / gradient_accumulate_every)

        print(f'training loss: {loss.item():.4f} | ce: {ce_loss.item():.4f} | l2: {l2_loss.item():.4f} | l2 (no action): {l2_no_action_loss.item():.4f} | sigreg: {sreg_loss.item():.4f}')

        if accelerator.is_main_process:
            wandb.log(dict(
                loss = loss.item(),
                ce_loss = ce_loss.item(),
                l2_loss = l2_loss.item(),
                l2_no_action_loss = l2_no_action_loss.item(),
                sigreg_loss = sreg_loss.item()
            ))

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                loss, (ce_loss, l2_loss, l2_no_action_loss, sreg_loss) = model(next(val_loader), return_loss_breakdown = True)
                print(f'validation loss: {loss.item():.4f} | ce: {ce_loss.item():.4f} | l2: {l2_loss.item():.4f} | l2 (no action): {l2_no_action_loss.item():.4f} | sigreg: {sreg_loss.item():.4f}')

                if accelerator.is_main_process:
                    wandb.log(dict(
                        valid_loss = loss.item(),
                        valid_ce_loss = ce_loss.item(),
                        valid_l2_loss = l2_loss.item(),
                        valid_l2_no_action_loss = l2_no_action_loss.item(),
                        valid_sigreg_loss = sreg_loss.item()
                    ))

        if divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp.cpu().numpy())
            print(f'%s \n\n %s' % (prime, '*' * 100))

            generator = AutoregressiveWrapper(accelerator.unwrap_model(model).net)
            sample = generator.generate(
                prompts = inp,
                seq_len = generate_length,
                cache_kv = True
            )

            output_str = decode_tokens(sample.cpu().numpy())
            print(output_str)

if __name__ == '__main__':
    fire.Fire(train)
