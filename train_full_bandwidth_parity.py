# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "torch",
#     "tqdm",
#     "x-transformers",
# ]
# ///

# full bandwidth transformer on parity - Xi Wang et al. https://arxiv.org/abs/2608.08888
# latent feedback passes give a fixed depth transformer the serial computation to track state, which a standard transformer fails to length generalize
# run `uv run train_full_bandwidth_parity.py`

import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange

from x_transformers import TransformerWrapper, Decoder, FullBandwidth, default_device

# constants

BATCH_SIZE = 128
LEARNING_RATE = 3e-4

TRAIN_MAX_LENGTH = 32
EVAL_LENGTHS = (8, 16, 32, 64)
EVAL_SAMPLES = 512

NUM_STEPS = 6000
TEMPORAL_PARALLEL_PASSES = 3
EVAL_PASSES = (1, 2, 3)

MASTERY_LOSS_THRESHOLD = 0.01
MASTERY_STEPS = 10

GENERATE_LENGTH = 32

DEVICE = default_device()

# parity data

def make_batch(length, batch_size = BATCH_SIZE):
    seq = torch.randint(0, 2, (batch_size, length), device = DEVICE)
    labels = seq.cumsum(dim = -1) % 2
    return seq, labels

# model

def make_model(full_bandwidth = False):
    net = TransformerWrapper(
        num_tokens = 2,
        max_seq_len = 0,
        tie_embedding = True,
        post_emb_norm = True,
        attn_layers = Decoder(
            dim = 64,
            depth = 3,
            heads = 4,
            attn_dim_head = 32,
            shift_tokens = 1,
            depth_scale_residual = True
        )
    )

    if not full_bandwidth:
        return net.to(DEVICE)

    return FullBandwidth(net, temporal_parallel_passes = TEMPORAL_PARALLEL_PASSES).to(DEVICE)

# train

def train(model):
    is_full_bandwidth = isinstance(model, FullBandwidth)
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_length, mastered = 1, 0
    desc = 'full bandwidth' if is_full_bandwidth else 'standard'

    for i in tqdm.tqdm(range(NUM_STEPS), mininterval = 10., desc = f'training {desc}'):
        seq, labels = make_batch(train_length)

        if is_full_bandwidth:
            out = model(seq, temporal_parallel_passes = TEMPORAL_PARALLEL_PASSES, return_loss = False, return_all_pass_logits = True)
            losses = [F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels, reduction = 'none') for logits in out.all_pass_logits]
            last_loss = losses[-1][:, -1].mean()
            loss = sum(l.mean() for l in losses) / len(losses)
        else:
            logits = model(seq)
            losses = F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels, reduction = 'none')
            last_loss = losses[:, -1].mean()
            loss = losses.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        # length curriculum, grow once running parity at the last position is solved

        mastered = mastered + 1 if last_loss.item() < MASTERY_LOSS_THRESHOLD else 0

        if mastered >= MASTERY_STEPS and train_length < TRAIN_MAX_LENGTH:
            train_length += 1
            mastered = 0

        if (i + 1) % 1000 == 0:
            print(f'step {i + 1} | loss {loss.item():.4f} | train length {train_length}')

    return model

# eval

@torch.no_grad()
def report(model):
    is_full_bandwidth = isinstance(model, FullBandwidth)
    passes = EVAL_PASSES if is_full_bandwidth else (1,)

    model.eval()

    for length in EVAL_LENGTHS:
        seq, labels = make_batch(length, EVAL_SAMPLES)

        accuracies = []

        for p in passes:
            logits = model(seq, temporal_parallel_passes = p, return_loss = False) if is_full_bandwidth else model(seq)
            pred = logits[:, -1].argmax(dim = -1)
            accuracies.append((pred == labels[:, -1]).float().mean().item() * 100)

        print(f'  length {length:>3} | ' + '  '.join(f'{p} pass: {acc:5.1f}%' for p, acc in zip(passes, accuracies)))

    model.train()

# main

if __name__ == '__main__':
    print('\nstandard transformer')
    standard = train(make_model(full_bandwidth = False))
    report(standard)

    print('\nfull bandwidth transformer')
    full_bandwidth = train(make_model(full_bandwidth = True))
    report(full_bandwidth)

    # generation, latent feedback fusion ablated with should_fuse_latent

    print('\ngeneration')
    prompt = torch.randint(0, 2, (1, 8), device = DEVICE)

    for fuse_latent in (True, False):
        sample = full_bandwidth.generate(
            prompts = prompt,
            seq_len = GENERATE_LENGTH,
            should_fuse_latent = (lambda step: True) if fuse_latent else (lambda step: False)
        )
        print(f'  fuse latent = {fuse_latent}: {sample[0].tolist()}')
