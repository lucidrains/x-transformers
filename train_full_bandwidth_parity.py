# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "einops",
#     "fire",
#     "torch",
#     "tqdm",
#     "x-transformers",
# ]
# ///

# full bandwidth transformer on parity - Xi Wang et al. https://arxiv.org/abs/2608.08888
# latent feedback passes give a fixed depth transformer the serial computation to track state, which a standard transformer fails to length generalize
# run `uv run train_full_bandwidth_parity.py`

import fire
import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange

from x_transformers import TransformerWrapper, Decoder, default_device
from x_transformers.full_bandwidth import FullBandwidth

# constants

BATCH_SIZE = 128
LEARNING_RATE = 3e-4

TRAIN_MAX_LENGTH = 16
EVAL_LENGTHS = (8, 16, 24, 32, 48, 64)
EVAL_SAMPLES = 512

NUM_STEPS = 6000
TEMPORAL_PARALLEL_PASSES = 4
EVAL_PASSES = (1, 2, 3, 4)

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

def make_model(
    full_bandwidth = False,
    recirc_pairs = None,
    transition = 'glu',
    temporal_parallel_passes = TEMPORAL_PARALLEL_PASSES
):
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
            polar_pos_emb = True,
            shift_tokens = 1,
            depth_scale_residual = True
        )
    )

    if not full_bandwidth:
        return net.to(DEVICE)

    return FullBandwidth(
        net,
        temporal_parallel_passes = temporal_parallel_passes,
        recirc_pairs = recirc_pairs,
        transition = transition
    ).to(DEVICE)

# train

def train(
    model,
    num_steps = NUM_STEPS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    temporal_parallel_passes = TEMPORAL_PARALLEL_PASSES
):
    is_full_bandwidth = isinstance(model, FullBandwidth)
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

    train_length, mastered = 1, 0
    desc = 'full bandwidth' if is_full_bandwidth else 'standard'

    for i in tqdm.tqdm(range(num_steps), mininterval = 10., desc = f'training {desc}'):
        seq, labels = make_batch(train_length, batch_size)

        if is_full_bandwidth:
            out = model(
                seq,
                temporal_parallel_passes = temporal_parallel_passes,
                return_loss = False,
                return_all_pass_logits = True
            )
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
def report(model, passes = EVAL_PASSES):
    is_full_bandwidth = isinstance(model, FullBandwidth)
    eval_passes = passes if is_full_bandwidth else (1,)

    model.eval()

    for length in EVAL_LENGTHS:
        seq, labels = make_batch(length, EVAL_SAMPLES)

        accuracies = []

        for p in eval_passes:
            logits = model(seq, temporal_parallel_passes = p, return_loss = False) if is_full_bandwidth else model(seq)
            pred = logits[:, -1].argmax(dim = -1)
            accuracies.append((pred == labels[:, -1]).float().mean().item() * 100)

        extrap = f'({length / TRAIN_MAX_LENGTH:.1f}x)' if length > TRAIN_MAX_LENGTH else '(in-dist)'
        print(f'  length {length:>3} {extrap:>10} | ' + '  '.join(f'{p} pass: {acc:5.1f}%' for p, acc in zip(eval_passes, accuracies)))

    model.train()

# main

def main(
    num_steps = NUM_STEPS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    temporal_parallel_passes = TEMPORAL_PARALLEL_PASSES,
    recirc_pairs = 'paper',
    transition = 'glu',
    train_standard = True,
    eval_passes = EVAL_PASSES,
    seed = 42
):
    torch.manual_seed(seed)

    if train_standard:
        print('\nstandard transformer')
        standard = train(
            make_model(full_bandwidth = False),
            num_steps = num_steps,
            batch_size = batch_size,
            learning_rate = learning_rate
        )
        report(standard)

    print(f'\nfull bandwidth transformer (recirc: {recirc_pairs}, transition: {transition})')
    full_bandwidth = train(
        make_model(
            full_bandwidth = True,
            recirc_pairs = recirc_pairs,
            transition = transition,
            temporal_parallel_passes = temporal_parallel_passes
        ),
        num_steps = num_steps,
        batch_size = batch_size,
        learning_rate = learning_rate,
        temporal_parallel_passes = temporal_parallel_passes
    )
    report(full_bandwidth, passes = eval_passes)

    # generation, latent feedback fusion ablated with should_fuse_latent

    print('\ngeneration')
    prompt = torch.randint(0, 2, (1, 8), device = DEVICE)

    for fuse_latent in (True, False):
        sample = full_bandwidth.generate(
            prompts = prompt,
            seq_len = GENERATE_LENGTH,
            temperature = 0.,
            should_fuse_latent = (lambda step: True) if fuse_latent else (lambda step: False)
        )
        print(f'  fuse latent = {str(fuse_latent):<5}: {sample[0].tolist()}')

if __name__ == '__main__':
    fire.Fire(main)
