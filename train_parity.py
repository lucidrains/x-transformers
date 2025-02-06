import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F

from x_transformers import TransformerWrapper, Decoder

# constants

BATCH_SIZE = 256
LEARNING_RATE = 3e-4
EVAL_EVERY  = 500

EVAL_LENGTHS = (16, 32, 64, 128, 256, 512)
TRAIN_MAX_LENGTH = EVAL_LENGTHS[-2]

LOSS_THRES_INCREASE_LEN = 1e-3
MEET_CRITERIA_THRES_INCREASE_LEN = 10

HYBRIDIZE_WITH_RNN = True

# rnn for fully resolving state tracking by hybridization
# but will also look into gated delta net + negative eigenvalues (Songlin Yang et al) as a parallel solution

dim = 64
heads = 4
dim_head = 32
decoder_kwargs = dict()

if HYBRIDIZE_WITH_RNN:
    from torch.nn import GRU

    decoder_kwargs = dict(
        attn_hybrid_fold_axial_dim = 4, # even if recurrence is every 4 tokens, can generalize for parity
        attn_hybrid_learned_mix = True,
        attn_hybrid_module = GRU(dim, dim_head * heads, batch_first = True)
    )

# instantiate model

model = TransformerWrapper(
    num_tokens = 2,
    max_seq_len = 0,
    attn_layers = Decoder(
        dim = dim,
        depth = 3,
        heads = heads,
        attn_dim_head = dim_head,
        shift_tokens = 1, # helps a lot with parity training, but not able to generalize on its own
        **decoder_kwargs
    )
).cuda()

# optimizer

from lion_pytorch.cautious_lion import Lion

optimizer = Lion(model.parameters(), lr = LEARNING_RATE, cautious_factor = 0.1)

# data generator

def cycle(length):
    while True:
        seq = torch.randint(0, 2, (BATCH_SIZE, length)).cuda()
        labels = (seq.cumsum(dim = -1) % 2)
        yield (seq, labels)

# dataloaders

train_dl = cycle(TRAIN_MAX_LENGTH)

eval_dls = {eval_length: cycle(eval_length) for eval_length in EVAL_LENGTHS}

print(f'training at max length: {TRAIN_MAX_LENGTH}')

# training

i = 0
meet_criteria = 0
train_seq_len = 1
stop_length = EVAL_LENGTHS[-2]

with tqdm.tqdm(mininterval = 10., desc = 'training') as pbar:

    while train_seq_len < stop_length:
        model.train()

        seq, labels = next(train_dl)

        # length curriculum learning

        seq = seq[:, :train_seq_len]
        labels = labels[:, :train_seq_len]

        logits = model(seq)

        loss = F.cross_entropy(logits.transpose(-1, -2), labels, reduction = 'none')
        last_loss = loss[:, -1].mean()
        loss.mean().backward()

        if last_loss.item() < LOSS_THRES_INCREASE_LEN:
            meet_criteria += 1
        else:
            meet_criteria = 0

        if meet_criteria >= MEET_CRITERIA_THRES_INCREASE_LEN:
            meet_criteria = 0
            train_seq_len += 1
            print(f'criteria met, incrementing to {train_seq_len}')

        print(f'({train_seq_len})| {i}: {last_loss.item()}')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        last_step = train_seq_len == stop_length

        if last_step:
            print(f'made it to training length {train_seq_len}. running final eval to check for generalization')

        if last_step or (i + 1) % EVAL_EVERY == 0:

            model.eval()
            print('\n')

            for eval_length, eval_dl in eval_dls.items():
                incorrects = 0

                seq, labels = next(eval_dl)

                logits = model(seq)
                pred = logits[:, -1].argmax(dim = -1)
                incorrects = (pred != labels[:, -1]).abs().sum().item()

                frac_incorrect = incorrects * 100 / BATCH_SIZE

                print(f"{eval_length}\t - frac incorrect:\t {frac_incorrect:.1f}%")

            print('\n')

        i += 1
        pbar.update(1)
