# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch>=2.0",
#     "fire>=0.7",
#     "einx>=0.3.0",
#     "einops>=0.8.0",
#     "torch-einops-utils>=0.1.12",
#     "loguru",
#     "packaging>=21.0",
# ]
# ///

import fire
import torch

import einx

from x_transformers.x_transformers import TransformerWrapper, Decoder
from x_transformers.discover_wrapper import (
    DiscoverDecoder,
    TPREncoder,
    ROLE_SCHEMES,
    role_ids_for_tokens,
    num_roles_for_scheme
)

# train a bottleneck transformer to reverse lists of letters, then probe its encodings
# for the role-filler structure, and do constituent surgery - DISCOVER

# McCoy et al. https://arxiv.org/abs/2608.29530

def exists(v):
    return v is not None

def to_letters(tokens, num_fillers = 26):
    return ''.join(chr(65 + v) for v in tokens if v < num_fillers)

def lists(n, lengths, device, num_fillers):
    max_len = int(lengths.max())
    tokens = torch.randint(0, num_fillers, (n, max_len), device = device)
    valid = einx.less('n, b -> b n', torch.arange(max_len, device = device), lengths)
    return tokens.masked_fill(~valid, num_fillers + 1)

def reversal_targets(tokens, lengths, device, num_fillers):
    batch, seq_len = tokens.shape
    pad_id = num_fillers + 1
    positions = torch.arange(seq_len, device = device)
    valid = einx.less('n, b -> b n', positions, lengths)
    indices = einx.subtract('b, n -> b n', lengths - 1, positions).clamp(min = 0)
    targets = tokens[torch.arange(batch, device = device)[:, None], indices]
    return targets.masked_fill(~valid, pad_id)

def main(
    *,
    num_fillers: int = 26,
    len_max: int = 3,
    batch_size: int = 128,
    train_steps: int = 12000,
    dim: int = 192,
    depth: int = 4,
    heads: int = 6,
    fit_lists: int = 9600,
    eval_lists: int = 600,
    surgeries: int = 256,
    tpr_steps: int = 1500,
    tpr_mlp: int = 0,
    seed: int = 0,
    lr: float = 3e-4
):
    bos_id = num_fillers
    pad_id = num_fillers + 1
    num_tokens = num_fillers + 2

    device = torch.device(
        'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    torch.manual_seed(seed)

    decoder = TransformerWrapper(
        num_tokens = num_tokens,
        max_seq_len = 16,
        emb_dropout = 0.05,
        attn_layers = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dropout = 0.05,
            ff_dropout = 0.05,
            ff_mult = 4
        )
    ).to(device)

    model = DiscoverDecoder(
        decoder = decoder,
        num_fillers = num_fillers,
        bos_id = bos_id,
        pad_id = pad_id,
        max_seq_len = len_max
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    def evaluate():
        model.eval()
        accs = {}

        for length in range(1, len_max + 1):
            tokens = torch.randint(0, num_fillers, (256, length), device = device)
            lengths = torch.full((256,), length, device = device)
            outputs = model.decode(model.encode(tokens), lengths)
            accs[length] = (outputs[:, :length] == tokens.flip(-1)).all(dim = 1).float().mean().item()

        return accs

    print(f'training bottleneck decoder on reversal (lists of length 1..{len_max}, letters A..Z) [{device}]')

    best_acc = -1.
    best_state = None

    for i in range(train_steps):
        model.train()

        length = torch.randint(1, len_max + 1, (1,)).item()
        src = torch.randint(0, num_fillers, (batch_size, length), device = device)
        tgt = src.flip(-1)

        logits = model(src, tgt)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, num_tokens), tgt.reshape(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 1000 == 0:
            accs = evaluate()

            if sum(accs.values()) > best_acc:
                best_acc = sum(accs.values())
                best_state = {name: param.detach().clone() for name, param in model.named_parameters()}

            print(f'step {i + 1}: loss {loss.item():.3f} | accuracy by length ' + ' '.join(f'len {l}: {a:.3f}' for l, a in accs.items()), flush = True)

    if exists(best_state):
        model.load_state_dict(best_state)

    accs = evaluate()
    print('final accuracy by length: ' + ' '.join(f'len {l}: {a:.3f}' for l, a in accs.items()))
    print()

    num_lists = fit_lists + eval_lists
    lengths = torch.randint(1, len_max + 1, (num_lists,), device = device)
    tokens = lists(num_lists, lengths, device, num_fillers)
    targets = reversal_targets(tokens, lengths, device, num_fillers)

    print(f'DISCOVER: fitting a TPR per role scheme ({fit_lists} fit lists, {eval_lists} held-out)')

    mask = einx.less('n, b -> b n', torch.arange(len_max, device = device), lengths)
    encodings = model.encode(tokens, mask)

    generator = torch.Generator(device = device).manual_seed(seed)
    permutation = torch.randperm(num_lists, generator = generator, device = device)
    fit_idx, eval_idx = permutation[:fit_lists], permutation[fit_lists:fit_lists + eval_lists]

    print()
    print(f'TPR readout: {"mlp " + str(tpr_mlp) if tpr_mlp > 0 else "linear"}')
    print(f'{"role scheme":<16}{"approximation accuracy":>26}{"fit mse":>12}')
    print('-' * 54)

    results = {}

    for scheme in ROLE_SCHEMES:
        role_ids = role_ids_for_tokens(
            tokens,
            lengths,
            scheme,
            max_seq_len = len_max,
            num_fillers = num_fillers
        )

        tpr = TPREncoder(
            dim = model.dim,
            num_fillers = num_fillers,
            num_roles = num_roles_for_scheme(
                scheme,
                max_seq_len = len_max,
                num_fillers = num_fillers
            ),
            mlp_dim = tpr_mlp or None
        ).to(device)

        mse = tpr.fit(
            tokens[fit_idx],
            role_ids[fit_idx],
            encodings[fit_idx],
            mask = mask[fit_idx],
            steps = tpr_steps,
            batch_size = 1024
        )

        approx_encodings = tpr(tokens[eval_idx], role_ids[eval_idx], mask[eval_idx])
        outputs = model.decode(approx_encodings, lengths[eval_idx])
        accuracy = (outputs == targets[eval_idx]).all(dim = 1).float().mean().item()

        results[scheme] = dict(tpr = tpr, accuracy = accuracy, mse = mse)

        print(f'{scheme:<16}{accuracy:>26.3f}{mse:>12.4f}', flush = True)

    tpr = results['bidirectional']['tpr']

    lengths = torch.randint(1, len_max + 1, (surgeries,), device = device)
    tokens = lists(surgeries, lengths, device, num_fillers)
    positions = torch.tensor([torch.randint(0, int(l), (1,), device = device).item() for l in lengths], device = device)
    replacements = (einx.get_at('b [n], b -> b', tokens, positions) + torch.randint(1, num_fillers, (surgeries,), device = device)) % num_fillers

    edited_tokens = tokens.clone()
    einx.set_at('b [n], b, b -> b [n]', edited_tokens, positions, replacements)

    role_ids_old = role_ids_for_tokens(tokens, lengths, 'bidirectional', max_seq_len = len_max, num_fillers = num_fillers)
    role_ids_new = role_ids_for_tokens(edited_tokens, lengths, 'bidirectional', max_seq_len = len_max, num_fillers = num_fillers)

    outputs = model.surgery(tokens, edited_tokens, role_ids_old, role_ids_new, tpr)
    accuracy = (outputs == reversal_targets(edited_tokens, lengths, device, num_fillers)).all(dim = 1).float().mean().item()

    print(f'\nconstituent surgery (bidirectional roles): {accuracy:.3f} of {surgeries} edits decoded as if the letter at the edited role had been replaced\n')

    example = torch.tensor([[22, 23, 2]], device = device)  # W X C
    edited_example = torch.tensor([[22, 23, 3]], device = device)  # W X D
    length = torch.tensor([3], device = device)

    def example_decode(encodings):
        return model.decode(encodings, length)[0]

    print('\nworkthrough: substitution on the example list W X C (reversal should give C X W)')
    print(f'  network encoding:      {to_letters(example_decode(model.encode(example, torch.ones(1, len_max, device = device, dtype = torch.bool))), num_fillers)}')

    for scheme in ('bidirectional', 'right_to_left', 'bag_of_words'):
        role_ids = role_ids_for_tokens(example, length, scheme, max_seq_len = len_max, num_fillers = num_fillers)
        tpr_encodings = results[scheme]['tpr'](example, role_ids)
        print(f'  {scheme} substitution: {to_letters(example_decode(tpr_encodings), num_fillers)} (approx accuracy {results[scheme]["accuracy"]:.3f})')

    roles_old = role_ids_for_tokens(example, length, 'bidirectional', max_seq_len = len_max, num_fillers = num_fillers)
    roles_new = role_ids_for_tokens(edited_example, length, 'bidirectional', max_seq_len = len_max, num_fillers = num_fillers)

    print(f'  surgery C -> D:        {to_letters(model.surgery(example, edited_example, roles_old, roles_new, tpr)[0], num_fillers)}')

if __name__ == '__main__':
    fire.Fire(main)
