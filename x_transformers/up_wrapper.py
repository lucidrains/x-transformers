# https://arxiv.org/abs/2506.20057
# Peter Bloem

from __future__ import annotations
from functools import partial
from random import randrange, uniform

import torch
from torch import nn, cat, tensor, randperm
from torch.nn import LSTM, GRU, Module

from x_transformers.x_transformers import (
    TransformerWrapper,
    AutoregressiveWrapper
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# random sequences, mixture of random and constant (unsure why constant is needed)

def random_sequences(
    num_tokens,
    seq_len,
    num_samples_random,
    num_samples_constant,
    shuffle = True,
    device = None
):
    assert num_samples_random > 0 or num_samples_constant > 0

    rand_seq = torch.randint(0, num_tokens, (num_samples_random, seq_len))
    const_seq = torch.full((num_samples_constant, seq_len), randrange(num_tokens))

    all_seq = cat((rand_seq, const_seq))

    if exists(device):
        all_seq = all_seq.to(device)

    if not shuffle:
        return all_seq

    # shuffle with randperm

    rand_indices = randperm(all_seq.shape[0], device = all_seq.device)
    return all_seq[rand_indices]

# synthetic data generator

class SyntheticDataGenerator(Module):
    def __init__(
        self,
        dim,
        num_tokens,
        max_seq_len = 512,
        hidden_size = None,
        use_gru = False,
        network_klass = None
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(num_tokens, dim)

        hidden_size = default(hidden_size, dim)

        default_network_klass = partial(LSTM if not use_gru else GRU, batch_first = True)
        network_klass = default(network_klass, default_network_klass)

        self.net = network_klass(dim, hidden_size)

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

        self.apply(self.init_)

    def reset_(self):
        for m in self.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        self.apply(self.init_)

    @torch.no_grad()
    def init_(self, m):
        if isinstance(m, nn.Linear):
            m.weight *= uniform(0., 1.1) # he scales the lstm weights from 0 to 1.1

    @torch.inference_mode()
    @torch.compile
    def generate(
        self,
        length,
        seed = None,
        condition = None,
        temperature = 1e-4 # he uses a near greedy temperature
    ):
        assert exists(seed) or exists(condition)
        prefix = [*filter(exists, (seed, condition))]
        seq_len = self.max_seq_len

        seq = torch.cat(prefix, dim = -1)

        net_input = seq
        hiddens = None

        for _ in range(length):

            logits, hiddens = self.forward(net_input, hiddens)

            last_logit = logits[:, -1]
            prob = (last_logit / temperature).softmax(dim = -1)

            sampled = torch.multinomial(prob, 1)
            net_input = sampled

            seq = torch.cat((seq, sampled), dim = -1)

        return seq[:, -seq_len:]

    def forward(
        self,
        input,
        hiddens = None
    ):

        tokens = self.embed(input)

        embed, hidden = self.net(tokens, hiddens)

        logits = self.to_logits(embed)

        return logits, hidden

# classes

class UniversalPretrainWrapper(Module):
    def __init__(
        self,
        model: TransformerWrapper,
        data_generator: SyntheticDataGenerator | Module | None = None,
        buffer_size = None,
        num_reset = 20,
        batch_size = 32,
        seq_len = 512,
        seed_length = 8,
        reset_turing_machine_every = 0,
        keep_buffer_on_cpu = False
    ):
        super().__init__()

        self.model = model
        self.ar_wrapped = AutoregressiveWrapper(model)

        assert model.attn_layers.causal

        num_tokens = model.num_tokens
        dim = model.attn_layers.dim

        if not exists(data_generator):
            data_generator = SyntheticDataGenerator(
                num_tokens = num_tokens,
                dim = dim,
                max_seq_len = seq_len
            )

        self.reset_turing_machine_every = reset_turing_machine_every

        self.seq_len = seq_len
        self.data_generator = data_generator

        self.seed_length = seed_length
        self.batch_size = batch_size

        buffer_size = default(buffer_size, batch_size * 20)
        assert buffer_size > batch_size, f'data buffer size must be greater than batch size'

        assert divisible_by(num_reset, 2)
        self.num_reset = num_reset

        self.buffer_size = buffer_size

        self.random_sequences_fn = partial(random_sequences, num_tokens, seq_len)

        init_data_buffer = self.random_sequences_fn(buffer_size // 2, buffer_size // 2)

        if keep_buffer_on_cpu:
            self.synth_data_buffer = init_data_buffer
        else:
            self.register_buffer('synth_data_buffer', init_data_buffer)

        self.register_buffer('step', tensor(0))

    @property
    def device(self):
        return self.step.device

    def get_rand_sequences_from_buffer(self, size = None):
        size = default(size, self.batch_size)
        rand_indices = randperm(self.buffer_size, device = self.device)[:size]
        return self.synth_data_buffer[rand_indices]

    def forward(self):
        # following algorithm 1.

        conditions = self.get_rand_sequences_from_buffer()

        # get seeds, which appears to be random sequences with random crops of seed length

        seeds = self.get_rand_sequences_from_buffer()

        seq_arange = torch.arange(self.seed_length)
        rand_offset = torch.randint(0, self.seq_len - self.seed_length, (self.batch_size,))
        seq_start_pos = rand_offset[:, None] + seq_arange

        batch_arange = torch.arange(self.batch_size, device = self.device)[:, None]
        seeds = seeds[batch_arange, seq_start_pos]

        # seed, condition to turing machine

        generated = self.data_generator.generate(
            self.seq_len,
            condition = conditions.to(self.device),
            seed = seeds.to(self.device)
        )

        self.step.add_(1)

        # maybe reset turing machine

        if self.reset_turing_machine_every > 0 and divisible_by(self.step.item(), self.reset_turing_machine_every):
            self.data_generator.reset_()

        # reset

        if self.num_reset > 0:
            buffer_to_reset = self.get_rand_sequences_from_buffer(self.num_reset)

            with torch.no_grad():
                reset_sequences = self.random_sequences_fn(self.num_reset // 2, self.num_reset // 2, device = self.device)
                buffer_to_reset.copy_(reset_sequences)

        # place "enriched" random generated sequences back

        with torch.no_grad():
            conditions.copy_(generated)

        # sample yet again according to pseudocode

        data = self.get_rand_sequences_from_buffer().to(self.device)

        return self.ar_wrapped(data)
