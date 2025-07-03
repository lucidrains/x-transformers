import torch
from torch import nn
from torch.nn import LSTM, Module

from x_transformers.x_transformers import (
    TransformerWrapper,
    AutoregressiveWrapper
)

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# synthetic data generator

class SyntheticDataGenerator(Module):
    def __init__(
        self,
        dim,
        num_tokens,
        max_seq_len = 512,
        hidden_size = None
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.embed = nn.Embedding(num_tokens, dim)

        hidden_size = default(hidden_size, dim)
        self.lstm = LSTM(dim, hidden_size, batch_first = True)

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

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

        embed, hidden = self.lstm(tokens, hiddens)

        logits = self.to_logits(embed)

        return logits, hidden

# classes

class UniversalPretrainWrapper(Module):
    def __init__(
        self,
        model: TransformerWrapper,
        data_generator: SyntheticDataGenerator | None = None,
        buffer_size = None,
        batch_size = 32,
        seq_len = 512,
        seed_length = 8
    ):
        super().__init__()

        self.model = model
        self.ar_wrapped = AutoregressiveWrapper(model)

        assert model.attn_layers.causal

        if not exists(data_generator):
            data_generator = SyntheticDataGenerator(
                num_tokens = model.num_tokens,
                dim = model.attn_layers.dim
            )

        self.seq_len = seq_len
        self.data_generator = data_generator

        self.seed_length = seed_length
        self.batch_size = batch_size
        buffer_size = default(buffer_size, batch_size * 20)
        assert buffer_size > batch_size, f'data buffer size must be greater than batch size'

        init_data_buffer = torch.randint(0, model.num_tokens, (buffer_size, seq_len))
        self.register_buffer('synth_data_buffer', init_data_buffer)

    @property
    def device(self):
        return self.synth_data_buffer.device

    def forward(
        self
    ):

        randperm = torch.randperm(self.batch_size, device = self.device)

        seeds = self.synth_data_buffer[randperm][:self.seed_length]

        synthetic_data = self.data_generator.generate(self.seq_len, seed = seeds)

        return self.ar_wrapped(synthetic_data)
