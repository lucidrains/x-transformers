import torch
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
        dim
    ):
        super().__init__()

        self.embed = nn.Embedding(num_tokens, dim)

        self.lstm = LSTM(dim, batch_first = True)

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def generate(
        self,
        length,
        seed = None,
        condition = None,
        temperature = 1e-4 # he uses a near greedy temperature
    ):
        return length

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
        batch_size,
        model: TransformerWrapper,
        data_generator: SyntheticDataGenerator,
        buffer_size = None,
        seq_len = 512,
    ):
        super().__init__()

        self.model = model
        self.ar_wrapped = AutoregressiveWrapper(model)

        assert model.attn_layers.causal

        self.data_generator = data_generator

        self.register_buffer('synth_data_buffer', torch.randint((buffer_size, seq_len)))

    def forward(
        self
    ):

        synthetic_data = self.data_generator()

        return self.ar_wrapped(synthetic_data)
