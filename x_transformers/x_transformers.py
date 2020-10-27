import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers

def exists(val):
    return val is not None

class XTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
