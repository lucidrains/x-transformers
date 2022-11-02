import torch
from torch import nn
import torch.nn.functional as F

def exists(val):
    return val is not None

class ContinuousAutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        if num_dims == 2:
            start_tokens = start_tokens[None, :]        

        b, t, _, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            last = self.net(x, **kwargs)[:, -1:]
            out = torch.cat((out, last), dim = -2)

        out = out[:, t:]

        if num_dims == 2:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        inp, target = x[:, :-1], x[:, 1:]

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(inp, **kwargs)
        loss = F.mse_loss(out, target, reduction = 'none')

        if exists(mask):
            loss = loss[mask]

        return loss.mean()
