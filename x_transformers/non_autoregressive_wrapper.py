import torch
from torch import nn
import torch.nn.functional as F

def exists(val):
    return val is not None

# non-autoregressive training logic

def uniform_training(t):
    # work in progress
    num_to_mask = None
    inp = None
    return inp

def random_training(t, mask_index):
    to_mask = torch.randint(0,2, t.size())
    inp = torch.where(to_mask==0, mask_index, t)
    return inp

# Non-autoregressive wrapper class

class NonAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        mask_index, 
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.
    ):
        super().__init__()
        self.mask_index = mask_index
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len
        
        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        iteration=1,
        **kwargs
    ):
        device = start_tokens.device

        was_training = self.net.training 
    
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        
        out = start_tokens

        while iteration > 0:

            logits = self.net(out, **kwargs)
            probs = F.softmax(logits, dim=-1)
            token_probs, tokens = probs.max(dim=-1)
            out = tokens

            iteration -= 1

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)

        return out

    def forward(self, x, training_logic = None, **kwargs):
    
        seq, ignore_index = x.shape[1], self.ignore_index       

        if training_logic is random_training:
            inp = random_training(x)
        elif training_logic is uniform_training:
            # work in progress
            pass
        else:
            inp = torch.full_like(x, self.mask_index).long().to(x.device)
            
        target = x[:,:]
        
        out = self.net(inp, **kwargs)    

        out = out.transpose(1, 2) 

        loss = F.cross_entropy(
            out,            
            target,
            ignore_index = ignore_index
        )

        return loss
