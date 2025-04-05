import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from x_transformers.x_transformers import Decoder, TransformerWrapper

from einops import repeat, rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# entropy based tokenizer applied in byte-latent transformer paper
# they use a simple entropy threshold for segmenting a string into variable sized tokens

# https://arxiv.org/abs/2412.09871

class EntropyBasedTokenizer(Module):
    def __init__(
        self,
        decoder: TransformerWrapper,
        entropy_threshold: float
    ):
        super().__init__()
        assert isinstance(decoder.attn_layers, Decoder)

        self.decoder = decoder
        self.entropy_threshold = entropy_threshold

    @torch.no_grad()
    def forward(
        self,
        seq,
        return_segmented_seq = False
    ):
        self.decoder.eval()

        batch, seq_len, device = *seq.shape, seq.device

        _, intermediates = self.decoder(seq, return_logit_entropies = True)

        entropies = intermediates.logit_entropies

        over_thres_mask = entropies >= self.entropy_threshold

        arange = torch.arange(seq_len, device = device) + 1
        arange = repeat(arange, 'n -> b n', b = batch)

        # get a tensor of Int['b num_tokens'] with the token lengths, zero padded

        boundaries = over_thres_mask.clone()
        boundaries[..., -1] = True # last token is always a boundary

        num_tokens = boundaries.sum(dim = -1) # number of tokens

        boundaries = arange[boundaries].split(num_tokens.tolist())

        # get the token lengths

        token_lengths = []

        for one_boundary in boundaries:
            padded_boundary = F.pad(one_boundary, (1, 0), value = 0.)
            one_token_lengths = padded_boundary[1:] - padded_boundary[:-1]

            token_lengths.append(one_token_lengths)

        token_lengths = pad_sequence(token_lengths, batch_first = True)

        # early return

        if not return_segmented_seq:
            return token_lengths

        # segment the sequence based on the token lengths

        segmented_seq = []

        for one_seq, one_token_length in zip(seq, token_lengths):

            one_token_length = one_token_length[one_token_length > 0]

            splitted_seq = one_seq.split(one_token_length.tolist())
            segmented_seq.append(splitted_seq)

        return segmented_seq
