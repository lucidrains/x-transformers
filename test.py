import torch
from x_transformers import TransformerWrapper, Decoder

model = TransformerWrapper(
    num_tokens = 20000,
    max_seq_len = 512,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8,
        rotary_pos_emb = True
    )
).cuda()

x = torch.randint(0, 256, (1, 512)).cuda()

model(x) # (1, 1024, 20000)
