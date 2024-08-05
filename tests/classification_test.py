from x_transformers import TransformerWrapper, Encoder
import torch
from torch import nn

# CLS token test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    use_cls=True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10))
y = torch.tensor([0, 1])

print(x.shape)
logits = transformer(x)
print(logits.shape)
loss = nn.CrossEntropyLoss()(logits, y)

print(loss)

# BCE cls token

transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=1, # num_classes 
    use_cls=True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10)).float()
y = torch.tensor([0, 1]).float()

print(x.shape)
logits = transformer(x)
loss = nn.BCEWithLogitsLoss()(logits, y)

print(loss)

# pooling test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    pooling=nn.AdaptiveAvgPool1d((1,)),
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10))
y = torch.tensor([0, 1])

print(x.shape)
logits = transformer(x)
print(logits.shape)
loss = nn.CrossEntropyLoss()(logits, y)

print(loss)

# pooling BCE test

# pooling test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=1, # num_classes 
    pooling=nn.AdaptiveAvgPool1d((1,)),
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10)).float()
y = torch.tensor([0, 1]).float()

print(x.shape)
logits = transformer(x)
print(logits.shape)
loss = nn.BCEWithLogitsLoss()(logits, y)

print(loss)

# normal test 

transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    pooling=nn.AdaptiveAvgPool1d((1,)),
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (1, 10))
y = torch.tensor([0])

print(x.shape)
logits = transformer(x)
print(logits.shape)