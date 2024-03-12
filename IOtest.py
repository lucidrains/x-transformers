from x_transformers.multi_IO.IO_wrapper import MultiIOTransformerWrapper
from x_transformers.multi_IO.autoregressive_multiO import MultiOAutoregressiveWrapper
from x_transformers import Decoder, AutoregressiveWrapper, TransformerWrapper
import torch
"""
model = AutoregressiveWrapper(
    MultiIOTransformerWrapper(
        num_tokens=8,
        max_seq_len=10,
        use_abs_pos_emb=True,
        emb_dropout=0.1,
        post_emb_norm=True,
        attn_layers=Decoder(max_seq_len=10, dim=4, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ))
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(10000):
    x = torch.Tensor([[1,2,3]]).long()
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
"""

# multi input to multi output
model = MultiOAutoregressiveWrapper(
    outputs=2,
    pad_value=torch.Tensor([0, 0]),
    net=MultiIOTransformerWrapper(
        num_tokens=[8, 4],
        max_seq_len=10,
        use_abs_pos_emb=True,
        max_mem_len=10,
        shift_mem_down=1,
        emb_dropout=0.1,
        post_emb_norm=True,
        input_attn_layers=[
            Decoder(dim=2, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
            Decoder(dim=1, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
        output_attn_layers=[
            Decoder(dim=4, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
            Decoder(dim=4, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
        tie_embedding=True,
        l2norm_embed=True,
        emb_frac_gradient=0.1,
        attn_z_loss_weight=0.1,
        attn_layers=Decoder(
            dim=4,
            depth=1,
            heads=1,
            rotary_pos_emb=True,
            attn_flash=True,
            use_scalenorm=True,
            ff_glu=True,
        )
    ))
x = torch.Tensor(torch.randint(1, 3, (1, 10, 2))).float()
print(x)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for i in range(10000):
    x = torch.Tensor([[[1,1], [2,2], [3,3]]]).float()
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# for i in range(100):

"""
model = MultiIOTransformerWrapper(
    num_tokens=8,
    max_seq_len=10,
    use_abs_pos_emb=True,
    max_mem_len=10,
    shift_mem_down=1,
    emb_dropout=0.1,
    post_emb_norm=True,
    output_attn_layers=[
        Decoder(dim=1, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
        Decoder(dim=2, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
    memory_tokens_interspersed_every=1,
    num_memory_tokens=2,
    tie_embedding=False,
    l2norm_embed=True,
    emb_frac_gradient=0.1,
    attn_z_loss_weight=0.1,
    attn_layers=Decoder(
        dim=4,
        depth=1,
        heads=1,
        rotary_pos_emb=True,
        attn_flash=True,
        use_scalenorm=True,
        ff_glu=True,
    )
)
x = torch.Tensor(torch.randint(1, 3, (1, 10))).float()
print(x)
print(model(x))

model = MultiIOTransformerWrapper(
    num_tokens=[8, 4, 5],
    max_seq_len=10,
    use_abs_pos_emb=True,
    max_mem_len=10,
    logits_dim=[1, 2],
    shift_mem_down=1,
    emb_dropout=0.1,
    post_emb_norm=True,
    input_attn_layers=[
        Decoder(dim=2, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
        Decoder(dim=1, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
        Decoder(dim=1, depth=1, heads=1, rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
    memory_tokens_interspersed_every=[1, 1, 1],
    num_memory_tokens=[2, 1, 1],
    tie_embedding=False,
    l2norm_embed=True,
    emb_frac_gradient=0.1,
    attn_z_loss_weight=0.1,
    attn_layers=Decoder(
        dim=4,
        depth=1,
        heads=1,
        rotary_pos_emb=True,
        attn_flash=True,
        use_scalenorm=True,
        ff_glu=True,
    )
)
x = torch.Tensor(torch.randint(1, 3, (1, 10, 3))).float()
print(x)
print(model(x))"""
