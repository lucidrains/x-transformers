from x_transformers.x_transformers import AttentionLayers

from x_transformers.multi_IO.IO_wrapper import MultiIOTransformerWrapper
from x_transformers.multi_IO.autoregressive_multiO import MultiOAutoregressiveWrapper
from x_transformers.multi_IO.xl_autoregressive_wrapper_multiO import MultiOXLAutoregressiveWrapper
from x_transformers import Decoder, AutoregressiveWrapper, TransformerWrapper, XLAutoregressiveWrapper
import torch

"""
model = XLAutoregressiveWrapper(
    pad_value=0,  # padding now properly works, testing with 0/other values
    net=TransformerWrapper(
        num_tokens=5,
        #max_seq_len=5,
        max_seq_len=3,
        use_abs_pos_emb=True,
        emb_dropout=0.1,
        post_emb_norm=True,
        attn_layers=AttentionLayers(
            dim=5, depth=1, heads=1, flash_attn=True,
            use_scalenorm=True, ff_glu=True, causal=True, sandwich_norm = True,
            alibi_pos_bias = True,
            alibi_num_heads = 1
        )
    )
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.Tensor([[2, 2, 3, 1, 2, 1, 1, 0, 0]]).long()
for i in range(1000):
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print(model(x, return_outputs=True))
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
print('eos_test: ', model.generate(prompts=torch.Tensor([[2, 2, 3]]).long(), seq_len=5, eos_token=1))
print('generate_test: ', model.generate(prompts=torch.Tensor([[1, 2, 3]]).long(), seq_len=5))
# """
"""
# multi input to multi output
model = MultiOAutoregressiveWrapper(
    outputs=3,
    # add_attn_z_loss=True,
    pad_value=torch.Tensor([0, 0, 0]),
    net=MultiIOTransformerWrapper(
        num_tokens=[3, 3, 3],
        autoregressive=True,
        max_seq_len=5,
        # use_abs_pos_emb=True,
        input_attn_layers=[
            AttentionLayers(dim=4, depth=1, heads=1, causal=True, attn_flash=True),
            # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True),
            AttentionLayers(dim=4, depth=1, heads=1, causal=True, attn_flash=True),
            AttentionLayers(dim=4, depth=1, heads=1, causal=True, attn_flash=True), ],
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True)],
        # output_attn_layers=[
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True),
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True),
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True)],
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
        # l2norm_embed=True,
        attn_layers=AttentionLayers(
            dim=12,
            depth=2,
            heads=4,
            # rotary_pos_emb=True,
            attn_flash=True,
            # use_scalenorm=True,
            # ff_glu=True,
            causal=True
        )
    ))
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# x = torch.Tensor(torch.randint(1, 3, (1, 10, 2))).float()
# print(x)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.Tensor([[[1, 1, 2], [1, 2, 0], [1, 2, 1], [0, 0, 0], [0, 0, 0]]]).long()
# print(x.shape)
for i in range(2000):
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print('logits_test: ', model(x, return_outputs=True)[1][0])
print('eos_test: ', model.generate(
                    prompts=torch.Tensor([[[1, 1, 2]]]).float(),
                    eos_token=torch.Tensor([[[1, 2, 1]]]).float(),
                    seq_len=3))
print('generate_test: ', model.generate(prompts=torch.Tensor([[[1, 1, 2]]]).float(), seq_len=3))
# """

# """
# multi input to multi output
model = MultiOXLAutoregressiveWrapper(
    outputs=3,
    # add_attn_z_loss=True,
    pad_value=torch.Tensor([0, 0, 0]),
    net=MultiIOTransformerWrapper(
        num_tokens=[3, 3, 3],
        autoregressive=True,
        max_seq_len=2,
        # use_abs_pos_emb=True,
        input_attn_layers=[
            AttentionLayers(dim=4, depth=1, heads=1, causal=True),
            # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True),
            AttentionLayers(dim=4, depth=1, heads=1, causal=True),
            AttentionLayers(dim=4, depth=1, heads=1, causal=True), ],
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True)],
        # output_attn_layers=[
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True),
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True),
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, ),
        #    AttentionLayers(dim=8, depth=1, heads=2, causal=True)],
        # rotary_pos_emb=True, attn_flash=True, use_scalenorm=True, ff_glu=True, )],
        # l2norm_embed=True,
        attn_layers=AttentionLayers(
            dim=12,
            depth=2,
            heads=4,
            # rotary_pos_emb=True,
            attn_flash=True,
            # use_scalenorm=True,
            # ff_glu=True,
            causal=True
        )
    ))
print(sum(p.numel() for p in model.parameters()))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# x = torch.Tensor(torch.randint(1, 3, (1, 10, 2))).float()
# print(x)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x = torch.Tensor([[[1, 1, 2], [1, 2, 2], [1, 2, 2], [0, 0, 0]]]).long()
# print(x.shape)
for i in range(200):
    loss = model(x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
print('logits_test: ', model(x, return_outputs=True)[1][0])
print('eos_test: ', model.generate(
    prompts=torch.Tensor([[[1, 1, 2]]]).float(),
    eos_token=torch.Tensor([[[1, 2, 2]]]).float(),
    seq_len=3))
print('generate_test: ', model.generate(prompts=torch.Tensor([[[1, 1, 2]]]).float(), seq_len=3))

#"""
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
    #memory_tokens_interspersed_every=1,
    #num_memory_tokens=2,
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

print("MODEL 2")
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
    #memory_tokens_interspersed_every=[1, 1, 1],
    #num_memory_tokens=[2, 1, 1],
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
