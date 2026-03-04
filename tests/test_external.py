import torch
import pytest
from x_transformers.x_transformers import Attend, Attention, AttentionLayers

param = pytest.mark.parametrize

def reset_exp_det():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@pytest.mark.skipif(
    not torch.cuda.is_available() or \
    torch.cuda.get_device_capability()[0] < 8 or \
    __import__('importlib').util.find_spec('flash_attn') is None,
    reason="CUDA compute capability must be >= 8 and flash_attn must be installed"
)
@param('exp', (
    dict(causal=True, same_partition=True, pos_enc='rotary_pos_emb'),
    dict(causal=False, same_partition=True, pos_enc='rotary_pos_emb'),
    dict(causal=False, same_partition=False, pos_enc='rotary_pos_emb'),
    dict(causal=True, same_partition=True, pos_enc='rotary_xpos'))
)
def test_flash_pack_seq(exp):
    seq_len = 1024
    dim = 256
    n_part = 4
    n_layers = 4
    causal = exp['causal']
    same_partition = exp.get('same_partition', False)
    atl_kwargs = {exp['pos_enc']: True}
    mem_len = 128 if not same_partition else seq_len
    x = torch.randn((seq_len, dim)).cuda().half()
    mem = torch.randn((mem_len, dim)).cuda().half()

    pad_val = 99.0 # float('-inf')
    def partition(x, num_parts):
        total = x.shape[0]
        split_points = sorted(torch.randint(1, total, (num_parts - 1,)).tolist())
        splits = torch.tensor_split(x, split_points, dim=0)
        import numpy as np
        attn_cu_lengths = torch.tensor([0] + np.cumsum([split.shape[0] for split in splits]).tolist()).int().cuda()
        return splits, attn_cu_lengths
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            if same_partition and causal:
                splits, attn_cu_lengths = partition(x, n_part)
                # Split mem using the cutting points from attn_cu_lengths
                split_points = attn_cu_lengths[1:-1].tolist()
                splits_mem = torch.tensor_split(mem, split_points, dim=0)
                attn_cu_lengths_context = attn_cu_lengths
                padded_batch = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True, padding_value=pad_val)
                padded_mem = torch.nn.utils.rnn.pad_sequence(splits_mem, batch_first=True, padding_value=pad_val)

                mask = (padded_batch != pad_val).any(dim=-1)
                context_mask = (padded_mem != pad_val).any(dim=-1)
                max_len = padded_batch.shape[1]
                attn_mask = mask.unsqueeze(1) & torch.tril(torch.ones((max_len, max_len), dtype=torch.bool)).cuda().unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(1)
            else:
                splits, attn_cu_lengths = partition(x, n_part)
                splits_mem, attn_cu_lengths_context = partition(mem, n_part)
                padded_batch = torch.nn.utils.rnn.pad_sequence(splits, batch_first=True, padding_value=pad_val)
                padded_mem = torch.nn.utils.rnn.pad_sequence(splits_mem, batch_first=True, padding_value=pad_val)

                mask = (padded_batch != pad_val).any(dim=-1)
                context_mask = (padded_mem != pad_val).any(dim=-1)
                attn_mask = mask.unsqueeze(2) & context_mask.unsqueeze(1)
                attn_mask = attn_mask.unsqueeze(1)


            # Standard padding
            reset_exp_det()
            atd = Attend(flash = False, flash_pack_seq = False,causal=causal).cuda().eval()
            o_atd = atd(q=padded_batch[:,None], k=padded_mem[:,None], v=padded_mem[:,None], mask=attn_mask)
            o_atd = o_atd[0][:,0]
            o_atd = torch.cat([o[~(m == pad_val).all(-1)] for o, m in zip(o_atd, padded_batch)], dim=0)

            att=Attention(dim=dim,flash=False,causal=causal).cuda().eval()
            o_att = att(
                x=padded_batch,
                context=padded_mem,
                attn_mask = attn_mask
            )
            o_att = torch.cat([o[~(m == pad_val).all(-1)] for o, m in zip(o_att, padded_batch)], dim=0)

            atl = AttentionLayers(dim=dim, depth=n_layers, cross_attend=True, causal=causal, attn_flash=True, attn_flash_pack_seq=False, **atl_kwargs).cuda().eval()
            o_atl = atl(
                x=padded_batch,
                context=padded_mem,
                context_mask = context_mask,
                mask = mask,
            )
            o_atl = torch.cat([o[~(m == pad_val).all(-1)] for o, m in zip(o_atl, padded_batch)], dim=0)


            # Block masking
            reset_exp_det()
            atd_block = Attend(flash = True, flash_pack_seq = True, causal=causal).cuda().eval()

            flash_pack_seq_kwargs = dict(
                cu_seqlens_q=attn_cu_lengths,
                max_seqlen_q = attn_cu_lengths.diff().max().item(),
                cu_seqlens_k=attn_cu_lengths,
                max_seqlen_k = attn_cu_lengths.diff().max().item()
            )
            flash_pack_seq_kwargs_context = dict(
                cu_seqlens_q=attn_cu_lengths,
                max_seqlen_q = attn_cu_lengths.diff().max().item(),
                cu_seqlens_k=attn_cu_lengths_context,
                max_seqlen_k = attn_cu_lengths_context.diff().max().item()
            )
            o_atd_block = atd_block(x[None,None], mem[None,None], mem[None,None], flash_pack_seq_kwargs=flash_pack_seq_kwargs_context)
            o_atd_block = o_atd_block[0][0,0]


            att_block=Attention(dim=dim,flash=True,flash_pack_seq=True, causal=causal).cuda().eval()
            o_att_block = att_block(
                x = x.unsqueeze(0),
                context = mem.unsqueeze(0),
                flash_pack_seq_kwargs=flash_pack_seq_kwargs_context
            )[0]

            atl_block = AttentionLayers(dim=dim, depth=n_layers, cross_attend=True, causal=causal, attn_flash=True, attn_flash_pack_seq=True, **atl_kwargs).cuda().eval()
            o_atl_block = atl_block(
                x = x.unsqueeze(0),
                context = mem.unsqueeze(0),
                flash_pack_seq_kwargs=flash_pack_seq_kwargs,
                flash_pack_seq_context_kwargs=flash_pack_seq_kwargs_context,
            )[0]
            torch.testing.assert_close(o_atd, o_atd_block , atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(o_att, o_att_block , atol=5e-3, rtol=5e-3)
            torch.testing.assert_close(o_atl, o_atl_block , atol=5e-3, rtol=5e-3)
