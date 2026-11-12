import pytest
import torch
from torch.nn import functional as F

from x_transformers import PrefixDecoder
from x_transformers.x_transformers import Attention, AttentionLayers


@pytest.fixture(params = ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.mark.parametrize('flash', [False, True])
@pytest.mark.parametrize('rank', [2, 3, 4])
@pytest.mark.parametrize('cross_attend', [False, True])
@pytest.mark.parametrize('num_mem_kv', [0, 4])
def test_token_mask_matches_explicit_memory_padding(device, flash, rank, cross_attend, num_mem_kv):
    torch.manual_seed(0)
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = num_mem_kv, flash = flash).to(device).eval()
    x = torch.randn(2, 6, 32, device = device)
    context = torch.randn(2, 9, 32, device = device) if cross_attend else None
    key_length = 9 if cross_attend else 6
    mask = torch.rand(6, key_length, device = device) > 0.3
    mask[:, 0] = True
    if rank == 3:
        mask = mask.expand(2, -1, -1).clone()
        mask[1, :, -1] = False
    elif rank == 4:
        mask = mask.expand(2, 2, -1, -1).clone()
        mask[1, :, :, -1] = False
    valid = torch.ones(2, key_length, dtype = torch.bool, device = device)
    valid[1, -1] = False
    kwargs = dict(context = context, context_mask = valid) if cross_attend else dict(mask = valid)
    padded = F.pad(mask, (num_mem_kv, 0), value = True)
    with torch.no_grad():
        expected = model(x, attn_mask = padded, **kwargs)
        actual = model(x, attn_mask = mask, **kwargs)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('flash', [False, True])
def test_prefix_decoder_with_memory_matches_explicit_mask(device, flash):
    torch.manual_seed(1)
    model = PrefixDecoder(dim = 32, depth = 1, heads = 2, attn_dim_head = 32,
                          attn_num_mem_kv = 4, attn_flash = flash).to(device).eval()
    x = torch.randn(2, 6, 32, device = device)
    mask = torch.ones(6, 6, dtype = torch.bool, device = device).tril()
    mask[:, :2] = True
    with torch.no_grad():
        expected = AttentionLayers.forward(model, x, attn_mask = F.pad(mask, (4, 0), value = True))
        actual = model(x, prefix_attn_len = 2)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('flash', [False, True])
def test_preserve_complete_and_broadcast_masks(device, flash):
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = 4, flash = flash).to(device).eval()
    for key_length in (1, 6):
        x = torch.randn(2, key_length, 32, device = device)
        # A singleton key dimension already broadcasts to memory and real keys.
        broadcast = torch.zeros(key_length, 1, dtype = torch.bool, device = device)
        complete = broadcast.expand(key_length, key_length + 4)
        captured = []
        handle = model.attend.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs['mask'].clone()), with_kwargs = True)
        try:
            with torch.no_grad():
                model(x, attn_mask = complete)
                model(x, attn_mask = broadcast)
        finally:
            handle.remove()
        assert not captured[0].any()
        assert not captured[1].any()
        assert captured[0].shape[-1] == key_length + 4
        assert captured[1].shape[-1] == 1


def test_memory_mask_combines_with_cached_keys(device):
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = 4).to(device).eval()
    x = torch.randn(2, 6, 32, device = device)
    with torch.no_grad():
        _, cache = model(x[:, :5], return_intermediates = True)
        mask = torch.ones(1, 6, dtype = torch.bool, device = device)
        mask[:, 2] = False
        expected = model(x[:, 5:], cache = cache, attn_mask = F.pad(mask, (4, 0), value = True))
        actual = model(x[:, 5:], cache = cache, attn_mask = mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('flash', [False, True])
def test_memory_only_context(device, flash):
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = 4, flash = flash).to(device).eval()
    x = torch.randn(2, 6, 32, device = device)
    context = torch.empty(2, 0, 32, device = device)
    with torch.no_grad():
        expected = model(x, context = context, attn_mask = torch.ones(6, 4, dtype = torch.bool, device = device))
        actual = model(x, context = context, attn_mask = torch.empty(6, 0, dtype = torch.bool, device = device))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('flash', [False, True])
def test_memory_mask_backward_matches_explicit_padding(device, flash):
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = 4, flash = flash).to(device)
    x = torch.randn(2, 6, 32, device = device, requires_grad = True)
    mask = torch.ones(6, 6, dtype = torch.bool, device = device).tril()
    weight = torch.randn_like(x)
    actual = model(x, attn_mask = mask)
    actual_grad = torch.autograd.grad((actual * weight).sum(), (x, model.mem_k, model.mem_v))
    expected = model(x, attn_mask = F.pad(mask, (4, 0), value = True))
    expected_grad = torch.autograd.grad((expected * weight).sum(), (x, model.mem_k, model.mem_v))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b)
        assert torch.isfinite(a).all()


def test_additional_kv_complete_mask_is_unchanged(device):
    model = Attention(dim = 32, heads = 2, dim_head = 16, num_mem_kv = 4).to(device).eval()
    x = torch.randn(2, 6, 32, device = device)
    kv = (torch.randn(2, 2, 3, 16, device = device), torch.randn(2, 2, 3, 16, device = device))
    mask = torch.ones(6, 13, dtype = torch.bool, device = device)
    mask[:, :3] = False
    with torch.no_grad():
        expected = model(x)
        actual = model(x, attn_mask = mask, additional_key_values = kv)
    torch.testing.assert_close(actual, expected)
