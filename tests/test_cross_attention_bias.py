import pytest
import torch
from torch.nn import functional as F

from x_transformers import CrossAttender, Decoder, PrefixDecoder, TransformerWrapper


@pytest.fixture(params = ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def device(request):
    return request.param


@pytest.mark.parametrize('flash', [False, True])
@pytest.mark.parametrize('rank', [2, 3, 4])
@pytest.mark.parametrize('memory', [0, 4])
def test_bias_changes_output_and_preserves_none_and_zero(device, flash, rank, memory):
    torch.manual_seed(3)
    model = CrossAttender(dim = 32, depth = 2, heads = 2, attn_dim_head = 16,
                         cross_attn_num_mem_kv = memory, attn_flash = flash).to(device).eval()
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    bias = torch.randn(6, 5 + memory, device = device)
    if rank == 3:
        bias = bias.expand(2, -1, -1).clone()
    elif rank == 4:
        bias = bias.expand(2, 2, -1, -1).clone()
    with torch.no_grad():
        base = model(x, context = context)
        explicit_none = model(x, context = context, cross_attn_bias = None)
        zero = model(x, context = context, cross_attn_bias = torch.zeros_like(bias))
        guided = model(x, context = context, cross_attn_bias = bias)
        per_layer = model(x, context = context, cross_attn_bias = [bias, bias])
    torch.testing.assert_close(base, explicit_none, rtol = 0, atol = 0)
    torch.testing.assert_close(base, zero)
    torch.testing.assert_close(guided, per_layer)
    assert not torch.allclose(base, guided)
    assert torch.isfinite(guided).all()


@pytest.mark.parametrize('flash', [False, True])
def test_bias_gradient_and_padding(device, flash):
    torch.manual_seed(4)
    model = CrossAttender(dim = 32, depth = 1, heads = 2, attn_dim_head = 16,
                         cross_attn_num_mem_kv = 4, attn_flash = flash).to(device)
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    valid = torch.tensor([[True, True, True, False, False]] * 2, device = device)
    scores = torch.randn(2, 1, 6, 5, device = device, requires_grad = True)
    bias = F.pad(2 * scores.sigmoid(), (4, 0))
    out = model(x, context = context, context_mask = valid, cross_attn_bias = bias)
    gradient, = torch.autograd.grad((out * torch.randn_like(out)).sum(), scores)
    assert torch.isfinite(gradient).all()
    assert gradient[..., :3].abs().sum() > 0
    assert torch.count_nonzero(gradient[..., 3:]) == 0
    with torch.no_grad():
        changed = bias.clone()
        changed[..., -2:] = 1000
        actual = model(x, context = context, context_mask = valid, cross_attn_bias = changed)
    torch.testing.assert_close(actual, out)


def test_bias_routes_by_layer_with_reordering_and_dropout(device):
    model = CrossAttender(dim = 32, custom_layers = ('c', 'c'), heads = 2, attn_dim_head = 16).to(device)
    x = torch.randn(2, 6, 32, device = device)
    contexts = (torch.randn(2, 5, 32, device = device), torch.randn(2, 7, 32, device = device))
    biases = (torch.randn(6, 5, device = device), torch.randn(6, 7, device = device))
    observed = []
    def hook(index):
        def capture(module, args, kwargs):
            observed.append((index, kwargs.get('attn_bias')))
        return capture
    handles = [layer[1].register_forward_pre_hook(hook(i), with_kwargs = True) for i, layer in enumerate(model.layers)]
    try:
        with torch.no_grad():
            model.eval()(x, context = contexts, cross_attn_bias = biases, layers_execute_order = (1, 0))
            assert [i for i, _ in observed] == [1, 0]
            assert observed[0][1] is biases[1] and observed[1][1] is biases[0]
            observed.clear()
            model.layer_dropouts = (1., 0.)
            model.train()(x, context = contexts, cross_attn_bias = (None, biases[1]))
            assert len(observed) == 1 and observed[0][0] == 1 and observed[0][1] is biases[1]
    finally:
        for handle in handles:
            handle.remove()
    with pytest.raises(AssertionError, match = 'cross attention biases'):
        model(x, context = contexts, cross_attn_bias = [biases[0]])


@pytest.mark.parametrize('flash', [False, True])
@pytest.mark.parametrize('cache_age', [1, 2])
@pytest.mark.parametrize('memory', [0, 4])
def test_bias_cached_decode_matches_full_decode(device, flash, cache_age, memory):
    torch.manual_seed(5)
    model = Decoder(dim = 32, depth = 2, heads = 2, attn_dim_head = 32,
                    cross_attend = True, cross_attn_num_mem_kv = memory, attn_flash = flash).to(device).eval()
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    bias = torch.randn(2, 1, 6, 5 + memory, device = device)
    with torch.no_grad():
        expected = model(x, context = context, cross_attn_bias = bias)
        _, cache = model(x[:, :-cache_age], context = context, cross_attn_bias = bias[..., :-cache_age, :], return_hiddens = True)
        full_bias = model(x, context = context, cache = cache, cache_age = cache_age, cross_attn_bias = bias)
        sliced_bias = model(x[:, -cache_age:], context = context, cache = cache, cache_age = cache_age,
                            input_not_include_cache = True, cross_attn_bias = bias[..., -cache_age:, :])
    torch.testing.assert_close(full_bias, expected[:, -cache_age:], rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(sliced_bias, full_bias, rtol = 1e-4, atol = 1e-5)


def test_reject_bias_with_context_token_dropout(device):
    model = CrossAttender(dim = 32, depth = 1, heads = 2, cross_attn_tokens_dropout = 0.2).to(device)
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    with pytest.raises(AssertionError, match = 'cross_attn_tokens_dropout'):
        model(x, context = context, cross_attn_bias = torch.zeros(6, 5, device = device))
    with torch.no_grad():
        model(x, context = context, cross_attn_bias = [None])
        model.eval()(x, context = context, cross_attn_bias = torch.zeros(6, 5, device = device))


def test_prefix_and_wrapper_forward_bias(device):
    model = TransformerWrapper(num_tokens = 32, max_seq_len = 6,
        attn_layers = PrefixDecoder(dim = 32, depth = 1, heads = 2, cross_attend = True)).to(device).eval()
    tokens = torch.randint(0, 32, (2, 6), device = device)
    context = torch.randn(2, 5, 32, device = device)
    with torch.no_grad():
        out = model(tokens, context = context, prefix_attn_len = 2, cross_attn_bias = torch.zeros(6, 5, device = device))
    assert out.shape == (2, 6, 32)


@pytest.mark.parametrize('flash', [False, True])
def test_bias_respects_pair_mask_and_broadcast_queries(device, flash):
    model = CrossAttender(dim = 32, depth = 1, heads = 2, attn_flash = flash).to(device).eval()
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    bias = torch.randn(2, 1, 1, 5, device = device)
    mask = torch.ones(6, 5, dtype = torch.bool, device = device)
    mask[:, -1] = False
    with torch.no_grad():
        expected = model(x, context = context, cross_attn_attn_mask = mask, cross_attn_bias = bias)
        bias[..., -1] = 1000
        actual = model(x, context = context, cross_attn_attn_mask = mask, cross_attn_bias = bias)
    torch.testing.assert_close(actual, expected)


def test_reject_bias_with_packed_context_or_invalid_rank(device):
    model = CrossAttender(dim = 32, depth = 1, heads = 2).to(device)
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    with pytest.raises(AssertionError, match = 'flash packed context'):
        model(x, context = context, cross_attn_bias = torch.zeros(6, 5, device = device), flash_pack_seq_context_kwargs = {})
    with pytest.raises(AssertionError, match = '2 to 4 dimensions'):
        model(x, context = context, cross_attn_bias = torch.zeros(5, device = device))


@pytest.mark.parametrize('flash', [False, True])
def test_bias_bfloat16_autocast_gradient(device, flash):
    torch.manual_seed(6)
    model = CrossAttender(dim = 32, depth = 1, heads = 2, attn_flash = flash).to(device)
    x, context = torch.randn(2, 6, 32, device = device), torch.randn(2, 5, 32, device = device)
    bias = torch.randn(2, 1, 6, 5, device = device, requires_grad = True)
    with torch.autocast(device_type = device, dtype = torch.bfloat16):
        out = model(x, context = context, cross_attn_bias = bias)
    gradient, = torch.autograd.grad((out.float() * torch.randn_like(out.float())).sum(), bias)
    assert torch.isfinite(out).all() and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
