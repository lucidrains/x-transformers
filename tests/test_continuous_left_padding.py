from copy import deepcopy

import pytest
import torch

from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder


@pytest.mark.parametrize('layer_type', (Encoder, Decoder))
@pytest.mark.parametrize('absolute_positions', (False, True))
@pytest.mark.parametrize('explicit_mask', (False, True))
@pytest.mark.parametrize('memory_tokens', (0, 2))
def test_continuous_sequence_start_masks_left_padding(layer_type, absolute_positions, explicit_mask, memory_tokens):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3,
        dim_out = 5,
        max_seq_len = 8,
        num_memory_tokens = memory_tokens,
        use_abs_pos_emb = absolute_positions,
        attn_layers = layer_type(
            dim = 16, depth = 1, heads = 2, attn_dim_head = 16,
            rotary_pos_emb = not absolute_positions
        )
    ).eval()
    reference = deepcopy(model)
    inputs = torch.randn(2, 6, 3, requires_grad = True)
    starts = torch.tensor([2, 1])
    mask = torch.arange(6)[None, :] >= starts[:, None]
    actual = model(inputs, seq_start_pos = starts, mask = mask if explicit_mask else None)
    if memory_tokens:
        reference_out = reference(inputs, seq_start_pos = starts, mask = mask)
        expected = [reference_out[row, start:] for row, start in enumerate(starts)]
    else:
        expected = [reference(inputs[row:row + 1, start:])[0] for row, start in enumerate(starts)]
    for row, start in enumerate(starts):
        torch.testing.assert_close(actual[row, start:], expected[row], rtol = 1e-4, atol = 1e-5)

    actual_grad, = torch.autograd.grad(actual[mask].square().sum(), inputs)
    expected_grad, = torch.autograd.grad(sum(out.square().sum() for out in expected), inputs)
    torch.testing.assert_close(actual_grad, expected_grad, rtol = 1e-4, atol = 1e-5)
    torch.testing.assert_close(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))


@pytest.mark.parametrize('absolute_positions', (False, True))
@pytest.mark.parametrize('input_not_include_cache', (False, True))
def test_continuous_left_padding_with_cache(absolute_positions, input_not_include_cache):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8,
        use_abs_pos_emb = absolute_positions,
        attn_layers = Decoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16,
                              rotary_pos_emb = not absolute_positions)
    ).eval()
    sequence = torch.randn(2, 5, 3)
    starts = torch.tensor([2, 1])
    with torch.no_grad():
        _, cache = model(sequence[:, :4], seq_start_pos = starts, return_intermediates = True)
        actual = model(
            sequence[:, -1:] if input_not_include_cache else sequence,
            cache = cache, seq_start_pos = starts,
            input_not_include_cache = input_not_include_cache
        )
        expected = torch.cat([model(sequence[row:row + 1, start:])[:, -1:]
                              for row, start in enumerate(starts)])
    torch.testing.assert_close(actual[:, -1:], expected, rtol = 1e-4, atol = 1e-5)


@pytest.mark.parametrize('absolute_positions', (False, True))
@pytest.mark.parametrize('external_memory', (False, True))
def test_continuous_left_padding_preserves_prefix_and_memory(absolute_positions, external_memory):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8,
        use_abs_pos_emb = absolute_positions,
        attn_layers = Decoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16,
                              rotary_pos_emb = not absolute_positions)
    ).eval()
    inputs = torch.randn(2, 6, 3, requires_grad = True)
    starts = torch.tensor([2, 1])
    mask = torch.arange(6)[None, :] >= starts[:, None]
    prefix = torch.randn(2, 2, 16, requires_grad = True)
    mems = [torch.randn(2, 3, 16)] if external_memory else None
    actual = model(inputs, seq_start_pos = starts, prepend_embeds = prefix, mems = mems)
    expected = model(inputs, seq_start_pos = starts, mask = mask, prepend_embeds = prefix, mems = mems)
    torch.testing.assert_close(actual[:, 2:][mask], expected[:, 2:][mask])
    actual_grads = torch.autograd.grad(actual[:, 2:][mask].square().sum(), (inputs, prefix))
    expected_grads = torch.autograd.grad(expected[:, 2:][mask].square().sum(), (inputs, prefix))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    torch.testing.assert_close(actual_grads[0][~mask], torch.zeros_like(actual_grads[0][~mask]))


@pytest.mark.parametrize('absolute_positions', (False, True))
def test_continuous_left_padding_pool_keeps_valid_prefix(absolute_positions):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8, average_pool_embed = True,
        use_abs_pos_emb = absolute_positions,
        attn_layers = Encoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16,
                              rotary_pos_emb = not absolute_positions)
    ).eval()
    reference = deepcopy(model)
    reference.average_pool_embed = False
    inputs = torch.randn(2, 6, 3)
    prefix = torch.randn(2, 2, 16)
    prefix_mask = torch.tensor([[True, False], [True, True]])
    starts = torch.tensor([2, 1])
    body_mask = torch.arange(6)[None, :] >= starts[:, None]
    actual = model(inputs, seq_start_pos = starts, prepend_embeds = prefix, prepend_mask = prefix_mask)
    all_outputs = reference(inputs, seq_start_pos = starts, mask = body_mask,
                            prepend_embeds = prefix, prepend_mask = prefix_mask)
    valid = torch.cat((prefix_mask, body_mask), dim = 1)
    expected = torch.stack([row[row_mask].mean(dim = 0) for row, row_mask in zip(all_outputs, valid)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('input_not_include_cache', (False, True))
@pytest.mark.parametrize('prepend', (False, True))
@pytest.mark.parametrize('cache_age', (0, 1, 2))
def test_continuous_left_padding_pool_uses_only_current_cached_queries(input_not_include_cache, prepend, cache_age):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8, average_pool_embed = True,
        attn_layers = Decoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16)
    ).eval()
    reference = deepcopy(model)
    reference.average_pool_embed = False
    sequence = torch.randn(2, 6, 3)
    prefix = torch.randn(2, 2, 16) if prepend else None
    starts = torch.tensor([2, 1])
    mask = torch.arange(6)[None, :] >= starts[:, None]
    with torch.no_grad():
        _, actual_cache = model(sequence[:, :4], seq_start_pos = starts,
                                prepend_embeds = prefix, return_intermediates = True)
        _, expected_cache = reference(sequence[:, :4], seq_start_pos = starts, mask = mask[:, :4],
                                      prepend_embeds = prefix, return_intermediates = True)
        inputs = sequence[:, -2:] if input_not_include_cache else sequence
        actual = model(inputs, seq_start_pos = starts, cache = actual_cache, cache_age = cache_age,
                       input_not_include_cache = input_not_include_cache, prepend_embeds = prefix)
        query_outputs = reference(inputs, seq_start_pos = starts, mask = mask, cache = expected_cache,
                                  cache_age = cache_age, input_not_include_cache = input_not_include_cache,
                                  prepend_embeds = prefix)
        query_mask = mask[:, -2:] if input_not_include_cache else mask
        if prepend:
            query_mask = torch.cat((torch.ones(2, 2, dtype = torch.bool), query_mask), dim = 1)
        if cache_age > 0:
            query_mask = query_mask[:, -cache_age:]
        expected = torch.stack([row[valid].mean(dim = 0) for row, valid in zip(query_outputs, query_mask)])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('memory_len', (0, 3))
def test_continuous_sequence_start_combines_with_explicit_key_mask(memory_len):
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8,
        attn_layers = Encoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16)
    ).eval()
    inputs = torch.randn(2, 6, 3)
    starts = torch.tensor([2, 1])
    left_mask = torch.arange(6)[None, :] >= starts[:, None]
    key_mask = torch.ones(2, memory_len + 6, dtype = torch.bool)
    key_mask[:, memory_len + 4] = False
    mems = [torch.randn(2, memory_len, 16)] if memory_len else None
    actual = model(inputs, seq_start_pos = starts, self_attn_kv_mask = key_mask, mems = mems)
    full_left_mask = torch.cat((torch.ones(2, memory_len, dtype = torch.bool), left_mask), dim = 1)
    expected = model(inputs, seq_start_pos = starts, mask = left_mask,
                     self_attn_kv_mask = full_left_mask & key_mask, mems = mems)
    torch.testing.assert_close(actual[left_mask], expected[left_mask])


def test_cached_full_sequence_keeps_explicit_history_masks():
    torch.manual_seed(0)
    model = ContinuousTransformerWrapper(
        dim_in = 3, dim_out = 3, max_seq_len = 8,
        attn_layers = Decoder(dim = 16, depth = 1, heads = 2, attn_dim_head = 16)
    ).eval()
    inputs = torch.randn(2, 6, 3)
    prefix = torch.randn(2, 2, 16)
    prefix_mask = torch.tensor([[True, False], [False, True]])
    starts = torch.tensor([2, 1])
    body_mask = torch.arange(6)[None, :] >= starts[:, None]
    body_mask[:, 3] = False
    with torch.no_grad():
        _, cache = model(inputs[:, :4], seq_start_pos = starts, mask = body_mask[:, :4],
                         prepend_embeds = prefix, prepend_mask = prefix_mask, return_intermediates = True)
        _, attentions = model(inputs, seq_start_pos = starts, mask = body_mask,
                              prepend_embeds = prefix, prepend_mask = prefix_mask,
                              cache = cache, cache_age = 0, return_attn = True)
    history_mask = torch.cat((prefix_mask, body_mask[:, :4]), dim = 1)
    current_mask = torch.cat((prefix_mask, body_mask), dim = 1)
    allowed_keys = torch.cat((history_mask, current_mask), dim = 1)
    for attention in attentions:
        blocked = ~allowed_keys[:, None, None, :].expand_as(attention)
        torch.testing.assert_close(attention[blocked], torch.zeros_like(attention[blocked]))
