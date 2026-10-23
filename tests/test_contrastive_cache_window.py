from copy import deepcopy

import pytest
import torch
from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


@pytest.mark.parametrize("wrap_amateur", [False, True])
@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("restricted", [False, True])
def test_contrastive_decoding_trims_amateur_cache_with_expert(
    wrap_amateur, depth, restricted
):
    torch.manual_seed(11)
    expert = TransformerWrapper(
        num_tokens=11,
        max_seq_len=4,
        attn_layers=Decoder(
            dim=16, depth=depth, heads=2, attn_dim_head=8, rotary_pos_emb=True
        ),
    )
    amateur = deepcopy(expert)
    expert_outputs = []
    amateur_outputs = []

    def record(outputs):
        def hook(module, args, output):
            logits, cache = output
            cache_lengths = [
                inter.cached_kv[0].shape[-2]
                for inter in cache.attn_intermediates
                if inter.layer_type == "a"
            ]
            outputs.append((logits.detach().clone(), cache_lengths))

        return hook

    expert.register_forward_hook(record(expert_outputs))
    amateur.register_forward_hook(record(amateur_outputs))
    wrapped_amateur = AutoregressiveWrapper(amateur) if wrap_amateur else amateur
    wrapper = AutoregressiveWrapper(expert)
    wrapper.generate(
        torch.tensor([[1, 2, 3, 4]]),
        seq_len=5,
        temperature=0,
        amateur_model=wrapped_amateur,
        cache_kv=True,
        restrict_to_max_seq_len=restricted,
    )

    # Identical native networks must see the same sliding context and produce the
    # same logits. The restricted mode keeps at most four KV tokens in both models.
    assert len(expert_outputs) == len(amateur_outputs) == 5
    for step, (
        (expert_logits, expert_lengths),
        (amateur_logits, amateur_lengths),
    ) in enumerate(zip(expert_outputs, amateur_outputs)):
        expected_length = 4 if restricted else 4 + step
        assert expert_lengths == amateur_lengths == [expected_length] * depth
        torch.testing.assert_close(expert_logits, amateur_logits)
