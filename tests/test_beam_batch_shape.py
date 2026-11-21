import pytest
import torch

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


@pytest.mark.parametrize('batch_shape', ((), (1,), (2,), (2, 2)))
@pytest.mark.parametrize('cache_kv', (False, True))
def test_all_beams_preserve_prompt_batch_dimensions(batch_shape, cache_kv):
    torch.manual_seed(0)
    net = TransformerWrapper(
        num_tokens = 11,
        max_seq_len = 8,
        attn_layers = Decoder(
            dim = 8,
            depth = 1,
            heads = 2,
            rotary_pos_emb = True
        )
    )
    wrapper = AutoregressiveWrapper(net)
    prompts = torch.randint(0, 11, (*batch_shape, 3))

    beams, scores = wrapper.beam_search(
        prompts,
        2,
        beams = 2,
        cache_kv = cache_kv,
        return_beams_and_scores = True
    )

    references = [
        wrapper.beam_search(
            prompt[None],
            2,
            beams = 2,
            cache_kv = cache_kv,
            return_beams_and_scores = True
        )
        for prompt in prompts.reshape(-1, 3)
    ]
    expected_beams = torch.cat([result[0] for result in references], dim = 1)
    expected_scores = torch.cat([result[1] for result in references], dim = 1)

    assert beams.shape == (2, *batch_shape, 2)
    assert scores.shape == (2, *batch_shape)
    assert torch.equal(beams, expected_beams.reshape(2, *batch_shape, 2))
    torch.testing.assert_close(scores, expected_scores.reshape(2, *batch_shape))

    best = wrapper.beam_search(prompts, 2, beams = 2, cache_kv = cache_kv)
    assert torch.equal(best, beams[0])
