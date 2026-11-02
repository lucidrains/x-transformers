import pytest
import torch

from x_transformers import AutoregressiveWrapper, Decoder, TransformerWrapper


def make_wrapper(training):
    net = TransformerWrapper(
        num_tokens = 8,
        max_seq_len = 8,
        attn_layers = Decoder(dim = 8, depth = 1, heads = 2)
    )
    return AutoregressiveWrapper(net).train(training)


@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('method', ('generate', 'beam_search'))
def test_generation_restores_mode_after_invalid_options(training, method):
    wrapper = make_wrapper(training)
    with pytest.raises(AssertionError):
        getattr(wrapper, method)(
            torch.tensor([[1, 2]]),
            1,
            filter_logits_fn = 'invalid'
        )

    assert wrapper.training is training
    assert wrapper.net.training is training


@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('method', ('generate', 'beam_search'))
def test_generation_restores_mode_after_model_error(training, method):
    wrapper = make_wrapper(training)
    error = RuntimeError('model failure')

    def fail(module, inputs):
        assert module.training is False
        raise error

    handle = wrapper.net.register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            getattr(wrapper, method)(torch.tensor([[1, 2]]), 1)
    finally:
        handle.remove()

    assert caught.value is error
    assert wrapper.training is training
    assert wrapper.net.training is training


@pytest.mark.parametrize('training', (False, True))
@pytest.mark.parametrize('method', ('generate', 'beam_search'))
def test_generation_restores_mode_after_success(training, method):
    wrapper = make_wrapper(training)
    modes = []
    handle = wrapper.net.register_forward_pre_hook(
        lambda module, inputs: modes.append(module.training)
    )
    try:
        output = getattr(wrapper, method)(torch.tensor([[1, 2]]), 1)
    finally:
        handle.remove()

    assert output.shape == (1, 1)
    assert modes == [False]
    assert wrapper.training is training
    assert wrapper.net.training is training
