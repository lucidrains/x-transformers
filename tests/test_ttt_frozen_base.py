from copy import deepcopy

import pytest
import torch

from x_transformers import Decoder, TransformerWrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper


VALUE_PATH = "attn_layers.layers.0.1.to_v"


@pytest.mark.parametrize("freeze", ["all", "adapted", "none"])
@pytest.mark.parametrize("method", ["forward", "generate"])
@pytest.mark.parametrize("episodic", [0, 2])
def test_ttt_adapts_fast_weights_without_unfreezing_the_base(freeze, method, episodic):
    torch.manual_seed(16)
    model = TransformerWrapper(
        num_tokens=16, max_seq_len=4, max_mem_len=4,
        attn_layers=Decoder(dim=16, depth=1, heads=2, attn_dim_head=8),
    )
    adapted = model.attn_layers.layers[0][1].to_v
    if freeze == "all":
        model.requires_grad_(False)
    elif freeze == "adapted":
        adapted.requires_grad_(False)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    reference_model = deepcopy(model)
    reference_model.attn_layers.layers[0][1].to_v.requires_grad_(True)
    options = dict(tbptt_steps=100, ttt_module_paths=(VALUE_PATH,), episodic_mem_len=episodic, ttt_lr=0.05)
    torch.manual_seed(42)
    wrapper = XLAutoregressiveWrapper(model, **options)
    torch.manual_seed(42)
    reference = XLAutoregressiveWrapper(reference_model, **options)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [6, 7, 8, 9, 10, 11, 12, 13, 14]])

    if method == "forward":
        with torch.no_grad():
            actual = wrapper(inputs)
            expected = reference(inputs)
        actual.backward()
        expected.backward()
    else:
        actual = wrapper.generate(inputs, seq_len=2, temperature=0)
        expected = reference.generate(inputs, seq_len=2, temperature=0)

    torch.testing.assert_close(actual, expected)
    wrapped_path = ("net." if episodic else "") + VALUE_PATH
    actual_fast = wrapper.ttt_wrappers[wrapped_path].batch_params
    expected_fast = reference.ttt_wrappers[wrapped_path].batch_params
    for name, parameter in actual_fast.items():
        assert parameter.requires_grad
        assert torch.isfinite(parameter).all()
        torch.testing.assert_close(parameter, expected_fast[name])
    assert not torch.allclose(actual_fast["weight"][0], original[VALUE_PATH + ".weight"])

    for (name, parameter), reference_parameter in zip(model.named_parameters(), reference_model.parameters()):
        # Wrapping changes the parameter path, not the underlying base parameter.
        original_name = name.replace("to_v.module.", "to_v.")
        torch.testing.assert_close(parameter, original[original_name], rtol=0, atol=0)
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        else:
            assert parameter.grad is None
    if freeze != "none":
        assert not adapted.weight.requires_grad
