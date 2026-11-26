from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from x_transformers import Decoder, TransformerWrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper


VALUE_PATH = "attn_layers.layers.0.1.to_v"
QUERY_PATH = "attn_layers.layers.0.1.to_q"
KEY_PATH = "attn_layers.layers.0.1.to_k"


def make_model():
    return TransformerWrapper(
        num_tokens=16, max_seq_len=4, max_mem_len=4,
        attn_layers=Decoder(dim=16, depth=1, heads=2, attn_dim_head=8),
    )


def make_wrapper(model, paths, low_rank=False, episodic=0):
    return XLAutoregressiveWrapper(
        model, tbptt_steps=100, ttt_module_paths=paths,
        ttt_low_rank_paths=((VALUE_PATH, 4),) if low_rank else (),
        episodic_mem_len=episodic, ttt_lr=0.05, ttt_wd=0.1,
    )


@pytest.mark.parametrize("case", ["repeated", "alias", "low_rank", "distinct_targets"])
@pytest.mark.parametrize("create_graph", [False, True])
@pytest.mark.parametrize("episodic", [0, 2])
def test_each_distinct_ttt_mapping_applies_one_sgd_update(case, create_graph, episodic):
    torch.manual_seed(16)
    if case == "repeated":
        paths = (VALUE_PATH, VALUE_PATH, VALUE_PATH)
        expected_paths = ((VALUE_PATH, VALUE_PATH),)
    elif case == "alias":
        paths = (VALUE_PATH, (VALUE_PATH, VALUE_PATH))
        expected_paths = ((VALUE_PATH, VALUE_PATH),)
    elif case == "low_rank":
        paths = (VALUE_PATH,)
        expected_paths = ((VALUE_PATH, VALUE_PATH),)
    else:
        paths = ((QUERY_PATH, KEY_PATH), (QUERY_PATH, VALUE_PATH))
        expected_paths = paths
    wrapper = make_wrapper(make_model(), paths, case == "low_rank", episodic)
    prefix = "net." if episodic else ""
    expected_paths = tuple((prefix + source, prefix + target) for source, target in expected_paths)
    wrapper.init_ttt(2)
    inputs = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]])
    labels = torch.tensor([[2, 3, 4, 5], [7, 8, 9, 10]])
    logits, intermediates = wrapper.net(inputs, return_intermediates=True)
    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none").mean(-1).sum()
    expected_parameters = {}
    for source, target in expected_paths:
        source_params = tuple(wrapper.ttt_wrappers[source].batch_params.values())
        target_params = wrapper.ttt_wrappers[target].batch_params
        grads = torch.autograd.grad(loss, source_params, retain_graph=True, create_graph=create_graph)
        expected_parameters[target] = {
            name: parameter - wrapper.ttt_lr * gradient - wrapper.ttt_lr * wrapper.ttt_wd * parameter
            for (name, parameter), gradient in zip(target_params.items(), grads)
        }

    wrapper.update_ttt(logits, labels, create_graph=create_graph, intermediates=intermediates)

    assert wrapper.ttt_paths_map == expected_paths
    for target, parameters in expected_parameters.items():
        for name, expected in parameters.items():
            actual = wrapper.ttt_wrappers[target].batch_params[name]
            torch.testing.assert_close(actual, expected)
            assert torch.isfinite(actual).all()


@pytest.mark.parametrize("low_rank", [False, True])
@pytest.mark.parametrize("episodic", [0, 2])
def test_redundant_ttt_configuration_matches_unique_training(low_rank, episodic):
    torch.manual_seed(16)
    model = make_model()
    duplicate_model = deepcopy(model)
    torch.manual_seed(42)
    unique = make_wrapper(model, () if low_rank else (VALUE_PATH,), low_rank, episodic)
    torch.manual_seed(42)
    duplicated = make_wrapper(duplicate_model, (VALUE_PATH, VALUE_PATH), low_rank, episodic)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [6, 7, 8, 9, 10, 11, 12, 13, 14]])

    expected = unique(inputs)
    actual = duplicated(inputs)
    expected.backward()
    actual.backward()

    torch.testing.assert_close(actual, expected)
    for parameter, reference in zip(duplicated.parameters(), unique.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad)
