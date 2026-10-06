import pytest
import torch
from x_transformers import Decoder
from x_transformers.continuous import (
    ContinuousAutoregressiveWrapper,
    ContinuousTransformerWrapper,
)


def make_model(loss_type, absolute_positions):
    torch.manual_seed(263)
    net = ContinuousTransformerWrapper(
        dim_in=2,
        dim_out=2,
        max_seq_len=12,
        use_abs_pos_emb=absolute_positions,
        attn_layers=Decoder(dim=8, depth=1, heads=2, attn_dim_head=4),
    )
    return ContinuousAutoregressiveWrapper(net, use_l1_loss=loss_type == "l1")


def independent_loss(wrapper, inputs, lens, steps, loss_type):
    valid = lens > steps
    chosen = inputs[valid]
    lengths = lens[valid]
    torch.manual_seed(277)
    prefix_lengths = (
        torch.rand(chosen.shape[0]) * (lengths - steps)
    ).floor().long() + 1
    row_losses = []
    for row, prefix in zip(chosen, prefix_lengths.tolist()):
        # Each reference row is unpadded and retains its complete history.
        history = row[None, :prefix]
        predictions = []
        for _ in range(steps):
            prediction = wrapper.net(history)[:, -1:]
            predictions.append(prediction)
            history = torch.cat((history, prediction), 1)
        predictions = torch.cat(predictions, 1)[0]
        error = predictions - row[prefix : prefix + steps]
        row_losses.append(
            error.abs().mean() if loss_type == "l1" else error.square().mean()
        )
    return torch.stack(row_losses).mean()


@pytest.mark.parametrize("loss_type", ["mse", "l1"])
@pytest.mark.parametrize("absolute_positions", [False, True])
@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("length_mode", ["none", "lens", "mask"])
def test_rollout_retains_complete_prefix_and_valid_row_positions(
    loss_type, absolute_positions, steps, length_mode
):
    wrapper = make_model(loss_type, absolute_positions)
    inputs = torch.randn(3, 9, 2, requires_grad=True)
    lens = (
        torch.full((3,), 9)
        if length_mode == "none"
        else torch.tensor([steps, 9, steps + 3])
    )
    mask = torch.arange(9)[None, :] < lens[:, None]
    kwargs = (
        {"lens": lens}
        if length_mode == "lens"
        else {"mask": mask}
        if length_mode == "mask"
        else {}
    )
    torch.manual_seed(277)
    actual = wrapper(inputs, rollout_steps=steps, **kwargs)
    expected = independent_loss(wrapper, inputs, lens, steps, loss_type)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual, inputs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
    if length_mode != "none":
        assert torch.equal(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))
        assert torch.equal(actual_grad[0], torch.zeros_like(actual_grad[0]))
        changed = inputs.detach().clone()
        changed[~mask] += 20
        torch.manual_seed(277)
        torch.testing.assert_close(
            wrapper(changed, rollout_steps=steps, **kwargs), actual.detach()
        )
    actual.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("loss_type", ["mse", "l1"])
@pytest.mark.parametrize("steps", [2, 3])
def test_minimum_valid_sequence_always_has_one_prefix_token(loss_type, steps):
    wrapper = make_model(loss_type, True)
    inputs = torch.randn(2, steps + 1, 2, requires_grad=True)
    lens = torch.full((2,), steps + 1)
    torch.manual_seed(277)
    actual = wrapper(inputs, rollout_steps=steps)
    expected = independent_loss(wrapper, inputs, lens, steps, loss_type)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("length_mode", ["none", "lens", "mask"])
def test_rollout_without_any_valid_prefix_reports_clear_error(steps, length_mode):
    wrapper = make_model("mse", True)
    inputs = torch.randn(2, steps, 2)
    lens = torch.full((2,), steps)
    kwargs = (
        {"lens": lens}
        if length_mode == "lens"
        else {"mask": torch.ones(2, steps, dtype=torch.bool)}
        if length_mode == "mask"
        else {}
    )
    with pytest.raises(ValueError, match="nonempty prefix"):
        wrapper(inputs, rollout_steps=steps, **kwargs)


def independent_general_mask_loss(wrapper, inputs, mask, steps, loss_type):
    # Enumerate legal cuts explicitly on the original timeline. Neither the
    # native alignment helper nor its vectorized eligibility calculation is used.
    legal_cuts = []
    rows = []
    for row_index, row_mask in enumerate(mask):
        valid_positions = row_mask.nonzero().flatten().tolist()
        last_valid_end = valid_positions[-1] + 1 if valid_positions else 0
        cuts = [
            prefix
            for prefix in range(1, inputs.shape[1] - steps + 1)
            if row_mask[prefix - 1]
            and prefix + steps <= last_valid_end
            and row_mask[prefix : prefix + steps].any()
        ]
        if cuts:
            rows.append(row_index)
            legal_cuts.append(cuts)
    torch.manual_seed(281)
    uniforms = torch.rand(len(rows)).tolist()
    errors = []
    valid_labels = []
    for row_index, cuts, uniform in zip(rows, legal_cuts, uniforms):
        prefix = cuts[int(uniform * len(cuts))]
        assert mask[row_index, prefix - 1]
        history = inputs[row_index : row_index + 1, :prefix]
        history_mask = mask[row_index : row_index + 1, :prefix]
        predictions = []
        for _ in range(steps):
            prediction = wrapper.net(history, mask=history_mask)[:, -1:]
            predictions.append(prediction)
            history = torch.cat((history, prediction), 1)
            history_mask = torch.cat(
                (history_mask, torch.ones(1, 1, dtype=torch.bool)), 1
            )
        prediction = torch.cat(predictions, 1)[0]
        difference = prediction - inputs[row_index, prefix : prefix + steps]
        errors.append(difference.abs() if loss_type == "l1" else difference.square())
        valid_labels.append(mask[row_index, prefix : prefix + steps])
    errors = torch.stack(errors)
    valid_labels = torch.stack(valid_labels)
    return errors[valid_labels].mean()


@pytest.mark.parametrize("loss_type", ["mse", "l1"])
@pytest.mark.parametrize("absolute_positions", [False, True])
@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("mask_kind", ["left_padding", "holes"])
def test_general_masks_choose_valid_queries_and_ignore_masked_input_values(
    loss_type, absolute_positions, steps, mask_kind
):
    wrapper = make_model(loss_type, absolute_positions)
    inputs = torch.randn(3, 9, 2, requires_grad=True)
    if mask_kind == "left_padding":
        mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, True],
                [False, False, True, True, True, True, True, True, True],
                [False, False, False, True, True, True, True, True, True],
            ]
        )
    else:
        mask = torch.tensor(
            [
                [False, False, False, False, False, False, False, False, True],
                [True, False, True, False, False, True, True, False, True],
                [False, True, False, True, True, False, True, True, True],
            ]
        )
    torch.manual_seed(281)
    actual = wrapper(inputs, rollout_steps=steps, mask=mask)
    expected = independent_general_mask_loss(wrapper, inputs, mask, steps, loss_type)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(actual, inputs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, inputs)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-5, atol=2e-6)
    assert torch.equal(actual_grad[~mask], torch.zeros_like(actual_grad[~mask]))
    assert torch.equal(actual_grad[0], torch.zeros_like(actual_grad[0]))
    changed = inputs.detach().clone()
    changed[~mask] += 20
    torch.manual_seed(281)
    torch.testing.assert_close(
        wrapper(changed, rollout_steps=steps, mask=mask), actual.detach()
    )
    actual.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in wrapper.parameters() if p.grad is not None
    )
