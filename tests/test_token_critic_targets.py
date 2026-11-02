import pytest
import torch
import torch.nn.functional as F
from torch import nn

from x_transformers import Encoder, TransformerWrapper
from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper


def make_generator():
    net = TransformerWrapper(
        num_tokens = 4,
        max_seq_len = 4,
        use_abs_pos_emb = False,
        attn_layers = Encoder(dim = 8, depth = 1, heads = 2)
    )
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.token_emb.emb.weight[:, 0] = 1.
        net.to_logits.weight[0, 0] = 1000.
    return net


@pytest.mark.parametrize('self_critic', (False, True))
@pytest.mark.parametrize('mask_fraction', (0.5, 1.))
@pytest.mark.parametrize('only_train_critic', (False, True))
def test_critic_targets_describe_its_actual_input(
    self_critic,
    mask_fraction,
    only_train_critic
):
    torch.manual_seed(0)
    net = make_generator()
    critic = None
    if not self_critic:
        critic = make_generator()
        critic.to_logits = nn.Linear(8, 1)

    wrapper = NonAutoregressiveWrapper(
        net,
        mask_id = 3,
        self_token_critic = self_critic,
        token_critic = critic,
        schedule = lambda times: torch.full_like(times, mask_fraction),
        no_replace_prob = 0.,
        random_token_prob = 0.,
        use_simple_mdlm_loss_weight = False
    )
    with torch.no_grad():
        wrapper.token_critic.to_logits.weight.zero_()
        wrapper.token_critic.to_logits.bias.fill_(2.)

    critic_inputs = []
    handle = wrapper.token_critic.register_forward_pre_hook(
        lambda module, args: critic_inputs.append(args[0].detach().clone())
    )
    original = torch.ones(2, 4, dtype = torch.long)
    try:
        losses = wrapper(
            original,
            only_train_critic = only_train_critic,
            generator_sample_temperature = 1.
        )
    finally:
        handle.remove()

    generated = critic_inputs[0]
    assert torch.equal(original, torch.ones_like(original))
    assert (generated == 0).sum().item() == int(mask_fraction * original.numel())
    expected_labels = (generated != original).float()
    expected_loss = F.binary_cross_entropy_with_logits(
        torch.full_like(expected_labels, 2.),
        expected_labels
    )
    torch.testing.assert_close(losses.critic_loss, expected_loss)
    losses.loss.backward()
    expected_gradient = torch.sigmoid(torch.tensor(2.)) - mask_fraction
    torch.testing.assert_close(
        wrapper.token_critic.to_logits.bias.grad.squeeze(),
        expected_gradient
    )
    if only_train_critic:
        assert losses.generator_loss is None
