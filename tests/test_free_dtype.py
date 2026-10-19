import pytest
import torch

from x_transformers.free_transformer import BinaryMapper, FreeTransformer


DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64)


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('training', (False, True))
def test_binary_mapper_preserves_dtype(dtype, training):
    mapper = BinaryMapper(bits = 2).to(dtype).train(training)
    logits = torch.randn(2, 3, 2, dtype = dtype, requires_grad = True)
    output, loss = mapper(logits)

    assert output.dtype == dtype
    assert output.shape == (2, 3, 4)
    torch.testing.assert_close(output.sum(dim = -1), torch.ones(2, 3, dtype = dtype))
    assert torch.isfinite(output).all()
    assert torch.equal(output, output.round())
    assert torch.isfinite(loss)
    if training:
        weights = torch.arange(4, dtype = dtype)
        ((output * weights).sum() + loss).backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


def make_model(dtype):
    return FreeTransformer(
        num_tokens = 8,
        dim = 8,
        max_seq_len = 8,
        heads = 2,
        attn_dim_head = 8,
        dec_head_depth = 1,
        dec_tail_depth = 1,
        latent_bits = 2
    ).to(dtype)


@pytest.mark.parametrize('dtype', DTYPES)
def test_free_transformer_trains_with_model_dtype(dtype):
    torch.manual_seed(0)
    model = make_model(dtype)
    loss = model(torch.tensor([[1, 2, 3, 4]]))

    assert torch.isfinite(loss)
    loss.backward()
    assert model.from_latent_to_condition.weight.grad is not None
    assert torch.isfinite(model.from_latent_to_condition.weight.grad).all()


@pytest.mark.parametrize('dtype', DTYPES)
@pytest.mark.parametrize('cache_kv', (False, True))
@pytest.mark.parametrize('batched_latents', (False, True))
def test_free_transformer_integer_latents_use_projection_dtype(
    dtype,
    cache_kv,
    batched_latents
):
    model = make_model(dtype)
    latents = torch.tensor([1, 2]) if batched_latents else 1
    projection_inputs = []
    handle = model.from_latent_to_condition.register_forward_pre_hook(
        lambda module, args: projection_inputs.append(args[0].dtype)
    )
    try:
        generated = model.generate(
            torch.tensor([[1, 2], [2, 3]]),
            4,
            latents = latents,
            use_kv_cache = cache_kv
        )
    finally:
        handle.remove()

    assert generated.shape == (2, 4)
    assert generated.dtype == torch.long
    assert projection_inputs == [dtype]
