# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.model_executor.layers.glm5_next_mhc_ops import load_lossless_mhc_fn


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_original_bf16_values_load_without_changing_the_parameter_storage(dtype):
    weight = torch.tensor([[0.0, -1.0, 0.125, 1e-20]], dtype=torch.bfloat16).to(dtype)
    param = torch.nn.Parameter(
        torch.zeros_like(weight, dtype=torch.bfloat16), requires_grad=False
    )
    pointer = param.data_ptr()
    load_lossless_mhc_fn(param, weight)
    assert param.data_ptr() == pointer
    assert torch.equal(param.float(), weight.float())


@pytest.mark.parametrize("value", [0.1, float("nan"), float("inf")])
def test_storage_loader_rejects_lossy_or_nonfinite_values_before_copying(value):
    param = torch.zeros(1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="alter checkpoint"):
        load_lossless_mhc_fn(param, torch.tensor([value], dtype=torch.float32))
    assert torch.equal(param, torch.zeros_like(param))


def test_storage_loader_rejects_broadcast_shapes_and_wrong_parameter_dtype():
    param = torch.zeros(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shapes differ"):
        load_lossless_mhc_fn(param, torch.ones(1, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="BF16 parameter"):
        load_lossless_mhc_fn(param.float(), torch.ones_like(param))
    with pytest.raises(ValueError, match="BF16/FP32 source"):
        load_lossless_mhc_fn(param, torch.ones(2, dtype=torch.float16))


def test_registered_fake_ops_keep_fp32_mixes_with_bf16_fn_storage():
    residual = torch.empty(2, 4, 4096, device="meta", dtype=torch.bfloat16)
    fn = torch.empty(24, 16384, device="meta", dtype=torch.bfloat16)
    scale = torch.empty(3, device="meta", dtype=torch.float32)
    base = torch.empty(24, device="meta", dtype=torch.float32)
    constants = (1e-5, 1e-6, 2.0, 20)
    post, comb, x = torch.ops.vllm.glm5_mhc_pre(residual, fn, scale, base, *constants)
    assert (post.shape, comb.shape, x.shape) == ((2, 4, 1), (2, 4, 4), (2, 4096))
    assert post.dtype == comb.dtype == torch.float32
    assert x.dtype == torch.bfloat16
    out = torch.ops.vllm.glm5_mhc_fused_post_pre(
        x, residual, post, comb, fn, scale, base, *constants
    )
    assert [t.shape for t in out] == [residual.shape, post.shape, comb.shape, x.shape]
    assert [t.dtype for t in out] == [
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.bfloat16,
    ]
