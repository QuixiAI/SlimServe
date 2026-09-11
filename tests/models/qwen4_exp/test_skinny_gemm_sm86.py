"""SM86 skinny decode GEMM parity: every planned (N, K) shape at decode token
counts 1..32 against fp32 torch, plus the fallback for unplanned shapes."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.skinny_gemm_sm86 import (
    MAX_M,
    SM86_SKINNY_PLANS,
    plan_for,
    skinny_gemm,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("shape", sorted(SM86_SKINNY_PLANS))
@pytest.mark.parametrize("m", [1, 3, 8, 16, 17, 24, 32])
def test_planned_shapes_match_fp32_reference(shape, m):
    n, k = shape
    cfg = plan_for(shape, m)
    if cfg is None:
        pytest.skip("bucket routed to cuBLAS")
    assert k % (cfg.split_k * cfg.block_k) == 0
    torch.manual_seed(0)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / k**0.5
    ref = x.float() @ w.float().t()
    out = skinny_gemm(x, w, cfg)
    assert out.dtype == torch.bfloat16 and out.shape == (m, n)
    # bf16 output rounding dominates: 2e-2 relative to the row scale (fp8/quant contract bound).
    scale = ref.abs().amax().clamp_min(1e-6)
    assert ((out.float() - ref).abs().amax() / scale).item() < 2e-2


def test_custom_op_falls_back_outside_the_plan():
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1000, 2560, dtype=torch.bfloat16, device="cuda")  # not a planned N
    out = torch.ops.vllm.qwen4_exp_skinny_gemm_sm86(x, w)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w))
    x_big = torch.randn(MAX_M + 1, 2560, dtype=torch.bfloat16, device="cuda")
    w_planned = torch.randn(320, 2560, dtype=torch.bfloat16, device="cuda")
    out = torch.ops.vllm.qwen4_exp_skinny_gemm_sm86(x_big, w_planned)
    torch.testing.assert_close(out, torch.nn.functional.linear(x_big, w_planned))


def test_bucket_lookup():
    # (320, 2560) went to cuBLAS below M=16 in the first sweep; the
    # single-launch kernel wins at every M (skinny_sweep_w8.out).
    assert plan_for((320, 2560), 3) is not None and plan_for((320, 2560), 24) is not None
    assert plan_for((4096, 2560), 33) is None and plan_for((1000, 2560), 3) is None


@pytest.mark.parametrize("m", [1, 3, 8, 16, 24, 32])
def test_split_k_is_reentrant_and_matches(m):
    """Single-launch split-K (last-CTA reduce) on the mixer shape: two
    consecutive calls must both match the reference (counters reset)."""
    from vllm.models.qwen4_exp.nvidia.skinny_gemm_sm86 import SkinnyCfg

    torch.manual_seed(1)
    x = torch.randn(m, 10240, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(336, 10240, dtype=torch.bfloat16, device="cuda") / 100.0
    ref = x.float() @ w.float().t()
    cfg = SkinnyCfg(32, 256, 4)
    for _ in range(3):
        out = skinny_gemm(x, w, cfg)
        scale = ref.abs().amax().clamp_min(1e-6)
        assert ((out.float() - ref).abs().amax() / scale).item() < 2e-2


@pytest.mark.parametrize("shape", sorted(SM86_SKINNY_PLANS))
@pytest.mark.parametrize("m", [1, 3, 8, 16, 24, 32])
def test_w8_matches_dequantized_reference(shape, m):
    from vllm.models.qwen4_exp.nvidia.skinny_gemm_sm86 import (
        dequantize_w8,
        plan_for_w8,
        quantize_w8,
        skinny_gemm,
    )

    torch.manual_seed(0)
    n, k = shape
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w8, scale = quantize_w8(w)
    wd = dequantize_w8(w8, scale)
    assert (wd.float() - w.float()).abs().max() <= (w.float().abs().max() / 127.0) * 1.01
    cfg = plan_for_w8(shape, m)
    assert cfg is not None
    y = skinny_gemm(x, w8, cfg, scale=scale)
    ref = x.float() @ wd.float().t()
    torch.testing.assert_close(y.float(), ref, atol=2e-2 * ref.abs().max().item(), rtol=2e-2)
    y_op = torch.ops.vllm.qwen4_exp_skinny_gemm_w8_sm86(x, w8, scale)
    torch.testing.assert_close(y_op, y)


def test_w8_large_m_paths_match():
    from vllm.models.qwen4_exp.nvidia.skinny_gemm_sm86 import dequantize_w8, quantize_w8

    torch.manual_seed(0)
    for shape in ((2560, 1536), (62080, 2560)):
        n, k = shape
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        w8, scale = quantize_w8(w)
        x = torch.randn(70, k, dtype=torch.bfloat16, device="cuda")
        y = torch.ops.vllm.qwen4_exp_skinny_gemm_w8_sm86(x, w8, scale)
        ref = x.float() @ dequantize_w8(w8, scale).float().t()
        torch.testing.assert_close(y.float(), ref, atol=2e-2 * ref.abs().max().item(), rtol=2e-2)


def test_process_weights_after_loading_quantizes(monkeypatch):
    import vllm.envs as envs
    from vllm.models.qwen4_exp.nvidia.skinny_gemm_sm86 import Qwen4ExpSkinnyLinearMethodSM86

    monkeypatch.setattr(envs, "VLLM_QWEN4_EXP_SKINNY_W8", True)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.randn(2560, 1536, dtype=torch.bfloat16, device="cuda"), requires_grad=False)
    Qwen4ExpSkinnyLinearMethodSM86().process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.int8 and layer.weight.shape == (2560, 1536)
    assert layer.weight_scale_sm86.shape == (2560, 12)
    x = torch.randn(3, 1536, dtype=torch.bfloat16, device="cuda")
    y = Qwen4ExpSkinnyLinearMethodSM86().apply(layer, x)
    assert y.shape == (3, 2560) and torch.isfinite(y.float()).all()
