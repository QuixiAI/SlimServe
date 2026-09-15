# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QuixiCore FP8-weight decode GEMM (csrc/quixicore/tm_cuda/fp8_decode_gemm.cuh):
x (bf16, M <= 16) @ dequant(W)^T with e4m3 weights and 128x128 fp32 block scales
or one fp32 scale per row, where dequant(W) = bf16(scale * W); and the
block-FP8 custom op around it."""

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytest.importorskip("vllm._quixicore_C")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_decode_gemm_fp8()),
    reason="needs CUDA and the QuixiCore decode_gemm_fp8 binding",
)
DEV = "cuda"
SHAPES = [
    (6144, 4096),  # dense gate_up
    (4096, 3072),  # dense down
    (1024, 4096),  # shared-expert gate_up
    (4096, 512),  # shared-expert down
    (4096, 1536),  # DSA q_b
    (4096, 4096),  # DSA o_proj
    (2336, 4096),  # fused_qkv_a (N tail inside a scale group)
]


def block_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    rows = (n + 127) // 128
    wp = torch.zeros(rows * 128, k, device=w.device, dtype=torch.float32)
    wp[:n] = w.float()
    wb = wp.view(rows, 128, k // 128, 128).permute(0, 2, 1, 3)
    amax = wb.abs().amax(dim=(2, 3), keepdim=True).clamp(min=1e-12)
    s = amax / 448.0
    q = (wb / s).clamp(-448, 448).permute(0, 2, 1, 3).reshape(rows * 128, k)[:n]
    return q.contiguous().to(torch.float8_e4m3fn), s.squeeze(3).squeeze(2).contiguous()


def dequant_bf16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    rows = s.shape[0]
    qp = torch.zeros(rows * 128, k, device=q.device, dtype=torch.float32)
    qp[:n] = q.float()
    d = qp.view(rows, 128, k // 128, 128) * s[:, None, :, None]
    return d.view(rows * 128, k)[:n].to(torch.bfloat16)


@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 3, 8, 13, 16])
def test_matches_dequantized_reference(n: int, k: int, m: int) -> None:
    torch.manual_seed(n * 31 + k + m)
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = block_quant(w)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    ref = x.float() @ dequant_bf16(q, s).float().t()
    out = quixicore_ops.decode_gemm_fp8(x, q, s)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    assert ((out.float() - ref).abs() / scale).max().item() < 2**-7


def channel_quant(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    s = amax / 448.0
    q = (w.float() / s).clamp(-448, 448)
    return q.contiguous().to(torch.float8_e4m3fn), s.reshape(-1).contiguous()


@pytest.mark.parametrize(
    "n,k",
    [
        (4096, 1536),  # a dense shape, the 32-row config
        (1024, 4096),  # the 8-row config
        (38720, 4096),  # GLM-5.3-Flash's TP4 vocabulary shard (the LM head)
    ],
)
@pytest.mark.parametrize("m", [1, 4, 16])
def test_channel_scales_match_dequantized_reference(n: int, k: int, m: int) -> None:
    torch.manual_seed(n + k + m)
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = channel_quant(w)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    ref = x.float() @ (q.float() * s[:, None]).to(torch.bfloat16).float().t()
    out = quixicore_ops.decode_gemm_fp8(x, q, s)
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    assert ((out.float() - ref).abs() / scale).max().item() < 2**-7
    # The [N, 1] layout compressed-tensors keeps for channel scales is the same.
    out2 = quixicore_ops.decode_gemm_fp8(x, q, s.reshape(n, 1))
    assert torch.equal(out, out2)


def test_bias_and_fp32_output() -> None:
    torch.manual_seed(0)
    n, k, m = 4096, 1536, 5
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = block_quant(w)
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(n, device=DEV, dtype=torch.float32)
    ref = x.float() @ dequant_bf16(q, s).float().t() + bias
    out = quixicore_ops.decode_gemm_fp8(x, q, s, bias, fp32_out=True)
    assert out.dtype == torch.float32
    scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
    assert ((out - ref).abs() / scale).max().item() < 2**-8


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_shared_memory_setup_follows_device_switches() -> None:
    # M16/N1024 selects the >48-KiB opt-in kernel. A process-wide bool
    # incorrectly treats setup on device0 as setup on device1 too.
    for device in (0, 1, 0):
        with torch.cuda.device(device):
            w = torch.randn(1024, 512, device="cuda", dtype=torch.bfloat16) * 0.02
            q, s = block_quant(w)
            x = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ dequant_bf16(q, s).float().t()
            out = quixicore_ops.decode_gemm_fp8(x, q, s)
            scale = ref.abs().amax(1, keepdim=True).clamp_min(1e-6)
            assert ((out.float() - ref).abs() / scale).max().item() < 2**-7


def test_rejects_unsupported_shapes_and_dtypes() -> None:
    w = (torch.randn(4096, 1536, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = block_quant(w)
    x = torch.randn(4, 1536, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):  # M > 16
        quixicore_ops.decode_gemm_fp8(
            torch.randn(17, 1536, device=DEV, dtype=torch.bfloat16), q, s
        )
    with pytest.raises(RuntimeError):  # wrong scale shape
        quixicore_ops.decode_gemm_fp8(x, q, s[:, :1].contiguous())
    with pytest.raises(RuntimeError):  # a per-row scale short of a row
        quixicore_ops.decode_gemm_fp8(x, q, torch.ones(4095, device=DEV))
    with pytest.raises(RuntimeError):  # bf16 weight
        quixicore_ops.decode_gemm_fp8(x, w, s)
    with pytest.raises(RuntimeError):  # N below the gate
        q2, s2 = block_quant(
            (torch.randn(512, 1536, device=DEV) * 0.02).to(torch.bfloat16)
        )
        quixicore_ops.decode_gemm_fp8(x, q2, s2)


def test_block_fp8_op_kernel_and_cutlass_branches() -> None:
    """The custom op takes the kernel at M <= 16 and reproduces the stock
    CUTLASS w8a8 path above, for 2-D and 3-D inputs."""
    from vllm.model_executor.layers.utils import (
        _cutlass_block_fp8_linear,
        maybe_quixicore_fp8_block_linear,
    )

    torch.manual_seed(1)
    n, k = 4096, 1536
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = block_quant(w)
    wd = dequant_bf16(q, s).float()
    for m in (1, 16, 17, 200):
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        out = maybe_quixicore_fp8_block_linear(x, q, s, None)
        assert out is not None and out.shape == (m, n)
        if m <= 16:
            ref = x.float() @ wd.t()
            scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            assert ((out.float() - ref).abs() / scale).max().item() < 2**-7
        else:
            assert torch.equal(out, _cutlass_block_fp8_linear(x, q, s))
    x3 = torch.randn(2, 3, k, device=DEV, dtype=torch.bfloat16)
    assert maybe_quixicore_fp8_block_linear(x3, q, s, None).shape == (2, 3, n)
    assert (
        maybe_quixicore_fp8_block_linear(x3, w, s, None) is None
    )  # bf16 weight: not ours


def test_channel_fp8_op_kernel_and_cutlass_branches() -> None:
    """The channel-scale op takes the kernel at M <= 16 (from the scheme's
    [K, N] weight view) and the per-token quant + CUTLASS path above."""
    from vllm.model_executor.layers.utils import (
        _cutlass_channel_fp8_linear,
        maybe_quixicore_fp8_channel_linear,
    )

    torch.manual_seed(2)
    n, k = 4096, 1536
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.bfloat16)
    q, s = channel_quant(w)
    scheme_weight, scheme_scale = q.t(), s.reshape(n, 1)  # as the scheme keeps them
    wd = (q.float() * s[:, None]).to(torch.bfloat16).float()
    for m in (1, 16, 17, 200):
        x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
        out = maybe_quixicore_fp8_channel_linear(x, scheme_weight, scheme_scale, None)
        assert out is not None and out.shape == (m, n)
        ref = x.float() @ wd.t()
        scale = ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        if m <= 16:
            assert ((out.float() - ref).abs() / scale).max().item() < 2**-7
        else:
            assert torch.equal(out, _cutlass_channel_fp8_linear(x, q, s.reshape(n, 1)))
            # W8A8: the per-token e4m3 activations round at 2^-4.
            assert ((out.float() - ref).abs() / scale).max().item() < 2**-4
    assert maybe_quixicore_fp8_channel_linear(x, w.t(), scheme_scale, None) is None
