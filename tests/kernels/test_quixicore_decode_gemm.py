# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""bf16 decode GEMM (quixicore decode_gemm): x[M, K] @ w[N, K]^T for M <= 16 on
tensor cores with fp32 accumulation, against an fp32 reference on the
GLM-5.3-Flash backbone shapes, including the N tail (6288 = 196.5 tiles),
bias, fp32 output and the shape gate."""

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore.ops import quixicore_ops as qc  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not qc.has_decode_gemm():
    pytest.skip("QuixiCore build without decode_gemm", allow_module_level=True)

DEV = "cuda"
SHAPES = [
    (6288, 4096),  # KDA in_proj
    (4096, 2048),  # KDA o_proj
    (2048, 4096),  # DSA fused_qkv_a
    (4096, 1536),  # DSA q_b / wq_b
    (4096, 4096),  # DSA o_proj
    (4096, 512),  # shared-expert down
    (2064, 640),  # N tail + odd K multiple
]


def _check(out, ref):
    # bf16 output: 2^-8 relative to the row scale, plus fp32 accumulation order.
    scale = ref.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    assert ((out.float() - ref).abs() / scale).max().item() < 2**-7


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 3, 8, 13, 16])
def test_decode_gemm_matches_fp32_reference(N, K, M):
    torch.manual_seed(N * 7 + K * 3 + M)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(N, K, device=DEV, dtype=torch.bfloat16) * 0.05
    ref = x.float() @ w.float().t()
    out = qc.decode_gemm(x, w)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    _check(out, ref)


def test_decode_gemm_bias_and_fp32_output():
    torch.manual_seed(0)
    x = torch.randn(5, 2048, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4096, 2048, device=DEV, dtype=torch.bfloat16) * 0.05
    b = torch.randn(4096, device=DEV, dtype=torch.float32)
    ref = x.float() @ w.float().t() + b
    out32 = qc.decode_gemm(x, w, b, True)
    assert out32.dtype == torch.float32
    torch.testing.assert_close(out32, ref, rtol=1e-3, atol=1e-2)
    _check(qc.decode_gemm(x, w, b), ref)


def test_decode_gemm_rejects_unsupported_shapes():
    x = torch.randn(17, 4096, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        qc.decode_gemm(x, w)  # M > 16
    with pytest.raises(RuntimeError):
        qc.decode_gemm(x[:1, :4032].contiguous(), w[:, :4032].contiguous())  # K % 128
    with pytest.raises(RuntimeError):
        qc.decode_gemm(x[:1], w[:1024])  # N < 2048
    with pytest.raises(RuntimeError):
        qc.decode_gemm(x[:1].float(), w)  # dtype
