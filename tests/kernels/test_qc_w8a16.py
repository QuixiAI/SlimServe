# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp8 weight-only (W8A16) skinny GEMM: pack/dequant round trip, GEMM parity
against the dequantized weights, and the linear method's load-time path."""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")

SHAPES = [(4096, 6144), (4096, 1536), (1536, 4096), (4096, 512), (512, 8192), (1536, 1024), (4096, 128)]


def _quant(w):
    amax = w.float().abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / 448.0
    wq = (w.float() / scale[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return wq, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("k,n", SHAPES)
def test_pack_dequant_round_trip(k, n):
    torch.manual_seed(k + n)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    wq, scale = _quant(w)
    packed = qc.w8a16_pack(wq.contiguous())
    back = qc.w8a16_dequant(packed, (scale * 256.0).float().contiguous(), n, k)
    torch.testing.assert_close(back.float(), wq.float() * scale[:, None], rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 7, 32, 100, 128])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("cfg", [0, 1, 4])
def test_w8a16_gemm_matches_dequantized_reference(m, k, n, cfg):
    torch.manual_seed(m * 7 + k + n)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    bias = torch.randn(n, device="cuda").to(torch.bfloat16)
    wq, scale = _quant(w)
    packed = qc.w8a16_pack(wq.contiguous())
    sc = (scale * 256.0).float().contiguous()
    ref = torch.nn.functional.linear(x.float(), wq.float() * scale[:, None], bias.float())
    out = qc.w8a16_gemm(x, packed, sc, bias, n, 256, cfg).float()
    rel = ((out - ref).abs() / (ref.abs() + 1e-2)).max().item()
    assert rel < 1e-2, f"max rel err {rel:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_linear_method_quantizes_at_load_and_dispatches():

    from vllm.model_executor.layers.quantization.qc_w8a16 import qc_w8a16_linear

    torch.manual_seed(0)
    n, k = 1536, 4096
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    wq, scale = _quant(w)
    packed = qc.w8a16_pack(wq.contiguous())
    sc = (scale * 256.0).float().contiguous()
    ref_w = wq.float() * scale[:, None]
    x = (torch.randn(64, k, device="cuda") * 0.5).to(torch.bfloat16)   # skinny kernel path
    out = qc_w8a16_linear(x, packed, sc, None, n, k).float()
    ref = torch.nn.functional.linear(x.float(), ref_w)
    assert ((out - ref).abs() / (ref.abs() + 1e-2)).max().item() < 1e-2
    # Prefill path: dequantize + cuBLAS. cuBLAS's bf16 GEMM reduces in
    # reduced precision at K=4096 (0.3-0.7 relative near zero crossings vs
    # an fp32 reference), so compare against the identical bf16 computation.
    x = (torch.randn(300, k, device="cuda") * 0.5).to(torch.bfloat16)
    out = qc_w8a16_linear(x, packed, sc, None, n, k)
    w_deq = qc.w8a16_dequant(packed, sc, n, k)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, w_deq), rtol=0, atol=0)
    torch.testing.assert_close(w_deq.float(), ref_w, rtol=1e-2, atol=1e-3)
    # torch.compile-visible op
    assert hasattr(torch.ops.vllm, "qc_w8a16_linear")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_every_gemm_operand_must_sit_on_the_activation_device():
    k, n, m = 4096, 512, 7
    torch.manual_seed(1)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    bias = torch.randn(n, device="cuda").to(torch.bfloat16)
    wq, scale = _quant(w)
    packed = qc.w8a16_pack(wq.contiguous())
    sc = (scale * 256.0).float().contiguous()
    ref = qc.w8a16_gemm(x, packed, sc, bias, n, 256, 0)
    for moved in ("packed", "sc", "bias"):
        args = dict(packed=packed, sc=sc, bias=bias)
        args[moved] = args[moved].to("cuda:1")
        with pytest.raises(RuntimeError, match="first operand's device"):
            qc.w8a16_gemm(x, args["packed"], args["sc"], args["bias"], n, 256, 0)
    # With every operand on cuda:1 the same call runs there and matches.
    other = qc.w8a16_gemm(
        x.to("cuda:1"), packed.to("cuda:1"), sc.to("cuda:1"), bias.to("cuda:1"), n, 256, 0
    )
    assert torch.equal(ref.cpu(), other.cpu())
