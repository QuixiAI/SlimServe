# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny bf16 GEMM (M <= 128, split-K, fp32 partials) parity vs an fp32
reference at the GLM-5.3 TP4 per-rank dense shapes."""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")

SHAPES = [(4096, 6144), (4096, 1536), (1536, 4096), (4096, 512), (512, 8192), (1536, 1024)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 7, 16, 32, 64, 100, 128])
@pytest.mark.parametrize("k,n", SHAPES)
@pytest.mark.parametrize("cfg", [0, 1])
def test_skinny_gemm_matches_fp32_reference(m, k, n, cfg):
    if cfg == 1 and m <= 64:
        pytest.skip("cfg 1 is the M > 64 tile")
    torch.manual_seed(m * 1000 + k + n)
    x = (torch.randn(m, k, device="cuda") * 0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    bias = torch.randn(n, device="cuda").to(torch.bfloat16)
    ref = torch.nn.functional.linear(x.float(), w.float(), bias.float())
    out = qc.skinny_gemm(x, w, bias, 256, cfg).float()
    rel = ((out - ref).abs() / (ref.abs() + 1e-2)).max().item()
    assert rel < 1e-2, f"max rel err {rel:.3e}"
    out_nb = qc.skinny_gemm(x, w, None, 128, cfg).float()
    ref_nb = torch.nn.functional.linear(x.float(), w.float())
    assert ((out_nb - ref_nb).abs() / (ref_nb.abs() + 1e-2)).max().item() < 1e-2
