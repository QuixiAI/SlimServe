# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 (e4m3) main-KV parity for the QSA sparse-gather kernel.

The 8x RTX 3090 profile stores the main QSA KV as float8_e4m3fn (unit scale)
and the Triton gather kernel decodes it arithmetically in-register, since
SM86 has no fp8 conversion instruction. Restored 2026-09-02 with the fp8
main-KV path (removed 2026-08-27, commit de0b2af8a).
"""

import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
def test_qsa_sparse_paged_attention_fp8_kv_matches_bf16():
    """FP8 main-KV dequant path parity against the BF16 kernel."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    torch.manual_seed(11)
    num_tokens, num_heads, kv_heads, head_dim = 12, 8, 1, 256
    page, pages, topk = 64, 32, 128
    q = torch.randn(
        num_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(pages, page, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(pages, device="cuda", dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    idx = torch.randint(
        0, pages * page, (num_tokens, topk), device="cuda", dtype=torch.int32
    )

    out_bf16 = qsa_sparse_paged_attention(q, k, v, idx, block_table, token_to_req)
    k8 = k.to(torch.float8_e4m3fn)
    v8 = v.to(torch.float8_e4m3fn)
    out_fp8 = qsa_sparse_paged_attention(q, k8, v8, idx, block_table, token_to_req)
    # Reference for the fp8 path: run the bf16 kernel on the dequantized cache
    # so the only difference is the in-kernel conversion.
    out_ref = qsa_sparse_paged_attention(
        q, k8.to(torch.bfloat16), v8.to(torch.bfloat16), idx, block_table, token_to_req
    )
    # The e4m3 decode is bit-exact over all 256 codes; the split-merge path
    # may round the normalizer at a slightly different point between compiled
    # variants, so allow sub-ULP drift.
    assert torch.allclose(out_fp8.float(), out_ref.float(), atol=1e-4, rtol=0)
    # And the quantization error itself stays small.
    err = (out_fp8.float() - out_bf16.float()).abs().max().item()
    assert err < 0.35, err


@requires_cuda
def test_e4m3_decode_is_exact_over_all_codes():
    """Every e4m3fn byte decodes to torch's own value (NaN codes to NaN)."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    codes = torch.arange(256, device="cuda", dtype=torch.uint8)
    ref = codes.view(torch.float8_e4m3fn).to(torch.bfloat16)
    # Drive the decoder through the kernel: one query, one head, values laid
    # out so each attended row returns exactly one code; a one-hot softmax
    # over a single index reads the value back unchanged.
    head_dim = 256
    k = torch.zeros(1, 256, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[0, :, 0, 0] = ref
    v8 = v.to(torch.float8_e4m3fn)
    v8.view(torch.uint8)[0, :, 0, 0] = codes  # keep NaN codes as NaN bytes
    k8 = k.to(torch.float8_e4m3fn)
    q = torch.zeros(1, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(1, device="cuda", dtype=torch.int32)
    finite = ~torch.isnan(ref)
    for i in torch.nonzero(finite).flatten().tolist():
        idx = torch.full((1, 1), i, device="cuda", dtype=torch.int32)
        out = qsa_sparse_paged_attention(q, k8, v8, idx, block_table, token_to_req)
        assert out[0, 0, 0].item() == ref[i].item(), (i, out[0, 0, 0].item(), ref[i].item())
