# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-pinned (UVA-gathered) PLE n-gram table parity with a device table."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpHostNGramEmbedding

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.bfloat16])
def test_host_gather_matches_device_lookup(dtype):
    torch.manual_seed(7)
    vocab, dim = 65536, 160
    with torch.device("cuda"):
        emb = Qwen4ExpHostNGramEmbedding(vocab, dim, dtype)
    src = torch.randn(vocab, dim, dtype=torch.bfloat16).to(dtype)
    emb.weight.data.copy_(src)

    ids = torch.randint(0, vocab, (257, 16), device="cuda", dtype=torch.int64)
    out = emb(ids)
    assert out.shape == (257, 16, dim)
    assert out.dtype == dtype
    ref = src[ids.reshape(-1).cpu()].reshape(257, 16, dim).cuda()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


@requires_cuda
def test_host_gather_replays_inside_cuda_graph():
    vocab, dim = 4096, 160
    with torch.device("cuda"):
        emb = Qwen4ExpHostNGramEmbedding(vocab, dim, torch.float8_e4m3fn)
    src = torch.randn(vocab, dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    emb.weight.data.copy_(src)

    ids = torch.randint(0, vocab, (64, 16), device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(2):
            emb(ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = emb(ids)
    ids.copy_(torch.randint(0, vocab, (64, 16), device="cuda", dtype=torch.int64))
    graph.replay()
    torch.cuda.synchronize()
    ref = src[ids.reshape(-1).cpu()].reshape(64, 16, dim).cuda()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


@requires_cuda
def test_qsa_sparse_paged_attention_fp8_kv_matches_bf16():
    """FP8 main-KV dequant path parity against the BF16 kernel."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    torch.manual_seed(11)
    num_tokens, num_heads, kv_heads, head_dim = 12, 8, 1, 256
    page, pages, topk = 64, 32, 128
    q = torch.randn(num_tokens, num_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
    k = torch.randn(pages, page, kv_heads, head_dim, device="cuda",
                    dtype=torch.bfloat16)
    v = torch.randn_like(k)
    block_table = torch.arange(pages, device="cuda",
                               dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(num_tokens, device="cuda", dtype=torch.int32)
    idx = torch.randint(0, pages * page, (num_tokens, topk), device="cuda",
                        dtype=torch.int32)

    out_bf16 = qsa_sparse_paged_attention(q, k, v, idx, block_table,
                                          token_to_req)
    k8 = k.to(torch.float8_e4m3fn)
    v8 = v.to(torch.float8_e4m3fn)
    out_fp8 = qsa_sparse_paged_attention(q, k8, v8, idx, block_table,
                                         token_to_req)
    # Reference for the fp8 path: run bf16 kernel on the dequantized cache so
    # the only difference is in-kernel conversion.
    out_ref = qsa_sparse_paged_attention(
        q, k8.to(torch.bfloat16), v8.to(torch.bfloat16), idx, block_table,
        token_to_req)
    # The e4m3 decode is bit-exact over all 256 codes; the split-merge path
    # may round the normalizer at a slightly different point between compiled
    # variants, so allow sub-ULP drift.
    assert torch.allclose(out_fp8.float(), out_ref.float(), atol=1e-4, rtol=0)
    # And the quantization error itself stays small.
    err = (out_fp8.float() - out_bf16.float()).abs().max().item()
    assert err < 0.35, err


def test_ple_host_table_is_shared_across_instances(monkeypatch):
    """Two ranks' tables must alias one /dev/shm segment: writes through one
    instance are visible to the other's gather, and host RAM is paid once."""
    import os

    from vllm.models.qwen4_exp.nvidia import ple_layer
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpHostNGramEmbedding

    vocab, dim = 64, 32
    path = f"/dev/shm/slimserve_ple_{vocab}x{dim}_float8_e4m3fn"
    if os.path.exists(path):
        os.unlink(path)
    try:
        a = Qwen4ExpHostNGramEmbedding(vocab, dim, torch.float8_e4m3fn)
        b = Qwen4ExpHostNGramEmbedding(vocab, dim, torch.float8_e4m3fn)
        assert os.path.exists(path), "shared segment was not created"
        src = torch.randn(vocab, dim, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        a.weight.data.copy_(src)
        # b never wrote; its view must alias a's pages.
        assert torch.equal(
            b.weight.data.view(torch.uint8), src.view(torch.uint8)
        ), "second instance does not alias the shared segment"
        ids = torch.arange(vocab, device="cuda", dtype=torch.int64)
        out = b(ids)
        ref = src.to(torch.bfloat16).to("cuda")
        assert torch.allclose(out.to(torch.bfloat16), ref.to(out.dtype).to(torch.bfloat16))
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_ple_host_table_falls_back_to_private_pinned(monkeypatch):
    """When the shared segment cannot be mapped, the layer keeps working
    with a per-rank pinned copy."""
    from vllm.models.qwen4_exp.nvidia import ple_layer
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpHostNGramEmbedding

    monkeypatch.setattr(ple_layer, "_shared_pinned_table", lambda *a: None)
    emb = Qwen4ExpHostNGramEmbedding(64, 32, torch.float8_e4m3fn)
    assert emb.weight.data.is_pinned()
