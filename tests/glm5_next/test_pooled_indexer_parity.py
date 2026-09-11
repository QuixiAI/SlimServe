# SPDX-License-Identifier: Apache-2.0
"""Paged pooled selection against the independent Transformers implementation."""

import pytest
import torch
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

from vllm.model_executor.layers.glm5_next_indexer import _ROW_DIM, _pooled_select
from vllm.model_executor.layers.glm5_next_pool_cache import update_pool_cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [1, 3, 4, 61, 67, 133])
@pytest.mark.parametrize("page_padding", [0, 2])
@pytest.mark.parametrize("compact", [False, True])
@torch.no_grad()
def test_pooled_indexer_parity(length, page_padding, compact):
    # 133 tokens actually prune pools at topk=64. The original 61-token
    # smoke test selected everything, and could not detect broken scores.
    torch.manual_seed(0)
    dev = "cuda"
    cfg = Glm5NextTextConfig(
        hidden_size=256,
        q_lora_rank=64,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=64,
        index_kpool=4,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        qk_rope_head_dim=0,
    )
    ref = Glm5NextTextIndexer(cfg, layer_idx=0).to(dev).to(torch.bfloat16)
    for parameter in ref.parameters():
        parameter.normal_(0, 0.2)
    ref.k_norm.weight.fill_(1.0)
    ref.k_norm.bias.zero_()
    x = torch.randn(1, length, cfg.hidden_size, device=dev, dtype=torch.bfloat16)
    qr = torch.randn(1, length, cfg.q_lora_rank, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(1, length, dtype=torch.bool, device=dev)
    ref_idx = ref(x, qr, mask, None)
    q = ref.wq_b(qr[0]).view(length, cfg.index_n_heads, cfg.index_head_dim)
    k = torch.nn.functional.layer_norm(
        ref.wk(x[0]).float(),
        (cfg.index_head_dim,),
        ref.k_norm.weight.float(),
        ref.k_norm.bias.float(),
        1e-6,
    ).to(torch.bfloat16)
    gate = torch.nn.functional.linear(x[0], ref.index_kpool_compress_gate)
    weights = (ref.weights_proj(x[0]).float() * cfg.index_n_heads**-0.5).contiguous()
    ape = ref.index_kpool_compress_ape.float().contiguous()
    block_size = 64 if compact else 16
    nblocks = (length + block_size - 1) // block_size
    # Packed cross-layer pages include unrelated layers in their stride;
    # physical block order need not be chronological.
    backing = torch.full(
        (nblocks, 1 + page_padding, block_size, _ROW_DIM),
        float("nan"),
        device=dev,
        dtype=torch.bfloat16,
    )
    cache = backing[:, 0]
    physical = torch.randperm(nblocks, device=dev)
    rows = torch.cat([k, gate], -1)
    for logical in range(nblocks):
        count = min(block_size, length - logical * block_size)
        cache[physical[logical], :count] = rows[
            logical * block_size : logical * block_size + count
        ]
    bt = physical.to(torch.int32).view(1, -1)
    if compact:
        cache = torch.full(
            (nblocks, 1 + page_padding, block_size, 64),
            float("nan"), device=dev, dtype=torch.bfloat16,
        )[:, 0]
        positions = torch.arange(length, device=dev, dtype=torch.int64)
        slots = physical[positions // block_size] * block_size + positions % block_size
        update_pool_cache(rows.contiguous(), slots, ape, cache)
    row_req = torch.zeros(length, dtype=torch.int32, device=dev)
    visible = torch.arange(1, length + 1, dtype=torch.int32, device=dev)
    ksel = cfg.index_topk // cfg.index_kpool
    width = (cfg.index_topk + cfg.index_kpool - 1 + 31) // 32 * 32
    out = torch.full((length, width), -1, dtype=torch.int32, device=dev)
    logits = torch.empty((length, (length + 3) // 4), dtype=torch.float32, device=dev)
    _pooled_select(
        q.contiguous(),
        weights,
        ape,
        cache,
        bt,
        row_req,
        visible,
        logits,
        logits.shape[1],
        block_size,
        cfg.index_head_dim**-0.5,
        ksel,
        out,
        cfg.index_kpool,
    )
    for row, (expected, actual) in enumerate(zip(ref_idx[0].tolist(), out.tolist())):
        expected_set = {v for v in expected if v >= 0}
        actual_set = {v for v in actual if v >= 0}
        assert actual_set == expected_set, (
            row,
            expected_set - actual_set,
            actual_set - expected_set,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.no_grad()
def test_pooled_logits_graph_dynamic_lengths():
    """Mixed requests, inactive programs, strided pages and >128 pool tiles."""
    from vllm.model_executor.layers.glm5_next_indexer import _pooled_logits_kernel

    torch.manual_seed(42)
    rows, heads, dim, kp, bs, length, max_pools = 4, 32, 128, 4, 64, 8196, 4096
    nblocks = (length + bs - 1) // bs
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(rows, heads, device="cuda") * heads**-0.5
    ape = torch.randn(kp, dim, device="cuda")
    backing = torch.full(
        (rows * nblocks, 3, bs, _ROW_DIM),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache = backing[:, 1]
    tokens = torch.randn(
        rows, nblocks * bs, _ROW_DIM, device="cuda", dtype=torch.bfloat16
    )
    physical = torch.randperm(rows * nblocks, device="cuda").view(rows, nblocks)
    cache[physical.flatten()] = tokens.view(rows * nblocks, bs, _ROW_DIM)
    bt = physical.to(torch.int32)
    row_req = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int32)
    visible = torch.tensor([0, 4, 64, 132], device="cuda", dtype=torch.int32)
    output = torch.full((rows, max_pools), float("nan"), device="cuda")

    def launch():
        _pooled_logits_kernel[(rows, 128)](
            q,
            w,
            ape,
            cache,
            bt,
            row_req,
            visible,
            output,
            max_pools,
            bt.stride(0),
            cache.stride(0),
            dim**-0.5,
            BLOCK_SIZE=bs,
            H=heads,
            D=dim,
            KP=kp,
            ROW=_ROW_DIM,
            BLOCK_P=16,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for lengths in ([0, 4, 64, 132], [3, 67, 1000, length]):
        visible.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        output.fill_(float("nan"))
        graph.replay()
        for row, (request, count) in enumerate(zip([2, 0, 3, 1], lengths)):
            pools = count // kp
            if not pools:
                assert output[row].isnan().all()
                continue
            data = tokens[request, : pools * kp].float().view(pools, kp, _ROW_DIM)
            probs = (data[:, :, dim:] + ape).softmax(dim=1)
            keys = (probs * data[:, :, :dim]).sum(dim=1).to(torch.bfloat16).float()
            scores = (keys @ q[row].float().T * dim**-0.5).relu()
            expected = (scores * w[row]).sum(dim=1)
            # FP32 exp/reduction order can round a pooled key to the adjacent
            # BF16 value. Test the independent oracle at BF16 precision;
            # the selection tests above still require exact token sets.
            torch.testing.assert_close(
                output[row, :pools], expected, atol=1e-3, rtol=2e-3
            )
