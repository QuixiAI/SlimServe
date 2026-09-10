# SPDX-License-Identifier: Apache-2.0
"""Unchanged sparse-MLA gates for the opt-in owned tensor-core implementation."""

import math

import pytest
import torch

from benchmarks.glm5_next_sparse_tc_candidate import sparse_tc_nope


def reference(q, cache, table, indices, lengths, scale):
    outputs = []
    bs = cache.shape[1]
    for row, length in enumerate(lengths):
        logical = indices[row, :length]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            outputs.append(torch.zeros_like(q[row], dtype=torch.float32))
            continue
        keys = cache[table[row, logical // bs].long(), logical % bs].float()
        scores = torch.matmul(q[row].float(), keys.T) * scale
        outputs.append(torch.softmax(scores, dim=-1) @ keys)
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale_width", [256, 512])
@pytest.mark.parametrize("split", [32, 64, 128])
@pytest.mark.parametrize(
    "rows,heads,bs",
    [(r, h, 576) for r in (1, 2, 4, 8, 16, 32) for h in (8, 16)]
    + [(8, 8, 64), (8, 16, 64)],
)
@torch.no_grad()
def test_sparse_tc_reference_native_and_changed_graph(
    rows, heads, bs, split, scale_width
):
    from vllm.quixicore import quixicore_ops as qc

    torch.manual_seed(12200 + rows + heads + bs)
    pages, width = 64, 2080
    backing = torch.full((pages, 3, bs, 512), 7.0, device="cuda", dtype=torch.bfloat16)
    cache = backing[:, 1]
    cache.normal_(std=0.5)
    cache_before = backing.clone()
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    table = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    indices = torch.arange(width, device="cuda", dtype=torch.int32).repeat(rows, 1)
    tlen = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    # The registered model scales by its256-wide original QK head, not
    # the512-wide absorbed latent. Also retain the legacy test's512 scale.
    scale = 1 / math.sqrt(scale_width)

    def candidate():
        return sparse_tc_nope(q, cache, table, indices, tlen, scale, split=split)

    candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = candidate()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        choices = (0, 1, 37, 1000, 2048)
        for replay in range(5):
            q.normal_(std=0.2)
            lengths = [choices[(row + replay) % len(choices)] for row in range(rows)]
            tlen.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            for row, length in enumerate(lengths):
                table[row].copy_(torch.randperm(pages, device="cuda").int())
                # Positive entries beyond tlen must be ignored; holes before
                # tlen must be skipped without terminating the selected list.
                indices[row].copy_(
                    torch.randperm(pages * bs, device="cuda")[:width].int()
                )
                indices[row, 5:length:7] = -1
            indices_before = indices.clone()
            graph.replay()
            expected = reference(q, cache, table, indices, lengths, scale)
            native = qc.mla_decode_bf16_sparse_nope(
                q,
                cache,
                table,
                indices,
                tlen,
                bs,
                scale,
                128,
                cache.stride(0) * cache.element_size(),
            )
            # Existing independent-reference normalized-max gate plus the
            # umbrella BF16 pointwise gate. Neither threshold is loosened.
            peak = expected.abs().max().item()
            error = (actual.float() - expected).abs().max().item()
            assert error == 0 if peak == 0 else error / peak < 5e-3
            torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.002)
            # Existing partitioned-versus-unpartitioned NoPE comparison gate.
            assert (actual.float() - native.float()).abs().max().item() < 1e-3
            assert torch.equal(backing, cache_before)
            assert torch.equal(indices, indices_before)
            for row, length in enumerate(lengths):
                if length == 0:
                    assert torch.count_nonzero(actual[row]).item() == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
