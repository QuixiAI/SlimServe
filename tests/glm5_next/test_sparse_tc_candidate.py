# SPDX-License-Identifier: Apache-2.0
"""Unchanged sparse-MLA gates for the opt-in owned tensor-core implementation."""

import math

import pytest
import torch

from benchmarks.glm5_next_sparse_tc_candidate import sparse_tc_nope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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


def _served_split(q, cache) -> int:
    """The split the serving dispatch selects for this shape and cache dtype
    on this device (0 when it keeps the native path)."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla.quixicore_mla_sparse import _sparse_tc_split

    return _sparse_tc_split(True, q, cache, SimpleNamespace(num_prefills=0))


def _skip_if_tile_does_not_fit(fn, *, served: bool):
    """Run `fn` once; when this device cannot host the kernel's tile, skip if
    the configuration is not served here and fail if it is.

    sm_120 has 99 KB of shared memory per block against SM80's 164 KB, so the
    SPLIT=128 tile (144 KB of gathered latents plus queries) and the fp8
    branch's int32 assembly (147 KB even at SPLIT=64) cannot load here. Those
    configurations are not served on this platform - the dispatch picks the
    64-wide bf16 tile - so the suite skips them rather than reporting a
    failure that no code path can reach. A configuration the dispatch DOES
    select on this device (`served`) that stops fitting is a real regression,
    so its OutOfResources propagates as a failure instead of a skip.
    """
    from triton.runtime.errors import OutOfResources

    try:
        return fn()
    except OutOfResources as err:
        if served:
            raise
        pytest.skip(
            f"tile needs {err.required} B of shared memory, device has "
            f"{err.limit} B; not a served configuration on this device"
        )


def test_served_configurations_fail_rather_than_skip_when_the_tile_does_not_fit():
    from triton.runtime.errors import OutOfResources

    def too_big():
        raise OutOfResources(147456, 101376, "shared memory")

    with pytest.raises(OutOfResources):
        _skip_if_tile_does_not_fit(too_big, served=True)
    with pytest.raises(pytest.skip.Exception):
        _skip_if_tile_does_not_fit(too_big, served=False)
    assert _skip_if_tile_does_not_fit(lambda: 7, served=True) == 7


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
    # The registered model scales by its 256-wide original QK head, not
    # the 512-wide absorbed latent. Also retain the legacy test's 512 scale.
    scale = 1 / math.sqrt(scale_width)

    def candidate():
        return sparse_tc_nope(q, cache, table, indices, tlen, scale, split=split)

    _skip_if_tile_does_not_fit(candidate, served=_served_split(q, cache) == split)
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
            # Existing partitioned-versus-unpartitioned NoPE comparison gate:
            # bf16-identical up to one rounding flip. Both kernels accumulate
            # in fp32 in different orders, so an fp32 value on a bf16 midpoint
            # (rows 32, split 32, replay 4: 0.272461 between 0.271484 and
            # 0.273438, each half an ulp from the reference) rounds to opposite
            # neighbours one ulp (2^-7 x magnitude) apart; anything wider than
            # that, or than 1e-3 where an ulp is smaller, is a real divergence.
            gap = (actual.float() - native.float()).abs()
            one_ulp = torch.finfo(torch.bfloat16).eps * native.float().abs()
            assert bool(
                (gap <= torch.maximum(one_ulp, torch.full_like(gap, 1e-3))).all()
            )
            assert torch.equal(backing, cache_before)
            assert torch.equal(indices, indices_before)
            for row, length in enumerate(lengths):
                if length == 0:
                    assert torch.count_nonzero(actual[row]).item() == 0
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


# ---- fp8 (e4m3) storage decoded in-kernel (2026-09-11) ----


@pytest.mark.parametrize("rows,heads", [(1, 8), (4, 16), (32, 16)])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
@torch.no_grad()
def test_sparse_tc_fp8_matches_native_fp8(rows, heads, kv_scale):
    import math

    from vllm.quixicore import quixicore_ops as qc
    from vllm.quixicore.sparse_mla_tc import sparse_tc_nope

    torch.manual_seed(7 + rows + heads)
    bs, pages, max_topk = 576, 40, 2080
    kv = (torch.randn(pages, bs, 512, device="cuda") * 0.5).to(torch.bfloat16)
    data = (kv.float() / kv_scale).to(torch.float8_e4m3fn).view(torch.uint8)
    q = (torch.randn(rows, heads, 512, device="cuda") * 0.2).to(torch.bfloat16)
    bt = torch.arange(pages, device="cuda", dtype=torch.int32).repeat(rows, 1)
    idx = torch.full((rows, max_topk), -1, device="cuda", dtype=torch.int32)
    for r in range(rows):
        n = int(torch.randint(1, 2048, (1,)))
        idx[r, :n] = torch.randperm(pages * bs, device="cuda")[:n].to(torch.int32)
    tlen = qc.sparse_topk_tlen(idx)
    scale = 1.0 / math.sqrt(512)
    native = qc.mla_decode_fp8_sparse_nope(
        q, data.reshape(-1), bt, idx, tlen, bs, scale, kv_scale, 128
    )
    tc = _skip_if_tile_does_not_fit(
        lambda: sparse_tc_nope(
            q, data, bt, idx, tlen, scale, split=64, kv_scale=kv_scale
        ),
        served=_served_split(q, data) == 64,
    )
    err = (
        tc.float() - native.float()
    ).abs().max().item() / native.float().abs().max().item()
    assert err < 5e-3, err
