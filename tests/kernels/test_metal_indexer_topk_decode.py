# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle for the native decode indexer top-k (dsv4_indexer_topk_decode).

Decode twin of test_metal_indexer_topk_prefill: one token per row, each row
carrying its own block-table row. Covers the direct 1,024-wide sort and the
512-way streaming merge above it, which is the path where only the first
IDXTK_KEEP ranks stay valid.

Run directly: .venv/bin/python tests/kernels/test_metal_indexer_topk_decode.py
"""
import torch

try:  # collected by pytest in CI; hand-run on the serving box without it
    import pytest

    pytestmark = pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires Apple Metal (MPS)",
    )
except ModuleNotFoundError:
    pass

DEV = "mps"
H, D = 64, 128
TOPK = 512
BS = 256  # kv block size


def _lut() -> torch.Tensor:
    lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return torch.nan_to_num(lut, nan=0.0).to(DEV)


def _cache(n_blocks: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    cache = torch.randint(0, 255, (n_blocks, BS, 132), dtype=torch.uint8)
    cache[..., :128] &= 0x7E
    cache[0, ::3, 5] = 0x7F  # e4m3 NaN codes the writer never emits
    cache[1 % n_blocks, 1::4, 77] = 0xFF
    scales = (torch.rand(n_blocks, BS, 1) * 0.5 + 0.5).to(torch.float32)
    cache[..., 128:132] = scales.view(torch.uint8).reshape(n_blocks, BS, 4)
    return cache.to(DEV)


def _reference(q, w, cache, bt_row, n: int, k: int, lut) -> torch.Tensor:
    j = torch.arange(n, device=DEV)
    packed = cache[bt_row[j // BS].long(), j % BS]
    kval = lut[packed[..., :128].long()]
    kscale = packed[..., 128:].contiguous().view(torch.float32).reshape(n)
    score = torch.einsum("hd,kd->hk", q.float(), kval)
    logits = torch.einsum("hk,h->k", torch.relu(score), w) * kscale
    return torch.topk(logits, min(k, n)).indices.cpu()


def _run(width: int, cands: list[int], seed: int) -> None:
    from vllm.quixicore.ops import quixicore_ops

    lut = _lut()
    tokens = len(cands)
    blocks_per_row = (width + BS - 1) // BS
    cache = _cache(blocks_per_row * tokens, seed)
    torch.manual_seed(seed + 1)
    q = (torch.randn(tokens, H, D, dtype=torch.float16, device=DEV) * 0.3).contiguous()
    w = (torch.rand(tokens, H, dtype=torch.float32, device=DEV) + 0.1).contiguous()
    # each decode row owns a disjoint block-table row
    bt = torch.arange(blocks_per_row * tokens, dtype=torch.int32, device=DEV)
    bt = bt.reshape(tokens, blocks_per_row).contiguous()
    cand = torch.tensor(cands, dtype=torch.int32, device=DEV)
    k_eff = min(TOPK, width)
    out = torch.full((tokens, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_decode(q, w, cache, bt, cand, out, width, k_eff)
    out2 = torch.full((tokens, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_decode(q, w, cache, bt, cand, out2, width, k_eff)
    torch.mps.synchronize()
    assert torch.equal(out, out2), "decode indexer top-k is not deterministic"
    for t, n in enumerate(cands):
        ref = _reference(q[t], w[t], cache, bt[t], n, k_eff, lut)
        got = out[t, : ref.numel()].long().cpu()
        assert torch.equal(got, ref), f"row {t}: top-k order differs"
        assert (out[t, min(k_eff, n):] == -1).all().item(), f"row {t}: pad not -1"


def test_decode_direct_sort_width() -> None:
    # every candidate count fits the 1,024-wide direct sort
    _run(width=1024, cands=[1024, 700, 513, 512, 17, 1], seed=0)


def test_decode_streaming_merge() -> None:
    # above 1,024 candidates the kernel streams 512-way merges; k_eff stays
    # at IDXTK_KEEP, the only k the host allows past the direct width
    _run(width=1537, cands=[1537, 1518, 1025], seed=3)


def main() -> None:
    test_decode_direct_sort_width()
    print("decode indexer top-k: direct sort exact")
    test_decode_streaming_merge()
    print("decode indexer top-k: streaming merge exact")


if __name__ == "__main__":
    main()
