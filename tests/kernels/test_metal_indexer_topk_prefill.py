# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle for the native prefill indexer top-k (dsv4_indexer_topk_prefill).

The kernel must reproduce the eager metal_indexer.py chain (request-local
candidate windows, deterministic tie order) on multi-request chunks.
Run directly: .venv/bin/python tests/kernels/test_metal_indexer_topk_prefill.py
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


def main() -> None:
    from vllm.quixicore.ops import quixicore_ops

    torch.manual_seed(0)
    # two requests in the chunk
    q_lens = [120, 80]
    k_lens = [300, 200]
    T = sum(q_lens)
    NK = sum(k_lens)

    q = (torch.randn(T, H, D, dtype=torch.float16, device=DEV) * 0.3).contiguous()
    weights = (torch.rand(T, H, dtype=torch.float32, device=DEV) + 0.1).contiguous()

    # 132-byte indexer cache: 128 e4m3 + fp32 scale
    n_blocks = 8
    cache = torch.randint(0, 255, (n_blocks, BS, 132), dtype=torch.uint8)
    cache[..., :128] &= 0x7E
    # Plant the two e4m3 NaN encodings the writer never emits (0x7F/0xFF,
    # unreachable through the &= 0x7E sanitize): the kernel's stale-slot
    # guard and the eager LUT must both decode them as 0.0, and the oracle
    # must prove that parity rather than avoid it.
    cache[0, ::3, 5] = 0x7F
    cache[1, 1::4, 77] = 0xFF
    scales = (torch.rand(n_blocks, BS, 1) * 0.5 + 0.5).to(torch.float32)
    cache[..., 128:132] = scales.view(torch.uint8).reshape(n_blocks, BS, 4)
    cache = cache.to(DEV)

    # block tables: request r's candidate j -> slot bt[r][j//BS]*BS + j%BS
    bt = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=DEV)

    cu = torch.tensor([0, k_lens[0], NK], dtype=torch.int32, device=DEV)
    ks_list, ke_list, req_list = [], [], []
    for r, ql in enumerate(q_lens):
        for t in range(ql):
            vis = min(3 * (t + 1), k_lens[r])  # causal-ish growth
            ks_list.append(int(cu[r]))
            ke_list.append(int(cu[r]) + vis)
            req_list.append(r)
    ks = torch.tensor(ks_list, dtype=torch.int32, device=DEV)
    ke = torch.tensor(ke_list, dtype=torch.int32, device=DEV)
    tok_req = torch.tensor(req_list, dtype=torch.int32, device=DEV)
    cand = (ke - ks).contiguous()
    k_eff = min(TOPK, NK)

    buf = torch.full((T, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_prefill(
        q, weights, cache, bt, tok_req, cand, buf, 1024, k_eff
    )
    buf2 = torch.full((T, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_prefill(
        q, weights, cache, bt, tok_req, cand, buf2, 1024, k_eff
    )
    torch.mps.synchronize()
    det = torch.equal(buf, buf2)
    print("determinism:", det)

    # eager reference (metal_indexer chain semantics, request-local columns)
    lut = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    )
    lut = torch.nan_to_num(lut, nan=0.0).to(DEV)
    mism = 0
    setdiff = 0
    for t in range(T):
        r = req_list[t]
        n = int(cand[t])
        if n == 0:
            continue
        j = torch.arange(n, device=DEV)
        blocks = bt[r][torch.div(j, BS, rounding_mode="floor")].to(torch.long)
        rows = cache[blocks, torch.remainder(j, BS)]
        k_vals = lut[rows[..., :128].to(torch.long)]
        k_scale = rows[..., 128:].contiguous().view(torch.float32).reshape(n)
        scores = torch.einsum("hd,kd->hk", q[t].float(), k_vals)
        logits = torch.einsum("hk,h->k", torch.relu(scores), weights[t]) * k_scale
        ke_row = min(k_eff, n)
        vals, idx = torch.topk(logits, ke_row)
        got = buf[t, :ke_row].to(torch.long).cpu()
        ref = idx.cpu()
        if not torch.equal(got, ref):
            mism += 1
            setdiff += len(set(ref.tolist()) ^ set(got[got >= 0].tolist())) // 2
        # pads beyond n candidates must be -1
        assert (buf[t, min(k_eff, n) :] == -1).all().item()
    print(
        f"rows compared {T}, exact-order mismatches {mism}, "
        f"set-membership diffs {setdiff}"
    )
    failed = []
    if not det:
        failed.append("determinism x2")
    if mism or setdiff:
        failed.append(f"mismatches={mism} setdiff={setdiff}")
    if failed:
        raise SystemExit(f"ORACLE MISMATCH: {failed}")

    # Cross the fixed-scratch boundary.  This exercises the native 512-way
    # streaming merge used for real request windows above 1,024 candidates.
    width = 1537
    rows = 2
    blocks_per_req = (width + BS - 1) // BS
    cache2 = torch.randint(0, 255, (blocks_per_req, BS, 132), dtype=torch.uint8)
    cache2[..., :128] &= 0x7E
    # Same NaN-code coverage as the first oracle block.
    cache2[0, ::5, 11] = 0x7F
    cache2[2, 2::7, 100] = 0xFF
    scales2 = torch.rand(blocks_per_req, BS, 1, dtype=torch.float32) * 0.5 + 0.5
    cache2[..., 128:132] = scales2.view(torch.uint8).reshape(blocks_per_req, BS, 4)
    cache2 = cache2.to(DEV)
    bt2 = torch.arange(blocks_per_req, dtype=torch.int32, device=DEV)[None, :]
    q2 = torch.randn(rows, H, D, dtype=torch.float16, device=DEV) * 0.3
    w2 = torch.rand(rows, H, dtype=torch.float32, device=DEV) + 0.1
    cand2 = torch.tensor([width, width - 19], dtype=torch.int32, device=DEV)
    req2 = torch.zeros(rows, dtype=torch.int32, device=DEV)
    got2 = torch.full((rows, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_prefill(
        q2, w2, cache2, bt2, req2, cand2, got2, width, TOPK
    )
    torch.mps.synchronize()
    lut2 = (
        torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    )
    lut2 = torch.nan_to_num(lut2, nan=0.0).to(DEV)
    for t, n in enumerate((width, width - 19)):
        j = torch.arange(n, device=DEV)
        packed = cache2[bt2[0, j // BS].long(), j % BS]
        kval = lut2[packed[..., :128].long()]
        kscale = packed[..., 128:].contiguous().view(torch.float32).reshape(n)
        score = torch.einsum("hd,kd->hk", q2[t].float(), kval)
        logits = torch.einsum("hk,h->k", torch.relu(score), w2[t]) * kscale
        ref = torch.topk(logits, TOPK).indices.cpu()
        assert torch.equal(got2[t].long().cpu(), ref)
    print("streaming native width 1537: exact")

    # Exercise the shipping profile's full compressed maximum without a
    # width-sized score tensor.  Zero queries make every score tie, so the
    # deterministic secondary key gives a cheap exact oracle [0, 512).
    max_width = 65536
    max_blocks = max_width // BS
    cache3 = torch.zeros((max_blocks, BS, 132), dtype=torch.uint8)
    scales3 = torch.ones((max_blocks, BS, 1), dtype=torch.float32)
    cache3[..., 128:132] = scales3.view(torch.uint8).reshape(max_blocks, BS, 4)
    cache3 = cache3.to(DEV)
    bt3 = torch.arange(max_blocks, dtype=torch.int32, device=DEV)[None, :]
    got3 = torch.full((1, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_prefill(
        torch.zeros((1, H, D), dtype=torch.float16, device=DEV),
        torch.ones((1, H), dtype=torch.float32, device=DEV),
        cache3,
        bt3,
        torch.zeros(1, dtype=torch.int32, device=DEV),
        torch.tensor([max_width], dtype=torch.int32, device=DEV),
        got3,
        max_width,
        TOPK,
    )
    torch.mps.synchronize()
    assert torch.equal(got3[0].cpu(), torch.arange(TOPK, dtype=torch.int32))
    print("streaming native width 65536: exact deterministic tie order")
    print("prefill indexer top-k oracle passed")


def test_metal_indexer_topk_prefill() -> None:
    main()


if __name__ == "__main__":
    main()
