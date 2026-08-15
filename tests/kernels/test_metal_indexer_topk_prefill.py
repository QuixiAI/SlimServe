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
        q, weights, cache, bt, tok_req, cand, buf, 1024, k_eff)
    buf2 = torch.full((T, TOPK), -1, dtype=torch.int32, device=DEV)
    quixicore_ops.dsv4_indexer_topk_prefill(
        q, weights, cache, bt, tok_req, cand, buf2, 1024, k_eff)
    torch.mps.synchronize()
    det = torch.equal(buf, buf2)
    print("determinism:", det)

    # eager reference (metal_indexer chain semantics, request-local columns)
    lut = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
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
        assert (buf[t, min(k_eff, n):] == -1).all().item()
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
    print("prefill indexer top-k oracle passed")


def test_metal_indexer_topk_prefill() -> None:
    main()


if __name__ == "__main__":
    main()
