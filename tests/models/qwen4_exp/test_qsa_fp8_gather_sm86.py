# SPDX-License-Identifier: Apache-2.0
"""fp8 paged QSA sparse gather (E10 launch shape) against an fp32 dense
reference over the selected rows, at decode and prefill row counts and
with padded (-1) indices."""
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

PAGE, KVH, HD, QH, TOPK = 16, 2, 256, 6, 2048


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 3, 24, 96, 600])
@pytest.mark.parametrize("ptr_table", [True, False])
def test_fp8_gather_matches_fp32_reference(rows, ptr_table):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    seq = 3000
    pages = (seq + PAGE - 1) // PAGE
    num_blocks = max(4096, max(1, rows // 3) * pages + 1)
    k8 = (torch.randn(num_blocks, PAGE, KVH, HD, device=dev) * 0.5).to(torch.float8_e4m3fn)
    v8 = (torch.randn(num_blocks, PAGE, KVH, HD, device=dev) * 0.5).to(torch.float8_e4m3fn)
    nreq = max(1, rows // 3)
    q = torch.randn(rows, QH, HD, dtype=torch.bfloat16, device=dev)
    block_table = torch.randperm(num_blocks, device=dev)[: nreq * pages].view(nreq, pages).to(torch.int32)
    tok = torch.stack([torch.randperm(seq, device=dev)[:TOPK] for _ in range(rows)]).to(torch.int32)
    tok[:, -5:] = -1
    token_to_req = (torch.arange(rows, device=dev) * nreq // rows).to(torch.int32)
    out = torch.empty(rows, QH, HD, dtype=torch.bfloat16, device=dev)
    page_offsets = block_table.to(torch.int64) * (PAGE * KVH * HD) if ptr_table else None
    qsa_sparse_paged_attention(q, k8, v8, tok, block_table, token_to_req, out, page_offsets=page_offsets)
    kf, vf = k8.float(), v8.float()
    check = range(rows) if rows <= 24 else list(range(0, rows, rows // 12))
    for r in check:
        req = int(token_to_req[r])
        sel = tok[r][tok[r] >= 0].long()
        pg = block_table[req][sel // PAGE].long()
        off = sel % PAGE
        K, V = kf[pg, off], vf[pg, off]
        for h in range(QH):
            kvh = h // (QH // KVH)
            s = (q[r, h].float() @ K[:, kvh].t()) * (HD**-0.5)
            ref = torch.softmax(s, 0) @ V[:, kvh]
            torch.testing.assert_close(out[r, h].float(), ref, atol=1e-2 * ref.abs().max().item() + 1e-4, rtol=1e-2)
