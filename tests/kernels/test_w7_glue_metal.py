# SPDX-License-Identifier: Apache-2.0
"""Metal decode-glue kernels (GLM-5.3-Flash W7) against their torch chains:
one-launch KV metadata (block-table gather + slot mappings), the mamba
"align" tail-block gather, the indexer pack (LayerNorm | gate | scaled
weights), int64-slot paged row insert, the shared-add-folded q2_K MoE sum
(bit-exact with `shared + T(sum)`), and the GEMV column-slice output."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


def _qc(name: str):
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has(name):
        pytest.skip(f"{name} not built")
    return quixicore_ops


# ---------------------------------------------------------------- kv meta
def _kv_meta_ref(idx_mapping, qsl, positions, srcs, block_sizes, enabled,
                 num_reqs, num_reqs_padded, num_tokens, num_tokens_padded):
    tables = []
    for src in srcs:
        dst = torch.zeros(
            num_reqs_padded, src.shape[1], dtype=torch.int32, device=DEV
        )
        dst[:num_reqs] = src[idx_mapping[:num_reqs].long()]
        tables.append(dst)
    slots = torch.full(
        (len(srcs), num_tokens_padded), -1, dtype=torch.int64, device=DEV
    )
    seg = torch.zeros(num_tokens, dtype=torch.int64, device=DEV)
    for r in range(1, num_reqs):
        seg[int(qsl[r]) :] += 1
    req = idx_mapping.long()[seg]
    pos = positions[:num_tokens]
    for g, (src, bs, en) in enumerate(zip(srcs, block_sizes, enabled)):
        bi = pos // bs if en else torch.zeros_like(pos)
        bo = pos % bs
        bn = src[req, bi].long()
        slots[g, :num_tokens] = bn * bs + bo
    return tables, slots


@pytest.mark.parametrize(
    "lens,num_reqs_padded,pad_tokens",
    [([1], 1, 0), ([1, 1, 1], 4, 1), ([1, 5, 1, 300], 4, 5), ([2048], 2, 0)],
)
def test_kv_meta_prepare(lens, num_reqs_padded, pad_tokens):
    q = _qc("kv_meta_prepare")
    torch.manual_seed(0)
    max_reqs = 8
    block_sizes = [64, 64, 16, 64, 1, 128]
    enabled = [True, True, True, False, True, True]
    max_len = 4096
    G = len(block_sizes)
    cols = [max_len // bs + 1 for bs in block_sizes]
    srcs = [
        torch.randint(0, 5000, (max_reqs, c), dtype=torch.int32, device=DEV)
        for c in cols
    ]
    dsts = [
        torch.full((max_reqs, c), 7, dtype=torch.int32, device=DEV) for c in cols
    ]
    num_reqs = len(lens)
    perm = torch.randperm(max_reqs)[:num_reqs].to(torch.int32)
    idx_mapping = perm.to(DEV)
    qsl_cpu = torch.tensor(
        [0] + [int(v) for v in torch.tensor(lens).cumsum(0)], dtype=torch.int32
    )
    qsl = qsl_cpu.to(DEV)
    num_tokens = int(qsl_cpu[-1])
    num_tokens_padded = num_tokens + pad_tokens
    positions = torch.randint(
        0, max_len, (num_tokens_padded + 3,), dtype=torch.int64, device=DEV
    )
    params = torch.tensor(
        [
            [s.stride(0), d.stride(0), s.shape[1], bs, int(en), 0, 0, 0]
            for s, d, bs, en in zip(srcs, dsts, block_sizes, enabled)
        ],
        dtype=torch.int32,
        device=DEV,
    )
    slots = torch.full((G, 4096), 99, dtype=torch.int64, device=DEV)
    q.kv_meta_prepare(
        idx_mapping, qsl, positions, params, slots, srcs, dsts,
        num_reqs, num_reqs_padded, num_tokens, num_tokens_padded,
    )
    tables, ref_slots = _kv_meta_ref(
        idx_mapping, qsl_cpu, positions, srcs, block_sizes, enabled,
        num_reqs, num_reqs_padded, num_tokens, num_tokens_padded,
    )
    for g in range(G):
        assert torch.equal(dsts[g][:num_reqs_padded], tables[g]), f"group {g} table"
    assert torch.equal(slots[:, :num_tokens_padded], ref_slots)
    if pad_tokens:
        assert bool((slots[:, num_tokens:num_tokens_padded] == -1).all())


# ------------------------------------------------------- mamba last blocks
@pytest.mark.parametrize("ncols", [1, 3])
def test_mamba_last_blocks(ncols):
    q = _qc("mamba_last_blocks")
    torch.manual_seed(1)
    R, cols, bs = 5, 40, 64
    bt = torch.randint(0, 1000, (R, cols), dtype=torch.int32, device=DEV)
    seq_lens = torch.tensor([0, 1, 64, 65, 2000], dtype=torch.int32, device=DEV)
    out = torch.empty(R, ncols, dtype=torch.int32, device=DEV)
    q.mamba_last_blocks(bt, seq_lens, out, bs)
    start = ((seq_lens - 1) // bs).clamp_(min=0)
    offs = torch.arange(ncols, device=DEV, dtype=torch.int32)
    idx = (start.unsqueeze(1) + offs).long()
    assert torch.equal(out, torch.gather(bt, 1, idx))


# ------------------------------------------------------------ indexer pack
@pytest.mark.parametrize("T", [1, 7])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_indexer_pack(T, dtype):
    q = _qc("glm5_indexer_pack")
    torch.manual_seed(2)
    D, H, eps, scale = 128, 32, 1e-6, 32**-0.5
    fused = (torch.randn(T, 2 * D + H, device=DEV) * 2.0).to(dtype)
    w = torch.randn(D, device=DEV) * 0.5 + 1.0
    b = torch.randn(D, device=DEV) * 0.1
    packed = torch.empty(T, 2 * D, dtype=dtype, device=DEV)
    weights = torch.empty(T, H, dtype=torch.float32, device=DEV)
    q.glm5_indexer_pack(fused, w, b, packed, weights, D, eps, scale)
    k_ref = torch.nn.functional.layer_norm(
        fused[:, :D].float(), (D,), w, b, eps
    ).to(dtype)
    ref_packed = torch.cat([k_ref, fused[:, D : 2 * D]], dim=-1)
    ref_w = fused[:, 2 * D :].float() * scale
    assert torch.equal(packed[:, D:], ref_packed[:, D:])
    diff = (packed[:, :D].float() - ref_packed[:, :D].float()).abs()
    assert diff.max().item() <= 0.05, diff.max().item()
    assert (diff == 0).float().mean().item() >= 0.95
    assert torch.allclose(weights, ref_w, rtol=1e-6, atol=1e-6)


# --------------------------------------------------- int64 slot row insert
def test_paged_row_insert_i64_matches_i32():
    q = _qc("paged_row_insert")
    torch.manual_seed(3)
    blocks, bs, dim, T = 6, 16, 512, 9
    rows = torch.randn(T, dim, device=DEV).to(torch.bfloat16)
    slots32 = torch.tensor(
        [-1, 3, 17, 95, 5, 40, 41, 42, 63], dtype=torch.int32, device=DEV
    )
    c32 = torch.zeros(blocks, bs, dim, dtype=torch.bfloat16, device=DEV)
    c64 = torch.zeros(blocks, bs, dim, dtype=torch.bfloat16, device=DEV)
    q.paged_row_insert(rows, c32, slots32)
    q.paged_row_insert(rows, c64, slots32.to(torch.int64))
    assert torch.equal(c32, c64)
    assert torch.equal(c64[95 // bs, 95 % bs], rows[3])


# --------------------------------------------- q2_K sum with folded add
def _finite_half_bytes(*shape):
    vals = torch.rand(*shape, dtype=torch.float32) * 0.5 + 0.25
    vals = vals.to(torch.float16)
    return vals.view(torch.uint8).reshape(*shape[:-1], shape[-1] * 2)


def _make_q2_k(E, n_rows, k):
    nb = k // 256
    blocks = torch.randint(0, 256, (E, n_rows, nb, 84), dtype=torch.uint8)
    blocks[..., 80:84] = _finite_half_bytes(E, n_rows, nb, 2)
    return blocks.reshape(E, n_rows, nb * 84).contiguous().to(DEV)


@pytest.mark.parametrize("tokens", [1, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_moe_sum_accumulate_is_shared_plus_sum(tokens, dtype):
    q = _qc("ggml_moe_a8_vec_sum")
    torch.manual_seed(4)
    E, n, k, topk = 8, 64, 1024, 8
    w = _make_q2_k(E, n, k)
    ids = torch.randint(0, E, (tokens, topk), dtype=torch.int32, device=DEV)
    tw = (torch.rand(tokens, topk, dtype=torch.float32) * 2.0 + 0.05).to(DEV)
    x = (torch.randn(tokens * topk, k, dtype=dtype) * 0.1).to(DEV)
    shared = (torch.randn(tokens, n) * 3.0).to(dtype).to(DEV)
    plain = torch.empty(tokens, n, dtype=dtype, device=DEV)
    q.ggml_moe_a8_vec_sum(x, w, ids, tw, topk, 10, n, tokens, plain)
    ref = shared + plain
    acc = shared.clone()
    q.ggml_moe_a8_vec_sum(
        x, w, ids, tw, topk, 10, n, tokens, acc, accumulate=True
    )
    assert torch.equal(acc, ref)


# ------------------------------------------------ GEMV column-slice output
def _make_q8_0(rows, k):
    nb = k // 32
    blocks = torch.randint(0, 256, (rows, nb, 34), dtype=torch.uint8)
    blocks[..., 0:2] = _finite_half_bytes(rows, nb, 1)
    return blocks.reshape(rows, nb * 34).contiguous().to(DEV)


@pytest.mark.parametrize("batch", [1, 2, 3, 8])
def test_gemv_out_slice(batch):
    q = _qc("ggml_mul_mat_vec_a8")
    torch.manual_seed(5)
    k, n1, n2 = 1024, 96, 64
    w1, w2 = _make_q8_0(n1, k), _make_q8_0(n2, k)
    x = (torch.randn(batch, k) * 0.2).to(torch.bfloat16).to(DEV)
    r1 = q.ggml_mul_mat_vec_a8(w1, x, 8, n1).clone()
    r2 = q.ggml_mul_mat_vec_a8(w2, x, 8, n2).clone()
    out = torch.full(
        (batch, n1 + n2), float("nan"), dtype=torch.bfloat16, device=DEV
    )
    got1 = q.ggml_mul_mat_vec_a8(w1, x, 8, n1, out=out[:, :n1])
    got2 = q.ggml_mul_mat_vec_a8(w2, x, 8, n2, out=out[:, n1:])
    assert got1.data_ptr() == out.data_ptr()
    assert got2.data_ptr() == out[:, n1:].data_ptr()
    assert torch.equal(out[:, :n1], r1)
    assert torch.equal(out[:, n1:], r2)
    # A dense caller output is written in place too.
    dense = torch.empty(batch, n1, dtype=torch.bfloat16, device=DEV)
    q.ggml_mul_mat_vec_a8(w1, x, 8, n1, out=dense)
    assert torch.equal(dense, r1)
