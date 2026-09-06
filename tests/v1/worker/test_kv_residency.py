# SPDX-License-Identifier: Apache-2.0
"""MainKVResidency: GPU hot window + host rows behind one offset table."""

import pytest
import torch

from vllm.v1.worker.gpu.kv_residency import PTR_SENTINEL, MainKVResidency

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

ROW = 8192  # bytes per block (13 layers' pages in production)
BS = 16  # tokens per block


def _res(num_blocks=12, gpu_rows=4):
    dev = torch.device("cuda")
    r = MainKVResidency(num_blocks=num_blocks, gpu_rows=gpu_rows, row_bytes=ROW, device=dev)
    r.bind(group_id=0, block_size=BS, max_reqs=4, table_width=8, max_tokens=64)
    return r


def test_written_blocks_bind_to_gpu_rows_and_offsets_point_at_them():
    r = _res()
    try:
        bt = torch.full((4, 8), -1, dtype=torch.int32, device="cuda")
        bt[0, :3] = torch.tensor([5, 6, 7], device="cuda")
        slots = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        slots[:4] = torch.tensor([7 * BS + 0, 7 * BS + 1, 7 * BS + 2, 7 * BS + 3], device="cuda")
        r.prepare_step(bt, slots, written_blocks=[7], protected_blocks=[7])
        torch.cuda.synchronize()
        row7 = int(r.row_of_block[7])
        assert row7 >= 1  # row 0 is scratch
        offs = r.page_offsets[:4]
        assert int(offs[0, 2]) == row7 * ROW  # GPU row
        assert int(offs[0, 0]) == (r.host_base - r.gpu_base) + 5 * ROW  # host row
        assert int(offs[0, 3]) == PTR_SENTINEL  # null entry
        remapped = r.slot_mapping[:64]
        assert remapped[:4].tolist() == [row7 * BS + k for k in range(4)]
        assert int(remapped[4]) == -1
    finally:
        r.release()


def test_window_full_demotes_coldest_unprotected_block_with_its_bytes():
    r = _res(gpu_rows=3)  # rows 1..2 usable
    try:
        bt = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        slots = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        r.prepare_step(bt, slots, written_blocks=[1], protected_blocks=[1])
        r.gpu[int(r.row_of_block[1])].fill_(11)
        r.prepare_step(bt, slots, written_blocks=[2], protected_blocks=[2])
        r.gpu[int(r.row_of_block[2])].fill_(22)
        # Third block: window full; block 1 (colder, unprotected) is demoted.
        r.prepare_step(bt, slots, written_blocks=[3], protected_blocks=[3, 2])
        torch.cuda.synchronize()
        assert int(r.row_of_block[1]) == -1 and int(r.row_of_block[2]) >= 1
        assert int(r.row_of_block[3]) >= 1
        assert int(r.host[1][0]) == 11 and int(r.host[1][-1]) == 11
        assert int(r.page_delta[1]) == (r.host_base - r.gpu_base) + 1 * ROW
        # The row block 3 took over was zeroed before its first write.
        assert int(r.gpu[int(r.row_of_block[3])].abs().sum()) == 0
        assert r.stats()["demotions"] == 1
    finally:
        r.release()


def test_protected_blocks_are_never_demoted():
    r = _res(gpu_rows=3)
    try:
        bt = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        slots = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        r.prepare_step(bt, slots, written_blocks=[1, 2], protected_blocks=[1, 2])
        with pytest.raises(RuntimeError):
            r.prepare_step(bt, slots, written_blocks=[3], protected_blocks=[1, 2, 3])
    finally:
        r.release()


def test_gather_reads_host_resident_pages_through_the_offset_table():
    """End to end with the real kernel: a block demoted to a host row is
    read back bit-exactly through the residency's page offsets."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    page, kv_heads, head_dim = BS, 1, 256
    page_bytes = page * kv_heads * 2 * head_dim  # one fp8 layer page == ROW
    assert page_bytes == ROW
    r = _res(num_blocks=6, gpu_rows=3)
    try:
        bt = torch.zeros((4, 8), dtype=torch.int32, device="cuda")
        bt[0, :3] = torch.tensor([0, 1, 2], device="cuda")
        slots = torch.full((64,), -1, dtype=torch.int64, device="cuda")
        # Fill blocks 0..2 with random fp8 bytes while resident, one per step.
        kv_full = torch.randint(0, 127, (6, page, kv_heads, 2 * head_dim), dtype=torch.uint8, device="cuda")
        for b in (0, 1, 2):
            r.prepare_step(bt, slots, written_blocks=[b], protected_blocks=[b])
            r.gpu[int(r.row_of_block[b])].copy_(kv_full[b].reshape(-1).view(torch.int8))
        # Reference: slab indexing over a plain GPU copy of the same pages.
        kv_ref = kv_full[:3].view(torch.float8_e4m3fn)
        k_ref, v_ref = kv_ref.split(head_dim, dim=-1)
        q = torch.randn(2, 6, head_dim, device="cuda", dtype=torch.bfloat16)
        idx = torch.randint(0, 3 * page, (2, 32), device="cuda", dtype=torch.int32)
        t2r = torch.zeros(2, device="cuda", dtype=torch.int32)
        ref = qsa_sparse_paged_attention(q, k_ref, v_ref, idx, bt[:1], t2r)
        # Now block 0 has been demoted (window of 2 rows); gather via the table.
        assert int(r.row_of_block[0]) == -1
        window = r.gpu.view(torch.uint8).view(r.gpu_rows, page, kv_heads, 2 * head_dim).view(torch.float8_e4m3fn)
        k_win, v_win = window.split(head_dim, dim=-1)
        out = qsa_sparse_paged_attention(
            q, k_win, v_win, idx, bt[:1], t2r, page_offsets=r.page_offsets[:1]
        )
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
    finally:
        r.release()
