# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HIP MLA decode kernel must match a float64 reference at any head count.

The point of writing this kernel was to stop the head count from being a
property of the kernel binary: AITER's gfx942 MLA ships pre-assembled with the
query head count baked in, so Kimi K3 at TP8 (12 heads per rank) had nothing to
run on. So the head counts here are not arbitrary -- 48/16/12 are what K3's 96
heads become at TP2/TP6/TP8, and 12 is the one that used to be impossible.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("HIP MLA decode is gfx942-only", allow_module_level=True)

qc = pytest.importorskip("vllm._quixicore_C")

LATENT = 512
ROPE = 64
ENTRY = LATENT + ROPE
SCALE = 192**-0.5


def reference(q, kv_cache, block_table, seq_lens, block_size):
    """Absorbed MLA decode in float64, one request at a time."""
    batch, heads, _ = q.shape
    out = torch.zeros(batch, heads, LATENT, dtype=torch.float64)
    for b in range(batch):
        rows = []
        for token in range(int(seq_lens[b])):
            block = int(block_table[b, token // block_size])
            rows.append(kv_cache[block, token % block_size].to(torch.float64))
        kv = torch.stack(rows)  # [T, 576]
        scores = (q[b].to(torch.float64) @ kv.T) * SCALE  # [H, T]
        weights = torch.softmax(scores, dim=-1)
        out[b] = weights @ kv[:, :LATENT]
    return out


def build_case(batch, heads, seq_lens, block_size, dtype, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    num_blocks = batch * max_blocks + 3
    kv_cache = (
        torch.randn(num_blocks, block_size, ENTRY, generator=generator) * 0.5
    ).to(device=device, dtype=dtype)
    q = (torch.randn(batch, heads, ENTRY, generator=generator) * 0.5).to(
        device=device, dtype=dtype
    )
    # Deliberately non-contiguous block assignment: a kernel that assumed
    # sequential pages would pass a tidy table and fail in production.
    perm = torch.randperm(num_blocks, generator=generator)[: batch * max_blocks]
    block_table = perm.reshape(batch, max_blocks).to(device=device, dtype=torch.int32)
    lens = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return q, kv_cache, block_table, lens


@pytest.mark.parametrize("heads", [12, 16, 48])
@pytest.mark.parametrize("batch,seq_lens", [(1, [37]), (3, [1, 64, 200])])
def test_matches_float64_reference(heads, batch, seq_lens):
    device = "cuda"
    block_size = 16
    q, kv_cache, block_table, lens = build_case(
        batch, heads, seq_lens, block_size, torch.bfloat16, device, seed=0
    )
    out = torch.empty(batch, heads, LATENT, device=device, dtype=torch.bfloat16)
    qc.mla_decode_fwd(q, kv_cache, block_table, lens, out, SCALE, int(lens.max()))

    expected = reference(
        q.cpu(), kv_cache.cpu(), block_table.cpu(), lens.cpu(), block_size
    )
    torch.testing.assert_close(
        out.cpu().to(torch.float64), expected, atol=6e-3, rtol=6e-3
    )


def test_context_longer_than_one_split():
    """Split-K only pays off if the cross-slice softmax merge is right.

    A single long sequence is the case that forces many slices, so an error in
    the running max/sum rescale shows up here and nowhere else.
    """
    device = "cuda"
    block_size = 16
    q, kv_cache, block_table, lens = build_case(
        1, 12, [2000], block_size, torch.bfloat16, device, seed=7
    )
    out = torch.empty(1, 12, LATENT, device=device, dtype=torch.bfloat16)
    qc.mla_decode_fwd(q, kv_cache, block_table, lens, out, SCALE, int(lens.max()))

    expected = reference(
        q.cpu(), kv_cache.cpu(), block_table.cpu(), lens.cpu(), block_size
    )
    torch.testing.assert_close(
        out.cpu().to(torch.float64), expected, atol=6e-3, rtol=6e-3
    )


def test_split_count_override_does_not_change_the_answer():
    """The split count is a tuning knob, so no choice may alter the output.

    Split-K correctness lives entirely in the running max/sum merge; a bug there
    shows up as an answer that drifts with the slice count rather than as a
    crash.
    """
    device = "cuda"
    block_size = 16
    q, kv_cache, block_table, lens = build_case(
        2, 12, [700, 300], block_size, torch.bfloat16, device, seed=11
    )
    expected = reference(
        q.cpu(), kv_cache.cpu(), block_table.cpu(), lens.cpu(), block_size
    )
    for splits in (1, 3, 8, 64, 256):
        out = torch.empty(2, 12, LATENT, device=device, dtype=torch.bfloat16)
        qc.mla_decode_fwd(
            q, kv_cache, block_table, lens, out, SCALE, int(lens.max()), splits
        )
        torch.testing.assert_close(
            out.cpu().to(torch.float64),
            expected,
            atol=6e-3,
            rtol=6e-3,
            msg=lambda m, s=splits: f"num_splits={s}: {m}",
        )


def test_large_paged_block_size():
    """K3 is hybrid, so its MLA pages are inflated to match the KDA state page.

    The observed value is 960 tokens, far from the 16 or 64 a kernel author
    would think to test.
    """
    device = "cuda"
    block_size = 960
    q, kv_cache, block_table, lens = build_case(
        2, 12, [1500, 900], block_size, torch.bfloat16, device, seed=3
    )
    out = torch.empty(2, 12, LATENT, device=device, dtype=torch.bfloat16)
    qc.mla_decode_fwd(q, kv_cache, block_table, lens, out, SCALE, int(lens.max()))

    expected = reference(
        q.cpu(), kv_cache.cpu(), block_table.cpu(), lens.cpu(), block_size
    )
    torch.testing.assert_close(
        out.cpu().to(torch.float64), expected, atol=6e-3, rtol=6e-3
    )
