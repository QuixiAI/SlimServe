# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle for the head=512 compressor fronts (dsv4_compress_front[_c128]).

Compares the native kernels against the verbatim eager math on synthetic
state caches: cr=4 overlap and cr=128 no-overlap (the threadgroup kernel
that skips non-boundary rows at prefill). Only boundary rows with
state_slot >= 0 are compared — the kernel leaves other rows unwritten by
contract. Bitwise identity is checked first; if the c128 softmax reduction
order diverges from MPS internals, the gate falls back to a 1-ulp bf16
tolerance and reports which level held. Determinism x2 is bitwise.

Run directly on the Mac Studio:
.venv/bin/python tests/kernels/test_metal_compress_front_c128.py
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


def _front512_reference(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    overlap: bool = True,
) -> torch.Tensor:
    """The verbatim eager head=512 math (no insert) the kernels replaced."""
    head_dim = 512
    positions_a = positions[:num_actual]
    req = token_to_req_indices[:num_actual].to(torch.long)
    history = compress_ratio * (2 if overlap else 1)
    history_offsets = torch.arange(
        history, device=positions_a.device, dtype=positions_a.dtype
    )
    source_pos = (
        positions_a.unsqueeze(1) - history + 1 + history_offsets.unsqueeze(0)
    )
    source_valid = source_pos >= 0
    safe_pos = source_pos.clamp_min(0)
    req_block_table = block_table.index_select(0, req)
    source_block_col = torch.div(
        safe_pos, block_size, rounding_mode="floor"
    ).clamp(max=req_block_table.shape[1] - 1)
    source_blocks = req_block_table.gather(
        1, source_block_col.to(torch.long)
    ).to(torch.long)
    source_offsets = torch.remainder(safe_pos, block_size).to(torch.long)
    head_offsets = (history_offsets >= compress_ratio).to(torch.long) * head_dim
    dims = torch.arange(head_dim, device=positions_a.device, dtype=torch.long)
    gather_dims = head_offsets.view(1, history, 1) + dims.view(1, 1, -1)
    gather_dims = gather_dims.expand(num_actual, -1, -1)
    score_mask = ~source_valid.unsqueeze(-1)
    rows = state_cache[source_blocks, source_offsets]
    values = rows.gather(2, gather_dims)
    scores = rows.gather(2, gather_dims + state_width)
    scores = scores.masked_fill(score_mask, -float("inf"))
    compressed = (values * torch.softmax(scores, dim=1)).sum(dim=1)
    compressed_fp32 = compressed.float()
    rrms = torch.rsqrt(
        compressed_fp32.square().mean(dim=-1, keepdim=True) + rms_norm_eps
    )
    return (compressed_fp32 * rrms * rms_norm_weight.float()).to(torch.bfloat16)


def _case(compress_ratio: int, overlap: bool, tokens: int, seed: int):
    from vllm.quixicore.ops import quixicore_ops

    torch.manual_seed(seed)
    state_width = 1024 if overlap else 512
    row_width = 2 * state_width
    block_size = 4 if compress_ratio == 4 else 8
    history = compress_ratio * (2 if overlap else 1)

    # One request whose positions run far enough for full history plus a
    # clamped prefix; block table maps position blocks 1:1 with a jumbled
    # physical order to exercise the indirection.
    max_pos = tokens + history + 16
    n_cols = (max_pos + block_size - 1) // block_size + 1
    perm = torch.randperm(n_cols, dtype=torch.int32)
    block_table = perm.unsqueeze(0).contiguous().to(DEV)
    n_blocks = n_cols + 2
    state_cache = (
        torch.randn(n_blocks, block_size, row_width, dtype=torch.float32) * 0.5
    ).to(DEV)

    # Tokens at consecutive positions ending on/around compress boundaries,
    # including early positions (clamped history) and some invalid slots.
    positions = torch.arange(tokens, dtype=torch.int64)
    positions = positions * (compress_ratio // 2 if compress_ratio == 4 else 32)
    positions = positions + compress_ratio - 1  # many boundary hits
    state_slots = torch.where(
        torch.arange(tokens) % 7 == 3,
        torch.tensor(-1, dtype=torch.int64),
        torch.arange(tokens, dtype=torch.int64),
    )
    t2r = torch.zeros(tokens, dtype=torch.int32)
    w = (torch.rand(512, dtype=torch.float32) * 1.5 + 0.25).to(DEV)
    eps = 1e-6

    pos_dev = positions.to(DEV)
    slots_dev = state_slots.to(DEV)
    t2r_dev = t2r.to(DEV)

    ref = _front512_reference(
        state_cache,
        tokens,
        t2r_dev,
        pos_dev,
        block_table,
        block_size,
        state_width,
        compress_ratio,
        w,
        eps,
        overlap=overlap,
    )
    got = quixicore_ops.dsv4_compress_front(
        state_cache,
        pos_dev.to(torch.int32),
        slots_dev,
        t2r_dev,
        block_table,
        w,
        tokens,
        block_size,
        state_width,
        compress_ratio,
        eps,
    )
    again = quixicore_ops.dsv4_compress_front(
        state_cache,
        pos_dev.to(torch.int32),
        slots_dev,
        t2r_dev,
        block_table,
        w,
        tokens,
        block_size,
        state_width,
        compress_ratio,
        eps,
    )

    valid = ((positions + 1) % compress_ratio == 0) & (state_slots >= 0)
    idx = valid.nonzero().flatten()
    ref_v = ref.cpu()[idx]
    got_v = got.cpu()[idx]
    again_v = again.cpu()[idx]

    bitwise = torch.equal(ref_v.view(torch.uint16), got_v.view(torch.uint16))
    max_err = (got_v.float() - ref_v.float()).abs().max().item() if len(idx) else 0.0
    scale = ref_v.float().abs().max().item() if len(idx) else 1.0
    tol_ok = max_err <= 8e-3 * max(scale, 1e-6)  # ~1 ulp of bf16
    det = torch.equal(got_v, again_v)
    tag = f"cr={compress_ratio} tokens={tokens} valid={len(idx)}"
    level = "BITWISE" if bitwise else ("tol" if tol_ok else "FAIL")
    return [
        (f"front {tag}: {level} (max err {max_err:.2e} / scale {scale:.2f})",
         bitwise or tol_ok),
        (f"determinism x2 {tag}", det),
    ]


def main() -> None:
    checks = []
    checks += _case(4, True, 64, 0)
    checks += _case(128, False, 48, 1)
    checks += _case(128, False, 129, 2)

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit(f"ORACLE MISMATCH: {failed}")
    print(f"all {len(checks)} compressor-front checks passed")


def test_metal_compress_front_c128() -> None:
    main()


if __name__ == "__main__":
    main()
