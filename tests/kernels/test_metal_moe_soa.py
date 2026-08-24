# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise gate for the Metal SoA Q2_K MoE expert repack.

The SoA kernels must consume the load-time per-expert plane permutation and
produce BIT-IDENTICAL outputs to the AoS multi-row kernels: same bytes, same
arithmetic order, only the addressing changes. Scales are generated as finite
halfs on purpose -- fully random scale bytes decode to inf/nan and mask real
mismatches (the vacuous-pass trap recorded in perf/metal_m1ultra_retrospective.md).

iq2_xxs has no SoA layout (both repacks measured slower than AoS on Apple),
so only the production Q2_K down repack is gated here.

Run directly on the Mac Studio: .venv/bin/python tests/kernels/test_metal_moe_soa.py
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

E, TOKENS, TOPK = 8, 3, 6
DEV = "mps"


def _finite_half_bytes(*shape) -> torch.Tensor:
    """Random finite fp16 scales in [0.25, 0.75), as raw little-endian bytes."""
    vals = (torch.rand(*shape, dtype=torch.float32) * 0.5 + 0.25).to(torch.float16)
    return vals.view(torch.uint8).reshape(*shape[:-1], shape[-1] * 2)


def _make_q2_k(n_rows: int, k: int) -> torch.Tensor:
    nb = k // 256
    blocks = torch.randint(0, 256, (E, n_rows, nb, 84), dtype=torch.uint8)
    blocks[..., 80:84] = _finite_half_bytes(E, n_rows, nb, 2)
    return blocks.reshape(E, n_rows, nb * 84).contiguous().to(DEV)


def main() -> None:
    from vllm.model_executor.layers.quantization.gguf.fused_moe import (
        _qc_metal_soa_repack,
    )
    from vllm.quixicore.ops import quixicore_ops

    torch.manual_seed(0)
    checks = []

    for dtype in (torch.float16, torch.bfloat16):
        tag = str(dtype).split(".")[-1]

        # Q2_K down: fp16 serves the g48 geometry, bf16 the 2x4 default.
        n, k = 64, 2048
        w = _make_q2_k(n, k)
        w_soa = _qc_metal_soa_repack(w, 84, ((16, 64), (0, 16), (80, 4)))
        x = (torch.randn(TOKENS * TOPK, k, dtype=dtype) * 0.1).to(DEV)
        ids1 = torch.randint(0, E, (TOKENS * TOPK, 1), dtype=torch.int32).to(DEV)
        ids1[0, 0] = -1  # exercise the skipped-expert zero-fill in both layouts

        ref = quixicore_ops.ggml_moe_a8_vec(x, w, ids1, 1, 10, n, TOKENS * TOPK)
        got = quixicore_ops.ggml_moe_a8_vec(
            x, w_soa, ids1, 1, 10, n, TOKENS * TOPK, soa=True
        )
        checks.append((f"q2_K mr {tag}", torch.equal(ref.cpu(), got.cpu())))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit(f"BITWISE MISMATCH: {failed}")
    print(f"all {len(checks)} SoA-vs-AoS comparisons bit-identical")


def test_metal_moe_soa() -> None:
    main()


if __name__ == "__main__":
    main()
