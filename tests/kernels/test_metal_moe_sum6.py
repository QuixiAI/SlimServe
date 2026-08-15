# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise gate for the Metal sum-folded Q2_K down GEMV (ggml_moe_a8_vec_sum).

qgemv_moe_mr_q2_K_sum must reproduce the unfused production chain exactly:
down vec kernel (per-slot rows, rounded to the activation dtype at store) ->
qc_moe_weighted_sum (sequential fp32 reduce over the expert slots, contract
off). The folded kernel replicates both rounding points; this oracle decides
whether the enclosing slot loop moved any fp-contraction choices inside the
dot walk (the divergent-codegen failure mode recorded for the SoA repack).
Scales are finite halfs on purpose -- random scale bytes decode to inf/nan
and mask real mismatches.

Run directly on the Mac Studio: .venv/bin/python tests/kernels/test_metal_moe_sum6.py
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

E = 8
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

    n, k = 64, 2048
    w = _make_q2_k(n, k)
    w_soa = _qc_metal_soa_repack(w, 84, ((16, 64), (0, 16), (80, 4)))

    # (tokens, topk) shapes: drafter-ish and verify-ish widths.
    for tokens, topk in ((3, 6), (36, 6)):
        ids = torch.randint(0, E, (tokens, topk), dtype=torch.int32)
        ids[0, 0] = -1  # exercise the skipped-expert zero contribution
        ids = ids.to(DEV)
        tw = (torch.rand(tokens, topk, dtype=torch.float32) * 2.0 + 0.05).to(DEV)

        for dtype in (torch.float16, torch.bfloat16):
            tag = f"{tokens}x{topk} {str(dtype).split('.')[-1]}"
            x = (torch.randn(tokens * topk, k, dtype=dtype) * 0.1).to(DEV)

            for soa, wq in ((False, w), (True, w_soa)):
                rows = quixicore_ops.ggml_moe_a8_vec(
                    x, wq, ids.reshape(-1, 1), 1, 10, n, tokens * topk, soa=soa
                )
                ref = torch.empty(tokens, n, dtype=dtype, device=DEV)
                quixicore_ops.moe_weighted_sum(
                    rows.reshape(tokens, topk, n), tw, ref
                )

                got = torch.empty(tokens, n, dtype=dtype, device=DEV)
                quixicore_ops.ggml_moe_a8_vec_sum(
                    x, wq, ids, tw, topk, 10, n, tokens, got, soa=soa
                )
                checks.append(
                    (f"q2_K sum6 {tag} soa={soa}",
                     torch.equal(ref.cpu(), got.cpu()))
                )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit(f"BITWISE MISMATCH: {failed}")
    print(f"all {len(checks)} sum6-vs-unfused comparisons bit-identical")


def test_metal_moe_sum6() -> None:
    main()


if __name__ == "__main__":
    main()
