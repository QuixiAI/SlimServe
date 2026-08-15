# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle gate for the Metal tiled MoE prefill GEMM.

qc_moe_mm_map0 + qc_moe_mm_id_iq2_xxs_w64 must reproduce the production GEMV
chain (ggml_moe_a8_vec, plain, no swiglu) within tolerance: the tile path
rounds each dequantized weight to fp16 before the simdgroup MMA (llama.cpp
does the same; the GEMV keeps weights fp32 through the dot walk) and the
fp32 accumulation associates differently, so bitwise identity is not
expected. The gate is norm-relative (max err <= 1e-2 * max |ref|): random
unscaled iq2 payloads drive outputs to +-500, where per-weight fp16
rounding random-walks to ~1 output ULP and per-element rtol misfires on
cancellation-small entries. Determinism is still required bitwise (same
input -> same output, twice).

Expert ids are unique per token on purpose -- the map0 membership scan is
branchless and duplicate ids within one token's top-k would corrupt it; the
DSV4 router emits unique ids, and the python gate keeps expert_map None.
Scales are finite halfs; random iq2_xxs payload bytes are always decodable
(256-entry grid, masked signs).

Run directly on the Mac Studio: .venv/bin/python tests/kernels/test_metal_moe_mm.py
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
QT_IQ2_XXS = 16
QT_Q2_K = 10


def _finite_half_bytes(*shape) -> torch.Tensor:
    """Random finite fp16 scales in [0.25, 0.75), as raw little-endian bytes."""
    vals = (torch.rand(*shape, dtype=torch.float32) * 0.5 + 0.25).to(torch.float16)
    return vals.view(torch.uint8).reshape(*shape[:-1], shape[-1] * 2)


def _make_iq2_xxs(e: int, n_rows: int, k: int) -> torch.Tensor:
    nb = k // 256
    blocks = torch.randint(0, 256, (e, n_rows, nb, 66), dtype=torch.uint8)
    blocks[..., 0:2] = _finite_half_bytes(e, n_rows, nb, 1)
    return blocks.reshape(e, n_rows, nb * 66).contiguous().to(DEV)


def _unique_topk_ids(tokens: int, e: int, topk: int) -> torch.Tensor:
    """Unique expert ids per token (the router contract map0 relies on)."""
    return (
        torch.rand(tokens, e).argsort(dim=1)[:, :topk].to(torch.int32).contiguous()
    )


def _make_q2_k(e: int, n_rows: int, k: int) -> torch.Tensor:
    nb = k // 256
    blocks = torch.randint(0, 256, (e, n_rows, nb, 84), dtype=torch.uint8)
    blocks[..., 80:84] = _finite_half_bytes(e, n_rows, nb, 2)
    return blocks.reshape(e, n_rows, nb * 84).contiguous().to(DEV)


def main() -> None:
    from vllm.model_executor.layers.quantization.gguf.fused_moe import (
        _qc_mm_min_tokens,
    )
    from vllm.quixicore.ops import quixicore_ops

    torch.manual_seed(0)
    checks = []

    # Decode widths must stay on the vec path: the python gate's token
    # threshold sits above the 6-token verify batch.
    checks.append(("decode gate (6 < min_tokens)", 6 < _qc_mm_min_tokens()))

    n, k = 128, 2048
    topk = 6

    for e in (8, 64, 256):
        w = _make_iq2_xxs(e, n, k)
        for tokens in (33, 64, 100):
            ids = _unique_topk_ids(tokens, e, topk).to(DEV)
            x = (torch.randn(tokens, k, dtype=torch.float16) * 0.1).to(DEV)

            ref = quixicore_ops.ggml_moe_a8_vec(
                x, w, ids, topk, QT_IQ2_XXS, n, tokens
            )
            got = quixicore_ops.ggml_moe_mm_id(
                x, w, ids, topk, QT_IQ2_XXS, n, tokens
            )
            again = quixicore_ops.ggml_moe_mm_id(
                x, w, ids, topk, QT_IQ2_XXS, n, tokens
            )

            ref_c, got_c, again_c = ref.cpu(), got.cpu(), again.cpu()
            max_err = (got_c.float() - ref_c.float()).abs().max().item()
            scale = ref_c.float().abs().max().item()
            tag = f"E={e} tokens={tokens}"
            checks.append(
                (
                    f"mm vs vec {tag} (max err {max_err:.2e} / scale {scale:.1f})",
                    max_err <= 1e-2 * scale,
                )
            )
            checks.append(
                (f"determinism x2 {tag}", torch.equal(got_c, again_c))
            )

    # Concentrated routing: every token picks experts 0..topk-1, so a few
    # experts own multiple 32-slot tiles and the rest early-exit at tpe=0.
    e = 64
    tokens = 100
    w = _make_iq2_xxs(e, n, k)
    ids = (
        torch.arange(topk, dtype=torch.int32).expand(tokens, topk).contiguous()
    ).to(DEV)
    x = (torch.randn(tokens, k, dtype=torch.float16) * 0.1).to(DEV)
    ref = quixicore_ops.ggml_moe_a8_vec(x, w, ids, topk, QT_IQ2_XXS, n, tokens)
    got = quixicore_ops.ggml_moe_mm_id(x, w, ids, topk, QT_IQ2_XXS, n, tokens)
    ref_c, got_c = ref.cpu(), got.cpu()
    max_err = (got_c.float() - ref_c.float()).abs().max().item()
    scale = ref_c.float().abs().max().item()
    checks.append(
        (
            f"mm vs vec concentrated E=64 tokens=100 "
            f"(max err {max_err:.2e} / scale {scale:.1f})",
            max_err <= 1e-2 * scale,
        )
    )

    # q2_K down tile (per-slot B rows, id-indexed): AoS and the resident SoA
    # plane layout, against the plain down GEMV (topk=1 per-slot framing).
    from vllm.model_executor.layers.quantization.gguf.fused_moe import (
        _qc_metal_soa_repack,
    )

    n2, k2 = 128, 2048
    for e in (8, 256):
        w = _make_q2_k(e, n2, k2)
        w_soa = _qc_metal_soa_repack(w, 84, ((16, 64), (0, 16), (80, 4)))
        for tokens in (33, 100):
            ids = _unique_topk_ids(tokens, e, topk).to(DEV)
            x = (
                torch.randn(tokens * topk, k2, dtype=torch.float16) * 0.1
            ).to(DEV)
            ref = quixicore_ops.ggml_moe_a8_vec(
                x, w, ids.reshape(-1, 1), 1, QT_Q2_K, n2, tokens * topk
            )
            for soa, wq in ((False, w), (True, w_soa)):
                got = quixicore_ops.ggml_moe_mm_id(
                    x, wq, ids, topk, QT_Q2_K, n2, tokens, soa=soa
                )
                again = quixicore_ops.ggml_moe_mm_id(
                    x, wq, ids, topk, QT_Q2_K, n2, tokens, soa=soa
                )
                ref_c, got_c, again_c = ref.cpu(), got.cpu(), again.cpu()
                max_err = (got_c.float() - ref_c.float()).abs().max().item()
                scale = ref_c.float().abs().max().item()
                tag = f"q2_K E={e} tokens={tokens} soa={soa}"
                checks.append(
                    (
                        f"mm vs vec {tag} (max err {max_err:.2e} / "
                        f"scale {scale:.1f})",
                        max_err <= 1e-2 * scale,
                    )
                )
                checks.append(
                    (f"determinism x2 {tag}", torch.equal(got_c, again_c))
                )

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if failed:
        raise SystemExit(f"ORACLE MISMATCH: {failed}")
    print(f"all {len(checks)} tiled-GEMM-vs-GEMV checks passed")


def test_metal_moe_mm() -> None:
    main()


if __name__ == "__main__":
    main()
