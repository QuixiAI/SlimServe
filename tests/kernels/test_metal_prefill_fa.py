# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle for the dense-causal prefill MMA FA (VLLM_QC_MLA_PREFILL_FA_MMA).

mla_prefill_dequant_slots + mla_prefill_fa_mma must reproduce the decode
walk (mla_decode_fp8_sparse_two_cache_packed) within the P-half-rounding
ULP class on dense-causal tables (norm-relative gate), deterministically
(bitwise x2). Run directly on the Mac Studio:
.venv/bin/python tests/kernels/test_metal_prefill_fa.py
"""

import time

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
T, H = 1920, 64
CR, W = 4, 128
SCALE = 0.0442
REL_TOL = 2e-2  # ULP-class FA vs the decode walk, norm-relative


def _make_cache(n_blocks: int) -> torch.Tensor:
    c = torch.randint(0, 255, (n_blocks, 256, 584), dtype=torch.uint8)
    c[..., :448] &= 0x7E
    rope = (
        (torch.rand(n_blocks, 256, 64) * 2 - 1)
        .to(torch.bfloat16)
        .view(torch.uint8)
        .reshape(n_blocks, 256, 128)
    )
    c[..., 448:576] = rope
    c[..., 576:584] = torch.randint(
        120, 134, (n_blocks, 256, 8), dtype=torch.uint8
    )
    return c.to(DEV)


def _pad_slots(t: torch.Tensor, n: int) -> torch.Tensor:
    npad = ((n + 31) // 32) * 32 + 32
    out = torch.full((npad,), -1, dtype=torch.int32, device=DEV)
    out[:n] = t[:n]
    return out


def main(bench: bool = False) -> None:
    from vllm.quixicore.ops import quixicore_ops as qc

    torch.manual_seed(0)
    q = (
        torch.randn(T, H, 512, dtype=torch.bfloat16, device=DEV) * 0.05
    ).contiguous()
    sinks = (torch.randn(H, dtype=torch.float32, device=DEV) * 0.1).contiguous()
    comp_cache = _make_cache(16)
    swa_cache = _make_cache(16)

    pos = torch.arange(T, dtype=torch.int32, device=DEV)
    lens_c = torch.div(pos + 1, CR, rounding_mode="floor").to(torch.int32)
    nc_true = int(lens_c.max().item())
    swa_lens = torch.minimum(pos + 1, torch.full_like(pos, W))
    lo_abs = pos + 1 - swa_lens  # window start position
    hi_abs = pos + 1

    # Reference via the decode kernel (per-token tables); batches < 64 keep
    # it on the decode/staged path.
    cw = 512
    sc = (
        torch.arange(cw, dtype=torch.int32, device=DEV)
        .view(1, cw)
        .expand(T, cw)
        .contiguous()
        .clone()
    )
    sc = torch.where(
        torch.arange(cw, device=DEV).view(1, cw) < lens_c.view(T, 1),
        sc,
        torch.tensor(-1, dtype=torch.int32, device=DEV),
    )
    ss = lo_abs.view(T, 1) + torch.arange(
        W, device=DEV, dtype=torch.int32
    ).view(1, W)
    ss = (
        torch.where(
            torch.arange(W, device=DEV).view(1, W) < swa_lens.view(T, 1),
            ss,
            torch.tensor(-1, dtype=torch.int32, device=DEV),
        )
        .to(torch.int32)
        .contiguous()
    )
    outs = []
    for s0 in range(0, T, 32):
        e0 = min(s0 + 32, T)
        outs.append(
            qc.deepseek_v4_sparse_attention(
                q[s0:e0],
                comp_cache,
                sc[s0:e0],
                lens_c[s0:e0],
                swa_cache,
                ss[s0:e0],
                swa_lens.to(torch.int32)[s0:e0],
                sinks,
                SCALE,
            )
        )
    ref = torch.cat(outs, 0)
    torch.mps.synchronize()

    axis_c = _pad_slots(
        torch.arange(nc_true, dtype=torch.int32, device=DEV), nc_true
    )
    axis_s = _pad_slots(
        torch.arange(T, dtype=torch.int32, device=DEV), T
    )  # positions ARE slots here
    kc = qc.deepseek_v4_prefill_dequant(comp_cache, axis_c)
    ks = qc.deepseek_v4_prefill_dequant(swa_cache, axis_s)

    def fa() -> torch.Tensor:
        return qc.deepseek_v4_prefill_fa(
            q,
            kc,
            ks,
            lens_c,
            lo_abs.to(torch.int32),
            hi_abs.to(torch.int32),
            sinks,
            SCALE,
        )

    a = fa()
    torch.mps.synchronize()
    b = fa()
    torch.mps.synchronize()
    det = torch.equal(a, b)

    dev_ = (a.float() - ref.float()).abs().max().item()
    scale_ref = ref.float().abs().max().item()
    rel = dev_ / scale_ref
    print(f"det={det} max_dev={dev_:.3e} rel={rel:.3e} ref_scale={scale_ref:.3e}")
    failed = []
    if not det:
        failed.append("determinism x2")
    if rel > REL_TOL:
        failed.append(f"rel {rel:.3e} > {REL_TOL:.0e}")
    if failed:
        raise SystemExit(f"ORACLE MISMATCH: {failed}")
    print("prefill FA oracle passed")

    if bench:
        for _ in range(3):
            fa()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            fa()
        torch.mps.synchronize()
        print(f"FA: {(time.perf_counter() - t0) / 10 * 1000:.2f} ms")


def test_metal_prefill_fa() -> None:
    main()


if __name__ == "__main__":
    main(bench=True)
