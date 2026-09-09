"""The prefill-shaped mHC partials kernel agrees with the decode-shaped one.

Both serve the split path (mode 2 here, so every T takes it). The prefill
kernel stages a 512-flat split of a 32-token tile in shared memory and runs
whole dot products per lane; it writes the same [T][32][25] partial layout
over a different flat-to-split assignment, so the mix sums differ only in
fp32 summation order (tolerance), while the fused post-mix residual is the
same expression in the same order (bit-exact). T = 33 and 1000 exercise the
tile tail; the threshold setter selects the kernel per call.
"""

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_dsv4_mhc_prefill()),
    reason="needs CUDA and the QuixiCore dsv4 mHC prefill kernel",
)
DEV = "cuda"
HC, H, MIXES = 4, 4096, 24
DEFAULT_MIN_T = 64  # the launcher's default; each test restores it
EPS = dict(
    rms_eps=1e-6,
    pre_eps=1e-2,
    sinkhorn_eps=1e-6,
    post_multiplier=0.5,
    sinkhorn_repeat=3,
)


def _inputs(seed, T, fn_dtype):
    g = torch.Generator(device=DEV).manual_seed(seed)
    residual = torch.randn(T, HC, H, device=DEV, generator=g).to(torch.bfloat16)
    fn = (torch.randn(MIXES, HC * H, device=DEV, generator=g) * 0.02).to(fn_dtype)
    hc_scale = torch.tensor([0.3, 0.2, 0.1], device=DEV)
    hc_base = torch.randn(MIXES, device=DEV, generator=g) * 0.1
    x = torch.randn(T, H, device=DEV, generator=g).to(torch.bfloat16)
    post = torch.rand(T, HC, device=DEV, generator=g)
    comb = torch.rand(T, HC, HC, device=DEV, generator=g)
    norm_weight = (1 + 0.1 * torch.randn(H, device=DEV, generator=g)).to(torch.bfloat16)
    return residual, fn, hc_scale, hc_base, x, post, comb, norm_weight


def _pre(min_t, residual, fn, hc_scale, hc_base, nw):
    quixicore_ops.set_dsv4_mhc_prefill_min_t(min_t)
    assert quixicore_ops.get_dsv4_mhc_prefill_min_t() == min_t
    out = quixicore_ops.dsv4_mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        EPS["rms_eps"],
        EPS["pre_eps"],
        EPS["sinkhorn_eps"],
        EPS["post_multiplier"],
        EPS["sinkhorn_repeat"],
        nw,
        1e-6 if nw is not None else 0.0,
    )
    torch.cuda.synchronize()
    return [t.clone() for t in out]


def _fused(min_t, x, residual, post, comb, fn, hc_scale, hc_base, nw):
    quixicore_ops.set_dsv4_mhc_prefill_min_t(min_t)
    out = quixicore_ops.dsv4_mhc_fused_post_pre(
        x,
        residual,
        post,
        comb,
        fn,
        hc_scale,
        hc_base,
        EPS["rms_eps"],
        EPS["pre_eps"],
        EPS["sinkhorn_eps"],
        EPS["post_multiplier"],
        EPS["sinkhorn_repeat"],
        nw,
        1e-6 if nw is not None else 0.0,
    )
    torch.cuda.synchronize()
    return [t.clone() for t in out]


def _close(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    torch.testing.assert_close(a.float(), b.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("T", [33, 256, 1000])
@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_norm", [False, True])
def test_pre_prefill_kernel_agrees(T, fn_dtype, with_norm):
    residual, fn, hc_scale, hc_base, _, _, _, norm_weight = _inputs(1, T, fn_dtype)
    nw = norm_weight if with_norm else None
    saved_mode = quixicore_ops.get_dsv4_mhc_mode()
    try:
        quixicore_ops.set_dsv4_mhc_mode(2)
        ref = _pre(0, residual, fn, hc_scale, hc_base, nw)
        new = _pre(1, residual, fn, hc_scale, hc_base, nw)
    finally:
        quixicore_ops.set_dsv4_mhc_mode(saved_mode)
        quixicore_ops.set_dsv4_mhc_prefill_min_t(DEFAULT_MIN_T)
    for a, b in zip(new, ref):
        _close(a, b)
    # mix sums of 16384 products of order 1e-2: fp32 order noise is far below 1e-4
    torch.testing.assert_close(new[0], ref[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(new[1], ref[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("T", [33, 256, 1000])
@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_norm", [False, True])
def test_fused_prefill_kernel_agrees_and_residual_is_bit_exact(T, fn_dtype, with_norm):
    residual, fn, hc_scale, hc_base, x, post, comb, norm_weight = _inputs(
        2, T, fn_dtype
    )
    nw = norm_weight if with_norm else None
    saved_mode = quixicore_ops.get_dsv4_mhc_mode()
    try:
        quixicore_ops.set_dsv4_mhc_mode(2)
        ref = _fused(0, x, residual, post, comb, fn, hc_scale, hc_base, nw)
        new = _fused(1, x, residual, post, comb, fn, hc_scale, hc_base, nw)
    finally:
        quixicore_ops.set_dsv4_mhc_mode(saved_mode)
        quixicore_ops.set_dsv4_mhc_prefill_min_t(DEFAULT_MIN_T)
    assert torch.equal(new[0], ref[0]), "fused post-mix residual must be bit-exact"
    torch.testing.assert_close(new[1], ref[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(new[2], ref[2], atol=1e-4, rtol=1e-4)
    _close(new[3], ref[3])


def test_threshold_gates_the_kernel():
    residual, fn, hc_scale, hc_base, _, _, _, _ = _inputs(3, 64, torch.float32)
    saved_mode = quixicore_ops.get_dsv4_mhc_mode()
    try:
        quixicore_ops.set_dsv4_mhc_mode(2)
        below = _pre(
            65, residual, fn, hc_scale, hc_base, None
        )  # T < min_t: decode-shaped kernel
        at = _pre(
            64, residual, fn, hc_scale, hc_base, None
        )  # T == min_t: prefill kernel
        off = _pre(0, residual, fn, hc_scale, hc_base, None)
    finally:
        quixicore_ops.set_dsv4_mhc_mode(saved_mode)
        quixicore_ops.set_dsv4_mhc_prefill_min_t(DEFAULT_MIN_T)
    for a, b in zip(below, off):
        assert torch.equal(a, b)
    torch.testing.assert_close(at[0], off[0], atol=1e-4, rtol=1e-4)
