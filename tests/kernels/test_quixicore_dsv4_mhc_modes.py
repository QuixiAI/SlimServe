"""The three T == 1 launch modes of the DSV4/GLM-5.3 mHC pre-transition agree.

Mode 0 is the cooperative fused kernel (grid sync), mode 1 the last-block
fused kernel (regular launch; same partials, same reduction order, so it must
be bit-exact against mode 0), mode 2 the three-kernel split path (a
different partial layout, so it is compared with a tolerance). Mode 1 is
also hammered in a loop to exercise the self-resetting completion counter.
"""

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_dsv4_mhc_modes()),
    reason="needs CUDA and the QuixiCore dsv4 mHC mode switch",
)
DEV = "cuda"
HC, H, MIXES = 4, 4096, 24
EPS = dict(
    rms_eps=1e-6,
    pre_eps=1e-2,
    sinkhorn_eps=1e-6,
    post_multiplier=0.5,
    sinkhorn_repeat=3,
)


def _inputs(seed, fn_dtype):
    g = torch.Generator(device=DEV).manual_seed(seed)
    residual = torch.randn(1, HC, H, device=DEV, generator=g).to(torch.bfloat16)
    fn = (torch.randn(MIXES, HC * H, device=DEV, generator=g) * 0.02).to(fn_dtype)
    hc_scale = torch.tensor([0.3, 0.2, 0.1], device=DEV)
    hc_base = torch.randn(MIXES, device=DEV, generator=g) * 0.1
    x = torch.randn(1, H, device=DEV, generator=g).to(torch.bfloat16)
    norm_weight = (1 + 0.1 * torch.randn(H, device=DEV, generator=g)).to(torch.bfloat16)
    return residual, fn, hc_scale, hc_base, x, norm_weight


def _run_pre(mode, residual, fn, hc_scale, hc_base, norm_weight):
    quixicore_ops.set_dsv4_mhc_mode(mode)
    assert quixicore_ops.get_dsv4_mhc_mode() == mode
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
        norm_weight,
        1e-6 if norm_weight is not None else 0.0,
    )
    torch.cuda.synchronize()
    return [t.clone() for t in out]


def _run_fused(mode, x, residual, post, comb, fn, hc_scale, hc_base, norm_weight):
    quixicore_ops.set_dsv4_mhc_mode(mode)
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
        norm_weight,
        1e-6 if norm_weight is not None else 0.0,
    )
    torch.cuda.synchronize()
    return [t.clone() for t in out]


def _assert_close(a, b, exact):
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        assert ta.shape == tb.shape and ta.dtype == tb.dtype
        if exact:
            assert torch.equal(ta, tb), (
                f"max diff {(ta.float() - tb.float()).abs().max()}"
            )
        else:
            torch.testing.assert_close(ta.float(), tb.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("with_norm", [False, True])
def test_pre_modes_agree(fn_dtype, with_norm):
    residual, fn, hc_scale, hc_base, _, norm_weight = _inputs(1, fn_dtype)
    nw = norm_weight if with_norm else None
    try:
        ref = _run_pre(0, residual, fn, hc_scale, hc_base, nw)
        last = _run_pre(1, residual, fn, hc_scale, hc_base, nw)
        split = _run_pre(2, residual, fn, hc_scale, hc_base, nw)
    finally:
        quixicore_ops.set_dsv4_mhc_mode(0)
    _assert_close(last, ref, exact=True)
    _assert_close(split, ref, exact=False)


@pytest.mark.parametrize("fn_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("with_norm", [False, True])
def test_fused_post_pre_modes_agree(fn_dtype, with_norm):
    residual, fn, hc_scale, hc_base, x, norm_weight = _inputs(2, fn_dtype)
    nw = norm_weight if with_norm else None
    try:
        post, comb, _ = _run_pre(0, residual, fn, hc_scale, hc_base, nw)
        ref = _run_fused(0, x, residual, post, comb, fn, hc_scale, hc_base, nw)
        last = _run_fused(1, x, residual, post, comb, fn, hc_scale, hc_base, nw)
        split = _run_fused(2, x, residual, post, comb, fn, hc_scale, hc_base, nw)
    finally:
        quixicore_ops.set_dsv4_mhc_mode(0)
    _assert_close(last, ref, exact=True)
    _assert_close(split, ref, exact=False)


def test_last_block_counter_survives_a_burst():
    """Back-to-back launches on one stream: the counter must reset every time."""
    residual, fn, hc_scale, hc_base, x, _ = _inputs(3, torch.float32)
    try:
        ref = _run_pre(0, residual, fn, hc_scale, hc_base, None)
        quixicore_ops.set_dsv4_mhc_mode(1)
        outs = []
        for _ in range(256):
            outs.append(
                quixicore_ops.dsv4_mhc_pre(
                    residual,
                    fn,
                    hc_scale,
                    hc_base,
                    EPS["rms_eps"],
                    EPS["pre_eps"],
                    EPS["sinkhorn_eps"],
                    EPS["post_multiplier"],
                    EPS["sinkhorn_repeat"],
                    None,
                    0.0,
                )
            )
        torch.cuda.synchronize()
    finally:
        quixicore_ops.set_dsv4_mhc_mode(0)
    for out in outs:
        _assert_close(list(out), ref, exact=True)
