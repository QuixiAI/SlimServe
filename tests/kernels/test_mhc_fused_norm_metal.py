# SPDX-License-Identifier: Apache-2.0
"""Metal mHC kernels with the fused layer-input RMSNorm epilogue
(``dsv4_mhc_fused_post_pre`` / ``dsv4_mhc_pre`` with ``norm_weight``):
the gate/Sinkhorn outputs are bit-identical to the unfused call, and the
layer input equals rms_norm(unfused layer input) * weight. The fused
post+pre host runs the post kernel and the split pre (dots + finalize) in
one encode; it must be bit-identical to calling the pair."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has_kernel("dsv4_mhc_pre_finalize_norm_bfloat16"):
        pytest.skip("norm-fused mHC kernels not built")
    return quixicore_ops


def _ref_norm(li: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x = li.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * w.float()).to(li.dtype)


def _inputs(T, H, dtype):
    torch.manual_seed(1)
    dev = "mps"
    residual = (torch.randn(T, 4, H, device=dev) * 0.7).to(dtype)
    x = torch.randn(T, H, device=dev).to(dtype)
    post_mix = torch.rand(T, 4, device=dev) * 0.5
    comb_mix = torch.softmax(torch.randn(T, 4, 4, device=dev), dim=-1)
    fn = torch.randn(24, 4 * H, device=dev) * 0.02
    scale = torch.tensor([0.9, 1.1, 0.8], device=dev)
    base = torch.randn(24, device=dev) * 0.1
    w = (1.0 + 0.1 * torch.randn(H, device=dev)).to(dtype)
    return residual, x, post_mix, comb_mix, fn, scale, base, w


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 7, 70])
def test_fused_post_pre_norm(dtype, T):
    qc = _qc()
    H = 4096
    residual, x, post_mix, comb_mix, fn, scale, base, w = _inputs(T, H, dtype)
    args = (x, residual, post_mix, comb_mix, fn, scale, base,
            1e-5, 1e-6, 1e-6, 2.0, 20)
    r0, p0, c0, li0 = qc.dsv4_mhc_fused_post_pre(*args)
    r1, p1, c1, li1 = qc.dsv4_mhc_fused_post_pre(*args, w, 1e-5)
    torch.mps.synchronize()
    assert torch.equal(r0, r1) and torch.equal(p0, p1) and torch.equal(c0, c1)
    ref = _ref_norm(li0, w, 1e-5)
    err = (li1.float() - ref.float()).abs()
    tol = 2e-2 * ref.float().abs().clamp(min=1.0)
    assert (err <= tol).all(), err.max().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 7, 70])
def test_pre_norm(dtype, T):
    qc = _qc()
    H = 4096
    residual, _, _, _, fn, scale, base, w = _inputs(T, H, dtype)
    args = (residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    p0, c0, li0 = qc.dsv4_mhc_pre(*args)
    p1, c1, li1 = qc.dsv4_mhc_pre(*args, w, 1e-5)
    torch.mps.synchronize()
    assert torch.equal(p0, p1) and torch.equal(c0, c1)
    ref = _ref_norm(li0, w, 1e-5)
    err = (li1.float() - ref.float()).abs()
    tol = 2e-2 * ref.float().abs().clamp(min=1.0)
    assert (err <= tol).all(), err.max().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("T", [1, 5, 33, 63, 64, 130])
def test_fused_post_pre_equals_pair(dtype, T):
    qc = _qc()
    H = 4096
    residual, x, post_mix, comb_mix, fn, scale, base, w = _inputs(T, H, dtype)
    for nw in (None, w):
        extra = () if nw is None else (nw, 1e-5)
        r0, p0, c0, li0 = qc.dsv4_mhc_fused_post_pre(
            x, residual, post_mix, comb_mix, fn, scale, base,
            1e-5, 1e-6, 1e-6, 2.0, 20, *extra,
        )
        r1 = qc.dsv4_mhc_post(x, residual, post_mix, comb_mix)
        p1, c1, li1 = qc.dsv4_mhc_pre(
            r1, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20, *extra
        )
        torch.mps.synchronize()
        assert torch.equal(r0, r1) and torch.equal(p0, p1)
        assert torch.equal(c0, c1) and torch.equal(li0, li1)
