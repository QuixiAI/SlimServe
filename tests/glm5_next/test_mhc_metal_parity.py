# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash mHC on Apple Metal: the registered ``glm5_mhc_*`` ops must
take the QuixiCore-Metal ``dsv4_mhc_*`` route (the SM80 SIMT Triton
transition is CUDA-only) with GLM's constants, and match the eager torch
reference in ``vllm/model_executor/kernels/mhc/torch.py``.

Call as made by ``glm5_next_mhc_ops.glm5_mhc_pre`` off SM80 (verbatim):

    dsv4_mhc_pre(residual[T, 4, D], fn[24, 4D] f32, hc_scale[3] f32,
                 hc_base[24] f32, rms_eps, pre_eps=hc_eps,
                 sinkhorn_eps=hc_eps, post_multiplier=2.0,
                 sinkhorn_repeat=hc_sinkhorn_iters, norm_weight=None,
                 norm_eps=rms_eps)

with GLM-5.3-Flash: hc_mult 4, hc_eps 1e-6, hc_sinkhorn_iters 20,
rms_norm_eps 1e-5, hc_post_alpha 2.0 (the reference's ``2 * sigmoid``).
The Metal binding (csrc/quixicore/tm_metal/qc_metal_serving.mm) takes the
same positional arguments and rejects a norm weight, which is what the
GLM layer passes off SM80 (``_fuse_mhc_norm`` False => separate RMSNorm).
"""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.kernels.mhc.torch import mhc_post_torch, mhc_pre_torch

HC = 4
HC_EPS = 1e-6
SINKHORN_ITERS = 20
RMS_EPS = 1e-5
POST_MULT = 2.0


def _metal_mhc_available() -> bool:
    if not torch.backends.mps.is_available():
        return False
    try:
        from vllm.quixicore.ops import quixicore_ops
    except Exception:
        return False
    return quixicore_ops.is_available() and all(
        quixicore_ops.has(n)
        for n in ("dsv4_mhc_pre", "dsv4_mhc_fused_post_pre", "dsv4_mhc_post")
    )


needs_metal_mhc = pytest.mark.skipif(
    not _metal_mhc_available(), reason="QuixiCore-Metal dsv4_mhc_* not importable"
)


def _inputs(T, D, seed):
    g = torch.Generator().manual_seed(seed)
    residual = (torch.randn(T, HC, D, generator=g) * 0.8).to(torch.bfloat16)
    fn = torch.randn((2 + HC) * HC, HC * D, generator=g) * (HC * D) ** -0.5
    scale = torch.tensor([0.9, 1.1, 0.7])
    base = torch.randn((2 + HC) * HC, generator=g) * 0.3
    x = (torch.randn(T, D, generator=g) * 0.5).to(torch.bfloat16)
    return residual, fn, scale, base, x


def test_metal_route_is_quixicore_not_sm80_simt():
    from vllm.model_executor.layers import glm5_next_mhc_ops as ops
    from vllm.platforms import current_platform

    if current_platform.is_cuda():
        pytest.skip("CUDA box: the SM80 SIMT route is expected there")
    assert ops._USE_SM80_SIMT is False


@needs_metal_mhc
@pytest.mark.parametrize("T", [1, 5, 64, 130])
@pytest.mark.parametrize("D", [256, 4096])
@torch.no_grad()
def test_dsv4_mhc_pre_matches_torch_reference(T, D):
    from vllm.quixicore.ops import quixicore_ops

    residual, fn, scale, base, _ = _inputs(T, D, 1)
    dev = "mps"
    post, comb, li = quixicore_ops.dsv4_mhc_pre(
        residual.to(dev), fn.to(dev), scale.to(dev), base.to(dev),
        RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS, None, RMS_EPS,
    )
    post_r, comb_r, li_r = mhc_pre_torch(
        residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS
    )
    torch.testing.assert_close(post.cpu().reshape(T, HC), post_r.reshape(T, HC),
                               atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb.cpu().reshape(T, HC, HC), comb_r, atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(li.cpu().reshape(T, D).float(), li_r.float(),
                               atol=2e-2, rtol=2e-2)


@needs_metal_mhc
@pytest.mark.parametrize("T", [3, 64])
@torch.no_grad()
def test_registered_glm5_ops_on_mps(T, monkeypatch):
    """The torch.ops.vllm.glm5_mhc_{pre,fused_post_pre,post} ops the GLM
    decoder layer calls, with GLM's argument set, on MPS tensors; the SM80
    Triton transition must not be reached."""
    from vllm.model_executor.layers import glm5_next_mhc_ops as ops

    def _boom(*a, **k):
        raise AssertionError("SM80 SIMT mhc_transition reached on Metal")

    monkeypatch.setattr(ops, "mhc_transition", _boom)
    D = 4096
    dev = "mps"
    residual, fn, scale, base, x = _inputs(T, D, 2)
    r, f, s, b, xx = (t.to(dev) for t in (residual, fn, scale, base, x))
    post, comb, li = torch.ops.vllm.glm5_mhc_pre(
        r, f, s, b, RMS_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS, None, RMS_EPS
    )
    post_r, comb_r, li_r = mhc_pre_torch(
        residual, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS
    )
    assert post.shape == (T, HC, 1) and comb.shape == (T, HC, HC) and li.shape == (T, D)
    torch.testing.assert_close(post.cpu(), post_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb.cpu(), comb_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(li.cpu().float(), li_r.float(), atol=2e-2, rtol=2e-2)

    # post: residual remix with the previous site's placement
    out = torch.ops.vllm.glm5_mhc_post(xx, r, post, comb)
    out_r = mhc_post_torch(x, residual, post_r, comb_r)
    torch.testing.assert_close(out.cpu().float(), out_r.float(), atol=3e-2, rtol=2e-2)

    # fused post+pre == post then pre on the remixed residual
    res2, post2, comb2, li2 = torch.ops.vllm.glm5_mhc_fused_post_pre(
        xx, r, post, comb, f, s, b, RMS_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS,
        None, RMS_EPS,
    )
    post2_r, comb2_r, li2_r = mhc_pre_torch(
        out_r, fn, scale, base, RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN_ITERS
    )
    torch.testing.assert_close(res2.cpu().float(), out_r.float(), atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(post2.cpu(), post2_r, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(comb2.cpu(), comb2_r, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(li2.cpu().float(), li2_r.float(), atol=5e-2, rtol=3e-2)


def test_projection_overlap_never_enabled_off_sm80():
    """glm5_mhc_project_runtime (torch.cuda.Stream) is gated behind
    projection_enabled(sm80=...); Metal passes sm80=False."""
    from vllm.model_executor.layers.glm5_next_mhc_project import projection_enabled

    extra = {"glm5_next_mhc_projection_overlap": True}
    assert projection_enabled(
        extra, sm80=False, hidden_size=4096, hc_mult=4, dtype=torch.bfloat16,
        lora=False,
    ) is False
    assert projection_enabled(
        extra, sm80=True, hidden_size=4096, hc_mult=4, dtype=torch.bfloat16,
        lora=False,
    ) is True


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
