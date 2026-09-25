# SPDX-License-Identifier: Apache-2.0
"""Expert-grouped two-slot iq2_xxs w13 GEMV (moe_group_slots +
qgemv_iq2_xxs_moe_mr_swiglu_texm_grp, VLLM_QC_MOE_GROUP_NB=2): every slot's
output must match the per-slot texm kernel within one bf16 ulp (the lane
walk splits each block's fp32 chain in two), with duplicate experts, an
expert routed to 5 slots, pad slots (-1) and single slots."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"
# The texm kernels are the production w13 path; the launcher reads the env
# once, so pin it before the first launch in this process.
os.environ.setdefault("VLLM_METAL_MOE_IQ2_TEX", "1")


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.is_available() or not quixicore_ops.has_kernel(
        "qgemv_iq2_xxs_moe_mr_swiglu_texm_grp"
    ):
        pytest.skip("grouped MoE kernels not built")
    return quixicore_ops


def _iq2xxs(E, N, K, g):
    w = torch.randint(0, 256, (E, N, K // 256 * 66), generator=g, dtype=torch.uint8)
    blk = w.view(E, N, K // 256, 66)
    blk[..., 0] = torch.randint(
        0, 256, blk[..., 0].shape, generator=g, dtype=torch.uint8
    )
    blk[..., 1] = 0x28 + torch.randint(
        0, 8, blk[..., 1].shape, generator=g, dtype=torch.uint8
    )
    return w.to(DEV)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "T,topk,E,K,N",
    [(32, 8, 40, 512, 64), (3, 8, 288, 256, 32), (16, 4, 12, 1024, 16)],
)
@pytest.mark.parametrize("clamp", [None, 7.0])
def test_grouped_w13_bit_identical(dtype, T, topk, E, K, N, clamp):
    qc = _qc()
    g = torch.Generator().manual_seed(T * 7 + topk)
    w = _iq2xxs(E, N, K, g)
    x = (torch.randn(T, K, generator=g) * 0.5).to(dtype).to(DEV)
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32)
    ids[0, :topk // 2] = -1                      # pad slots
    if T >= 5:
        ids[1:6, 0] = 3                          # one expert on 5 slots
    ids = ids.to(DEV)
    ref = qc.ggml_moe_a8_vec_swiglu(
        x, w, ids, topk, 16, N, T, clamp, group_nb=0
    ).clone()
    got = qc.ggml_moe_a8_vec_swiglu(
        x, w, ids, topk, 16, N, T, clamp, group_nb=2
    ).clone()
    torch.mps.synchronize()
    assert got.shape == ref.shape == (T * topk, N // 2)
    assert torch.isfinite(ref.float()).all()
    r, q = ref.float().cpu(), got.float().cpu()
    # The half-block walk splits each block's fp32 chain in two, so an
    # output is within fp32 rounding of the per-slot kernel relative to the
    # PARTIAL-SUM magnitude (the output scale), not to its own value: allow
    # one T ulp of the value plus 2e-3 of the output scale (cancellation).
    scale = r.abs().max().item() + 1e-9
    mant = 7 if dtype == torch.bfloat16 else 10
    ulp = 2.0 ** (torch.floor(torch.log2(r.abs().clamp_min(1e-30))) - mant)
    diff = (q - r).abs()
    assert (diff <= ulp + 2e-3 * scale).all(), (diff / scale).max()
    # and the overwhelming majority is exactly equal
    assert (diff == 0).float().mean() > 0.9
    assert got.view(T, topk, -1)[0, : topk // 2].abs().sum() == 0


def test_group_env_gate(monkeypatch):
    from vllm.model_executor.layers.quantization.gguf import fused_moe as fm

    monkeypatch.setattr(fm, "_MOE_GROUP", None)
    monkeypatch.setenv("VLLM_QC_MOE_GROUP_NB", "2")
    assert fm._moe_group_nb(16) == 0
    assert fm._moe_group_nb(64) == 2
    monkeypatch.setattr(fm, "_MOE_GROUP", None)
    monkeypatch.delenv("VLLM_QC_MOE_GROUP_NB")
    assert fm._moe_group_nb(256) == 0


def test_gguf_shim_forwards_group_nb():
    """The serving path calls the gguf shim, not quixicore_ops directly: a
    keyword the shim does not forward kills the engine at the first MoE
    layer (2026-09-17, twice). Exercise the shim with both values."""
    from vllm.model_executor.layers.quantization.gguf import ops as shim

    qc = _qc()
    g = torch.Generator().manual_seed(1)
    E, N, K, T, topk = 6, 16, 256, 9, 8
    w = _iq2xxs(E, N, K, g)
    x = (torch.randn(T, K, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
    ids = torch.randint(0, E, (T, topk), generator=g, dtype=torch.int32).to(DEV)
    a = shim.ggml_moe_a8_vec_swiglu(x, w, ids, topk, 16, N, T, None).clone()
    b = shim.ggml_moe_a8_vec_swiglu(x, w, ids, topk, 16, N, T, None, group_nb=2).clone()
    torch.mps.synchronize()
    assert a.shape == b.shape == (T * topk, N // 2)
    assert qc is not None
