# SPDX-License-Identifier: Apache-2.0
"""QuixiCore NVFP4 prefill MoE GEMM (csrc/quixicore/serving/nvfp4_moe_prefill_ampere.cuh)
against Marlin on identical packed weights: the direct GEMMs (w13 and the
topk-weighted w2) and the fused MoE end to end through fused_marlin_moe."""
import os

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.cuda.is_available(), reason="CUDA only"
)
E, K, N, TOPK = 16, 4096, 512, 8


class _Layer:
    pass


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0)
    dev = "cuda"
    layer = _Layer()
    layer.num_experts, layer.hidden_size = E, K
    layer.intermediate_size_per_partition, layer.params_dtype = N, torch.bfloat16
    w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=dev)
    w13_s = (torch.rand(E, 2 * N, K // 16, device=dev) * 0.9 + 0.1).to(torch.float8_e4m3fn)
    w2_s = (torch.rand(E, K, N // 16, device=dev) * 0.9 + 0.1).to(torch.float8_e4m3fn)
    w13_s2 = torch.rand(E, device=dev) * 0.03 + 0.005
    w2_s2 = torch.rand(E, device=dev) * 0.03 + 0.005
    return prepare_nvfp4_moe_layer_for_marlin(layer, w13, w13_s, w13_s2, w2, w2_s, w2_s2, True)


def _routing(M):
    tw, tids = torch.topk(torch.softmax(torch.randn(M, E, device="cuda"), -1), TOPK, dim=-1)
    return (tw / tw.sum(-1, keepdim=True)).float(), tids.int()


def _close(out, ref, rel_max=2e-2, mean_rel=3e-3):
    ref, out = ref.float(), out.float()
    d = (out - ref).abs()
    assert (d.max() / (ref.abs().max() + 1e-6)).item() < rel_max
    assert (d.sum() / (ref.abs().sum() + 1e-6)).item() < mean_rel


@pytest.mark.parametrize("M", [37, 500, 2048])
@pytest.mark.parametrize("stages", [2, 3, 4, 5])
def test_direct_gemms_match_marlin(weights, M, stages):
    import vllm._custom_ops as ops
    from vllm.quixicore.ops import quixicore_ops

    w13, w13_s, w13_s2, w2, w2_s, w2_s2 = weights
    dev = "cuda"
    ws = marlin_make_workspace_new(torch.device(dev), 4)
    qt = scalar_types.float4_e2m1f
    torch.manual_seed(M)
    tw, tids = _routing(M)

    def marlin(a, w, s, g, top_k, mul, size_m, size_n, size_k):
        sid, eid, npp = moe_align_block_size(tids, 64, E)
        c = torch.zeros(size_m * top_k, size_n, dtype=torch.bfloat16, device=dev)
        return ops.moe_wna16_marlin_gemm(
            a, c, w, None, s, None, g, None, None, None, ws, sid, eid, npp, tw,
            moe_block_size=64, top_k=top_k, mul_topk_weights=mul, b_q_type=qt,
            size_m=size_m, size_n=size_n, size_k=size_k, is_k_full=True,
            use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False,
        )

    def mine(a, w, s, g, top_k, mul, rows, size_n):
        sid, eid, npp = moe_align_block_size(tids, 128, E)
        c = torch.zeros(rows, size_n, dtype=torch.bfloat16, device=dev)
        return quixicore_ops.nvfp4_moe_gemm(
            a, w, s.view(torch.uint8), g, sid, eid, npp, tw if mul else None, c, top_k, mul, stages
        )

    x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
    _close(
        mine(x, w13, w13_s, w13_s2, TOPK, False, M * TOPK, 2 * N),
        marlin(x, w13, w13_s, w13_s2, TOPK, False, M, 2 * N, K),
    )
    inter = (torch.randn(M * TOPK, N, device=dev) * 0.5).to(torch.bfloat16)
    _close(
        mine(inter, w2, w2_s, w2_s2, 1, True, M * TOPK, K),
        marlin(inter, w2, w2_s, w2_s2, 1, True, M * TOPK, K, N),
    )


@pytest.mark.parametrize("M", [300, 1500])
def test_fused_moe_matches_marlin_path(weights, M, monkeypatch):
    w13, w13_s, w13_s2, w2, w2_s, w2_s2 = weights
    torch.manual_seed(M)
    x = (torch.randn(M, K, device="cuda") * 0.5).to(torch.bfloat16)
    tw, tids = _routing(M)

    def run():
        out = torch.empty_like(x)
        return fused_marlin_moe(
            x, w13, w2, None, None, w13_s, w2_s, tw, tids,
            quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E,
            activation=MoEActivation.SILU, global_scale1=w13_s2, global_scale2=w2_s2,
            output=out,
        )

    monkeypatch.setenv("VLLM_QC_NVFP4_PREFILL_MOE_MIN_ROWS", "0")
    ref = run()
    monkeypatch.setenv("VLLM_QC_NVFP4_PREFILL_MOE_MIN_ROWS", "1")
    assert M * TOPK >= E  # the threshold admits this batch
    _close(run(), ref)
