# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 decode MoE pair (csrc/quixicore/serving/nvfp4_moe_decode_ampere.cuh):
gate/up + SiLU into the [M * top_k, N] intermediate, then down + the top-k
combine (+ the shared-expert add) into [M, K], over the very Marlin-packed
expert tensors the layer serves, against a plain-torch dequantization of the
NVFP4 checkpoint tensors (e2m1 x e4m3 group scale x global scale)."""

import types

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.quixicore.ops import quixicore_ops
from vllm.scalar_type import scalar_types

pytest.importorskip("vllm._quixicore_C")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_nvfp4_moe_decode()),
    reason="needs CUDA and the QuixiCore nvfp4_moe_gemv bindings",
)
DEV = "cuda"
E, K, N, TOPK = 16, 4096, 512, 8   # the record's per-rank expert geometry, fewer experts
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def dequant(u8: torch.Tensor, sc: torch.Tensor, gs: float) -> torch.Tensor:
    """[E, rows, cols/2] packed e2m1 (even element in the low nibble) x
    [E, rows, cols/16] e4m3 scales x global -> [E, rows, cols] fp32."""
    tbl = torch.tensor(E2M1, device=u8.device)
    v = torch.stack([tbl[(u8 & 15).long()], tbl[(u8 >> 4).long()]], dim=-1)
    v = v.reshape(u8.shape[0], u8.shape[1], -1)
    return v * sc.float().repeat_interleave(16, dim=2) * gs


class Layer:
    def __init__(self, seed: int = 0, gs: float = 1 / 300.0):
        torch.manual_seed(seed)
        w13 = torch.randint(0, 256, (E, 2 * N, K // 2), dtype=torch.uint8, device=DEV)
        s13 = (torch.rand(E, 2 * N, K // 16, device=DEV) * 2 + 0.5).to(torch.float8_e4m3fn)
        w2 = torch.randint(0, 256, (E, K, N // 2), dtype=torch.uint8, device=DEV)
        s2 = (torch.rand(E, K, N // 16, device=DEV) * 2 + 0.5).to(torch.float8_e4m3fn)
        self.W13, self.W2 = dequant(w13, s13, gs), dequant(w2, s2, gs)
        l = types.SimpleNamespace()
        l.moe_config = types.SimpleNamespace(
            num_experts=E, hidden_dim=K, intermediate_size_per_partition=N
        )
        l.params_dtype = torch.bfloat16
        l.w13_weight, l.w13_weight_scale = w13.clone(), s13.clone()
        l.w13_weight_scale_2 = torch.tensor(gs, device=DEV)
        l.w2_weight, l.w2_weight_scale = w2.clone(), s2.clone()
        l.w2_weight_scale_2 = torch.tensor(gs, device=DEV)
        prepare_moe_fp4_layer_for_marlin(l)
        self.l = l
        self.s13 = l.w13_weight_scale.view(torch.uint8)
        self.s2 = l.w2_weight_scale.view(torch.uint8)

    def reference(self, x, ids, w, clamp=None):
        m = x.shape[0]
        act = torch.empty(m * TOPK, N, device=DEV)
        out = torch.zeros(m, K, device=DEV)
        for t in range(m):
            for k in range(TOPK):
                e = int(ids[t, k])
                gu = x[t].float() @ self.W13[e].t()
                gate, up = gu[:N], gu[N:]
                if clamp is not None:   # silu_and_mul_with_clamp, act-first form
                    gate = gate.clamp(max=clamp)
                    up = up.clamp(min=-clamp, max=clamp)
                a = torch.nn.functional.silu(gate) * up
                act[t * TOPK + k] = a
                out[t] += w[t, k] * (a.to(torch.bfloat16).float() @ self.W2[e].t())
        return act, out


def route(m: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    ids = torch.stack([torch.randperm(E, device=DEV, generator=g)[:TOPK] for _ in range(m)])
    w = torch.rand(m, TOPK, device=DEV, generator=g, dtype=torch.float32)
    return ids.to(torch.int32).contiguous(), (w / w.sum(-1, keepdim=True)).contiguous()


def rel(a: torch.Tensor, ref: torch.Tensor) -> float:
    return ((a.float() - ref).abs() / ref.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)).max().item()


@pytest.mark.parametrize("m", [1, 2, 4])
@pytest.mark.parametrize("nj", [1, 2, 4])
def test_kernels_match_dequantized_reference(m: int, nj: int) -> None:
    lay = Layer()
    x = torch.randn(m, K, device=DEV, dtype=torch.bfloat16)
    ids, w = route(m, 7 + m)
    act_ref, out_ref = lay.reference(x, ids, w)
    act = torch.empty(m * TOPK, N, device=DEV, dtype=torch.bfloat16)
    quixicore_ops.nvfp4_moe_gemv1(x, lay.l.w13_weight, lay.s13, lay.l.w13_weight_scale_2, ids, act, nj)
    assert rel(act, act_ref) < 2**-7   # Marlin's bf16 fragment products, fp32 sums, one bf16 rounding
    out = torch.empty(m, K, device=DEV, dtype=torch.bfloat16)
    quixicore_ops.nvfp4_moe_gemv2(
        act_ref.to(torch.bfloat16), lay.l.w2_weight, lay.s2, lay.l.w2_weight_scale_2, ids, w, None, out, nj
    )
    assert rel(out, out_ref) < 2**-7
    again = torch.empty_like(out)
    quixicore_ops.nvfp4_moe_gemv2(
        act_ref.to(torch.bfloat16), lay.l.w2_weight, lay.s2, lay.l.w2_weight_scale_2, ids, w, None, again, nj
    )
    assert torch.equal(out, again), "the combine must be deterministic"


def test_gemv1_clamp_matches_silu_and_mul_with_clamp() -> None:
    lay = Layer(4)
    m = 2
    x = 4 * torch.randn(m, K, device=DEV, dtype=torch.bfloat16)   # wide enough that the clamp bites
    ids, w = route(m, 21)
    act_ref, _ = lay.reference(x, ids, w, clamp=10.0)
    act_plain, _ = lay.reference(x, ids, w)
    assert not torch.allclose(act_ref, act_plain), "the test must exercise the clamp"
    act = torch.empty(m * TOPK, N, device=DEV, dtype=torch.bfloat16)
    quixicore_ops.nvfp4_moe_gemv1(
        x, lay.l.w13_weight, lay.s13, lay.l.w13_weight_scale_2, ids, act, clamp_limit=10.0
    )
    assert rel(act, act_ref) < 2**-7


@pytest.mark.parametrize("m", [1, 3])
def test_fused_marlin_moe_dispatches_the_pair(m: int, monkeypatch) -> None:
    lay = Layer(1)
    x = 4 * torch.randn(m, K, device=DEV, dtype=torch.bfloat16)
    ids, w = route(m, 11 + m)
    _, out_ref = lay.reference(x, ids, w, clamp=10.0)
    calls = []
    real = quixicore_ops.nvfp4_moe_gemv2

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(quixicore_ops, "nvfp4_moe_gemv2", staticmethod(spy))
    monkeypatch.setattr(marlin_moe, "_qc_nvfp4_decode_rows", lambda: 32)
    out = marlin_moe.fused_marlin_moe(
        hidden_states=x,
        w1=lay.l.w13_weight,
        w2=lay.l.w2_weight,
        bias1=None,
        bias2=None,
        w1_scale=lay.l.w13_weight_scale,
        w2_scale=lay.l.w2_weight_scale,
        topk_weights=w,
        topk_ids=ids,
        quant_type_id=scalar_types.float4_e2m1f.id,
        global_num_experts=E,
        activation=MoEActivation.SILU,
        global_scale1=lay.l.w13_weight_scale_2,
        global_scale2=lay.l.w2_weight_scale_2,
        workspace=lay.l.workspace,
        clamp_limit=10.0,   # the record's SiLU clamp (GLM-5.3), folded into gemv1
    )
    assert calls == [1]
    assert out.shape == (m, K) and out.dtype == torch.bfloat16
    assert rel(out, out_ref) < 2**-6
    # Above the row cap the Marlin path serves the batch and the pair is not called.
    monkeypatch.setattr(marlin_moe, "_qc_nvfp4_decode_rows", lambda: m * TOPK - 1)
    marlin_moe.fused_marlin_moe(
        hidden_states=x, w1=lay.l.w13_weight, w2=lay.l.w2_weight, bias1=None, bias2=None,
        w1_scale=lay.l.w13_weight_scale, w2_scale=lay.l.w2_weight_scale, topk_weights=w, topk_ids=ids,
        quant_type_id=scalar_types.float4_e2m1f.id, global_num_experts=E, activation=MoEActivation.SILU,
        global_scale1=lay.l.w13_weight_scale_2, global_scale2=lay.l.w2_weight_scale_2, workspace=lay.l.workspace,
    )
    assert calls == [1]


def test_gemv2_shared_add_and_input_weights() -> None:
    lay = Layer(2)
    m = 3
    ids, w = route(m, 5)
    act = torch.randn(m * TOPK, N, device=DEV, dtype=torch.bfloat16)
    shared = torch.randn(m, K, device=DEV, dtype=torch.bfloat16)
    args = (act, lay.l.w2_weight, lay.s2, lay.l.w2_weight_scale_2, ids)
    plain = quixicore_ops.nvfp4_moe_gemv2(*args, w, None, torch.empty(m, K, device=DEV, dtype=torch.bfloat16))
    folded = quixicore_ops.nvfp4_moe_gemv2(*args, w, shared, torch.empty(m, K, device=DEV, dtype=torch.bfloat16))
    want = plain.float() + shared.float()
    assert ((folded.float() - want).abs() / want.abs().max()).max().item() < 2**-7
    ones = torch.ones_like(w)
    a = quixicore_ops.nvfp4_moe_gemv2(*args, ones, None, torch.empty(m, K, device=DEV, dtype=torch.bfloat16))
    b = quixicore_ops.nvfp4_moe_gemv2(*args, None, None, torch.empty(m, K, device=DEV, dtype=torch.bfloat16))
    assert torch.equal(a, b)


def test_rejects_bad_shapes() -> None:
    lay = Layer(3)
    ids, w = route(1, 1)
    x = torch.randn(1, K, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):   # act rows must be M * top_k
        quixicore_ops.nvfp4_moe_gemv1(
            x, lay.l.w13_weight, lay.s13, lay.l.w13_weight_scale_2, ids,
            torch.empty(TOPK + 1, N, device=DEV, dtype=torch.bfloat16),
        )
    with pytest.raises(RuntimeError):   # nj must be 1, 2 or 4
        quixicore_ops.nvfp4_moe_gemv1(
            x, lay.l.w13_weight, lay.s13, lay.l.w13_weight_scale_2, ids,
            torch.empty(TOPK, N, device=DEV, dtype=torch.bfloat16), 3,
        )
    with pytest.raises(RuntimeError):   # topk_weights must be fp32 [M, top_k]
        quixicore_ops.nvfp4_moe_gemv2(
            torch.randn(TOPK, N, device=DEV, dtype=torch.bfloat16), lay.l.w2_weight, lay.s2,
            lay.l.w2_weight_scale_2, ids, w[:, :1].contiguous(), None,
            torch.empty(1, K, device=DEV, dtype=torch.bfloat16),
        )
