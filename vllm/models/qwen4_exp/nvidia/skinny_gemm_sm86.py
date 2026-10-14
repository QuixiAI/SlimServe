# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp decode GEMMs on Ampere (SM 8.6): weight-stationary split-K Triton
kernels for the bf16 dense projections at decode token counts.

cuBLAS serves these shapes inside the FULL decode graph at 320-600 GB/s of
the 844 GB/s measured copy roofline on RTX 3090, and regresses to ~420-470
GB/s on the 1536-wide and 2560x2048 shapes at M=24 (perf/results/2026-09-07/
qwen38fn-nvfp4-4-opt/skinny_gemm_graph.out). One program per (BLOCK_N,
K-split) tile with 128-bit loads, fp32 accumulation and an fp32 atomic
split-K reduce holds 725-965 GB/s: 1.01-1.27x at M=3 and 1.12-1.78x at M=24.

Opt-in per process through VLLM_QWEN4_EXP_SKINNY_GEMM=1 (a registered vLLM env
var, hence part of the torch.compile cache key), plus VLLM_QWEN4_EXP_SKINNY_W8=1
to store the routed weights as int8 (per-row, per-group scales) and halve the
decode weight stream; the Blackwell
route (low_latency_gemm.py) is the model for the hook: unquantized bf16
linears whose (N, K) has a plan get their quant_method swapped, and token
counts outside the plan fall back to torch.nn.functional.linear.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import vllm.envs as envs
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MAX_M = 128  # batched-decode row counts (max_num_seqs x (k+1), 96 at 32 x 3); larger M -> cuBLAS


@dataclass(frozen=True)
class SkinnyCfg:
    block_n: int
    block_k: int
    split_k: int
    num_warps: int = 4
    num_stages: int = 3


# (N, K) -> [(max_m, config-or-None)], first bucket with M <= max_m wins;
# None routes that bucket to cuBLAS. From the 2026-09-07 per-M sweep on GPU 4
# with the single-launch split-K kernel (skinny_sweep_w8.out/json, CUDA-graph
# replay, 2e-2 parity; cuBLAS only when within 3% of the best config, which
# no bucket met). K must divide split_k * block_k.
_C = SkinnyCfg
SM86_SKINNY_PLANS: dict[tuple[int, int], list[tuple[int, SkinnyCfg | None]]] = {
    (24, 2560): [(4, _C(16, 64, 8)), (8, _C(16, 64, 8)), (16, _C(16, 32, 8)), (24, _C(16, 64, 8)), (32, _C(16, 64, 8))],
    (320, 2560): [(4, _C(16, 128, 4)), (8, _C(16, 128, 4)), (16, _C(16, 32, 8)), (24, _C(16, 128, 4)), (32, _C(16, 128, 4))],
    (320, 10240): [(4, _C(16, 256, 4)), (8, _C(16, 256, 4)), (16, _C(16, 256, 4)), (24, _C(16, 256, 4)), (32, _C(16, 256, 4))],
    (336, 10240): [(4, _C(32, 256, 4)), (8, _C(32, 256, 4)), (16, _C(32, 256, 4)), (24, _C(32, 256, 4, num_warps=8)), (32, _C(32, 256, 4, num_warps=8))],
    (512, 2560): [(4, _C(32, 128, 4)), (8, _C(32, 128, 4)), (16, _C(32, 128, 4)), (24, _C(32, 128, 4, num_warps=8)), (32, _C(32, 128, 4, num_warps=8))],
    (640, 2560): [(4, _C(32, 128, 4)), (8, _C(32, 128, 4)), (16, _C(32, 128, 4)), (24, _C(32, 128, 4, num_warps=8)), (32, _C(32, 128, 4, num_warps=8))],
    (2048, 2560): [(4, _C(32, 128, 1)), (8, _C(32, 128, 1)), (16, _C(32, 256, 1)), (24, _C(32, 256, 1, num_warps=8)), (32, _C(32, 256, 1))],
    (2560, 160): [(4, _C(64, 32, 1)), (8, _C(64, 32, 1)), (16, _C(64, 32, 1)), (24, _C(32, 32, 1)), (32, _C(64, 32, 1, num_warps=8))],
    (2560, 1536): [(4, _C(32, 128, 1)), (8, _C(32, 128, 1)), (16, _C(32, 128, 1)), (24, _C(32, 128, 1, num_warps=8)), (32, _C(32, 256, 1))],
    (2560, 2560): [(4, _C(32, 128, 1)), (8, _C(32, 128, 1)), (16, _C(32, 128, 1)), (24, _C(32, 128, 1, num_warps=8)), (32, _C(32, 256, 1, num_warps=8))],
    (3584, 2560): [(4, _C(64, 128, 1, num_warps=8)), (8, _C(32, 128, 1)), (16, _C(64, 128, 1, num_warps=8)), (24, _C(64, 128, 1, num_warps=8)), (32, _C(64, 128, 1, num_warps=8))],
    (4096, 2560): [(4, _C(32, 64, 1)), (8, _C(64, 128, 1, num_warps=8)), (16, _C(64, 128, 1, num_warps=8)), (24, _C(64, 128, 1, num_warps=8)), (32, _C(64, 128, 1, num_warps=8))],
    (10240, 320): [(4, _C(64, 64, 1)), (8, _C(64, 64, 1)), (16, _C(32, 32, 1)), (24, _C(64, 64, 1, num_warps=8)), (32, _C(64, 64, 1, num_warps=8))],
    (62080, 2560): [(4, _C(64, 128, 1, num_warps=8)), (8, _C(32, 128, 1)), (16, _C(64, 128, 1, num_warps=8)), (24, _C(128, 128, 1, num_warps=8)), (32, _C(128, 128, 1, num_warps=8))],
}


def plan_for(shape: tuple[int, int], m: int) -> SkinnyCfg | None:
    buckets = SM86_SKINNY_PLANS.get(shape)
    if buckets is None or m < 1 or m > MAX_M:
        return None
    for max_m, cfg in buckets:
        if m <= max_m:
            return cfg
    return None


@triton.jit
def _skinny_gemm_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr, ws_ptr, counter_ptr, M, N, K, n_groups,
    stride_xm, stride_wn, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_K: tl.constexpr = 0,
):
    """y[M, N] = x[M, K] @ w[N, K]^T, one program per (N-tile, K-split).

    GROUP_K == 0: w is bf16. GROUP_K > 0: w is int8 with one fp16 scale per
    (row, GROUP_K columns) in s_ptr [N, n_groups]; every BLOCK_K chunk lies
    inside one group (host asserts GROUP_K % BLOCK_K == 0), so the chunk's
    integer-valued dot product is scaled once on the fp32 accumulator and the
    weight stream is half the bytes of bf16 (E8, 2026-09-07).

    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    n_mask = offs_n < N
    m_mask = offs_m < M
    k_per = K // SPLIT_K
    k0 = pid_k * k_per
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k0 + k_per, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=m_mask[:, None], other=0.0)
        if GROUP_K > 0:
            w8 = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0)
            s = tl.load(s_ptr + offs_n * n_groups + k // GROUP_K, mask=n_mask, other=0.0).to(tl.float32)
            w = w8.to(tl.float32).to(tl.bfloat16)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32) * s[None, :]
        else:
            w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0.0)
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(y_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        # Single-launch split-K: every program stores its fp32 partial into
        # the workspace slice [pid_k, BLOCK_M, BLOCK_N] of its N-tile, then
        # bumps the tile's arrival counter; the last arrival sums the SPLIT_K
        # partials, writes bf16 and resets the counter for the next call.
        # No zero-fill, no separate convert launch (the previous atomic-add
        # version cost two extra launches per call: 2026-09-07 census).
        part_ptrs = (
            ws_ptr
            + ((pid_n * SPLIT_K + pid_k) * BLOCK_M + offs_m[:, None]) * BLOCK_N
            + tl.arange(0, BLOCK_N)[None, :]
        )
        tl.store(part_ptrs, acc)
        tl.debug_barrier()
        arrived = tl.atomic_add(counter_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_part in tl.static_range(SPLIT_K):
                p = (
                    ws_ptr
                    + ((pid_n * SPLIT_K + k_part) * BLOCK_M + offs_m[:, None]) * BLOCK_N
                    + tl.arange(0, BLOCK_N)[None, :]
                )
                total += tl.load(p, volatile=True)
            tl.store(y_ptrs, total.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(counter_ptr + pid_n, 0, sem="release", scope="gpu")


_WORKSPACES: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _split_k_workspace(
    device: torch.device, n_tiles: int, split_k: int, block_m: int, block_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent per-(device, geometry) fp32 partial buffer and int32 tile
    counters (zeroed once; the kernel resets them). Distinct geometries
    never share a buffer, and calls of the same geometry are stream-ordered
    inside the decode graph, so the buffers are safe to reuse."""
    key = (device.index, n_tiles, split_k, block_m, block_n)
    ws = _WORKSPACES.get(key)
    if ws is None:
        ws = (
            torch.empty(n_tiles * split_k * block_m * block_n, dtype=torch.float32, device=device),
            torch.zeros(n_tiles, dtype=torch.int32, device=device),
        )
        _WORKSPACES[key] = ws
    return ws


def skinny_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    cfg: SkinnyCfg,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """y[M, N] = x[M, K] @ weight[N, K]^T (bf16 out, fp32 accumulate).
    weight is bf16, or int8 with `scale` [N, K // group] fp16 (see
    quantize_w8)."""
    M, K = x.shape
    N = weight.shape[0]
    # Tile rows: 16/32 at decode counts, 64/128 for the batched-decode
    # counts of 16-32 running requests (M = seqs x (k+1) up to 96), so one
    # weight stream serves every row instead of a per-call int8 dequant.
    block_m = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    n_tiles = triton.cdiv(N, cfg.block_n)
    grid = (n_tiles, cfg.split_k)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    if cfg.split_k == 1:
        ws, counters = y, y  # unused
    else:
        ws, counters = _split_k_workspace(x.device, n_tiles, cfg.split_k, block_m, cfg.block_n)
    if scale is None:
        group_k, n_groups, s = 0, 0, y
    else:
        n_groups = scale.shape[1]
        group_k = K // n_groups
        assert group_k % cfg.block_k == 0, (group_k, cfg)
        s = scale
    _skinny_gemm_kernel[grid](
        x, weight, s, y, ws, counters, M, N, K, n_groups,
        x.stride(0), weight.stride(0), y.stride(0),
        BLOCK_M=block_m, BLOCK_N=cfg.block_n, BLOCK_K=cfg.block_k, SPLIT_K=cfg.split_k,
        GROUP_K=group_k,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return y


# -- E8: int8 weight-only storage for the routed linears -------------------
# Symmetric per-(row, group) absmax/127 quantization done once after weight
# loading (process_weights_after_loading); the decode kernel streams int8
# and dequantizes in registers, prefill token counts dequantize to a bf16
# scratch and use cuBLAS. Group = the largest of 128/64/32 dividing K.
W8_DEQUANT_MAX_BYTES = 64 << 20  # bigger weights (lm_head) chunk through the kernel


def w8_group_for_k(k: int) -> int:
    for g in (128, 64, 32):
        if k % g == 0:
            return g
    raise ValueError(f"K={k} not a multiple of 32")


def quantize_w8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [N, K] -> (int8 [N, K], fp16 scales [N, K // group])."""
    n, k = weight.shape
    g = w8_group_for_k(k)
    wf = weight.float().view(n, k // g, g)
    scale = wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    q = torch.round(wf / scale).clamp_(-127, 127).to(torch.int8).view(n, k)
    return q.contiguous(), scale.view(n, k // g).to(torch.float16).contiguous()


@triton.jit
def _dequant_w8_kernel(
    w_ptr, s_ptr, y_ptr, N, K, n_groups,
    GROUP_K: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n < N)[:, None] & (offs_k < K)[None, :]
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask, other=0).to(tl.float32)
    s = tl.load(s_ptr + offs_n[:, None] * n_groups + offs_k[None, :] // GROUP_K, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_n[:, None] * K + offs_k[None, :], (w * s).to(tl.bfloat16), mask=mask)


def dequantize_w8(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = w8.shape
    n_groups = scale.shape[1]
    y = torch.empty((n, k), dtype=torch.bfloat16, device=w8.device)
    grid = (triton.cdiv(n, 64), triton.cdiv(k, 256))
    _dequant_w8_kernel[grid](w8, scale, y, n, k, n_groups, GROUP_K=k // n_groups, BLOCK_N=64, BLOCK_K=256, num_warps=4)
    return y


# (N, K) -> [(max_m, config)] for int8 weights; block_k must divide the
# group. Only the shapes where int8 beat the best bf16 config by >=10% at
# M=3 without losing >5% at M=24 (skinny_sweep_w8.out: 0.52-0.64x at M<=8,
# 0.8-0.9x at M=24-32; the <=7 MB shapes are latency-bound and stay bf16).
# Shapes without an entry derive one from the bf16 plan (tests only).
SM86_SKINNY_W8_PLANS: dict[tuple[int, int], list[tuple[int, SkinnyCfg]]] = {
    (2048, 2560): [(4, _C(64, 128, 2, num_warps=8)), (8, _C(32, 128, 2)), (16, _C(64, 128, 2, num_warps=8)), (24, _C(64, 128, 4)), (32, _C(64, 128, 2)), (64, _C(64, 128, 2, num_warps=8)), (128, _C(32, 64, 2, num_warps=8))],
    (2560, 1536): [(4, _C(32, 128, 1)), (8, _C(32, 128, 1)), (16, _C(64, 128, 2, num_warps=8)), (24, _C(64, 128, 2)), (32, _C(64, 128, 2, num_warps=8)), (64, _C(32, 64, 2)), (128, _C(32, 64, 2, num_warps=8))],
    (2560, 2560): [(4, _C(32, 128, 1)), (8, _C(32, 128, 1)), (16, _C(64, 128, 2, num_warps=8)), (24, _C(64, 128, 2)), (32, _C(64, 128, 2, num_warps=8)), (64, _C(64, 128, 2, num_warps=8)), (128, _C(32, 64, 2, num_warps=8))],
    (3584, 2560): [(4, _C(64, 128, 1, num_warps=8)), (8, _C(64, 128, 1, num_warps=8)), (16, _C(64, 128, 1, num_warps=8)), (24, _C(32, 128, 2)), (32, _C(32, 128, 2)), (64, _C(32, 64, 2)), (128, _C(32, 64, 2, num_warps=8))],
    (4096, 2560): [(4, _C(64, 128, 1, num_warps=8)), (8, _C(64, 128, 1, num_warps=8)), (16, _C(64, 128, 1, num_warps=8)), (24, _C(64, 128, 1)), (32, _C(64, 128, 1)), (64, _C(32, 64, 1)), (128, _C(64, 128, 1, num_warps=8))],
    (10240, 320): [(4, _C(128, 64, 1)), (8, _C(128, 64, 1)), (16, _C(128, 64, 1)), (24, _C(128, 64, 1, num_warps=8)), (32, _C(128, 32, 1)), (64, _C(128, 64, 1)), (128, _C(64, 64, 1))],
    (62080, 2560): [(4, _C(64, 128, 1, num_warps=8)), (8, _C(64, 128, 1, num_warps=8)), (16, _C(128, 128, 1, num_warps=8)), (24, _C(128, 128, 1)), (32, _C(128, 128, 1)), (64, _C(128, 64, 1)), (128, _C(64, 64, 1))],
}


def plan_for_w8(shape: tuple[int, int], m: int) -> SkinnyCfg | None:
    if m < 1 or m > MAX_M:
        return None
    buckets = SM86_SKINNY_W8_PLANS.get(shape)
    if buckets is not None:
        for max_m, cfg in buckets:
            if m <= max_m:
                return cfg
        # No bucket this wide yet: the widest one still beats the per-call
        # dequant fallback (2026-09-07 sweep, 64/128-row tiles).
        return buckets[-1][1]
    bf16 = SM86_SKINNY_PLANS.get(shape)
    if bf16 is None:
        return None
    group = w8_group_for_k(shape[1])
    cfg = next((c for max_m, c in bf16 if m <= max_m and c is not None), None)
    if cfg is None:
        cfg = next((c for _, c in bf16 if c is not None), None)
    if cfg is None:
        return None
    if cfg.block_k > group or group % cfg.block_k:
        cfg = SkinnyCfg(cfg.block_n, group, cfg.split_k, cfg.num_warps, cfg.num_stages)
    return cfg



def _is_sm86() -> bool:
    return current_platform.is_device_capability((8, 6))


def _packed_row_major(t: torch.Tensor) -> bool:
    return t.dim() == 2 and t.stride() == (t.shape[1], 1)


def _qwen4_exp_skinny_gemm_sm86(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    cfg = plan_for((weight.shape[0], weight.shape[1]), x2.shape[0])
    if (
        cfg is not None
        and x2.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and _packed_row_major(x2)
        and _packed_row_major(weight)
        and weight.shape[1] % (cfg.split_k * cfg.block_k) == 0
    ):
        return skinny_gemm(x2, weight, cfg).reshape(*x.shape[:-1], weight.shape[0])
    return torch.nn.functional.linear(x, weight)


def _qwen4_exp_skinny_gemm_sm86_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _qwen4_exp_skinny_gemm_w8_sm86(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8 weight-only linear: skinny kernel at decode token counts, bf16
    dequant + cuBLAS above (weights over W8_DEQUANT_MAX_BYTES chunk their
    rows through the kernel instead of materializing the bf16 copy)."""
    n, k = w8.shape
    x2 = x.reshape(-1, k)
    m = x2.shape[0]
    ok = x2.dtype == torch.bfloat16 and x2.stride(1) == 1 and _packed_row_major(w8)
    if ok and m <= MAX_M:
        cfg = plan_for_w8((n, k), m)
        if cfg is not None and k % (cfg.split_k * cfg.block_k) == 0:
            return skinny_gemm(x2, w8, cfg, scale=scale).reshape(*x.shape[:-1], n)
    if ok and n * k > W8_DEQUANT_MAX_BYTES:
        cfg = plan_for_w8((n, k), MAX_M)
        if cfg is not None and k % (cfg.split_k * cfg.block_k) == 0:
            parts = [
                skinny_gemm(x2[i : i + MAX_M], w8, cfg, scale=scale) for i in range(0, m, MAX_M)
            ]
            return torch.cat(parts, dim=0).reshape(*x.shape[:-1], n)
    return torch.nn.functional.linear(x, dequantize_w8(w8, scale))


def _qwen4_exp_skinny_gemm_w8_sm86_fake(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w8.shape[0]))


direct_register_custom_op(
    op_name="qwen4_exp_skinny_gemm_w8_sm86",
    op_func=_qwen4_exp_skinny_gemm_w8_sm86,
    fake_impl=_qwen4_exp_skinny_gemm_w8_sm86_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_skinny_gemm_sm86",
    op_func=_qwen4_exp_skinny_gemm_sm86,
    fake_impl=_qwen4_exp_skinny_gemm_sm86_fake,
)


class _SkinnyApply:
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)  # type: ignore[misc]
        weight = getattr(layer, "weight", None)
        if not (
            envs.VLLM_QWEN4_EXP_SKINNY_W8
            and weight is not None
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
            and (weight.shape[0], weight.shape[1]) in SM86_SKINNY_W8_PLANS
        ):
            return  # shapes without a measured int8 win stay bf16
        w8, scale = quantize_w8(weight.data)
        layer.weight = nn.Parameter(w8, requires_grad=False)
        layer.weight_scale_sm86 = nn.Parameter(scale, requires_grad=False)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if bias is None:
            weight = layer.weight
            if weight.dtype == torch.int8:
                return torch.ops.vllm.qwen4_exp_skinny_gemm_w8_sm86(x, weight, layer.weight_scale_sm86)
            return torch.ops.vllm.qwen4_exp_skinny_gemm_sm86(x, weight)
        return super().apply(layer, x, bias)  # type: ignore[misc]


class Qwen4ExpSkinnyLinearMethodSM86(_SkinnyApply, UnquantizedLinearMethod):
    pass


class Qwen4ExpSkinnyEmbeddingMethodSM86(_SkinnyApply, UnquantizedEmbeddingMethod):
    pass


def enable_qwen4_exp_skinny_gemm_sm86(module: nn.Module, dtype: torch.dtype) -> list[tuple[int, int]]:
    """Swap the plan-covered bf16 linears to the Triton route. Returns the
    (N, K) shapes routed (empty when the gate is closed)."""
    if not envs.VLLM_QWEN4_EXP_SKINNY_GEMM:
        return []
    if dtype != torch.bfloat16 or not _is_sm86():
        return []
    routed: list[tuple[int, int]] = []
    for child in module.modules():
        is_linear = isinstance(child, LinearBase) and type(child.quant_method) is UnquantizedLinearMethod
        is_head = isinstance(child, ParallelLMHead) and type(child.quant_method) is UnquantizedEmbeddingMethod
        if not (is_linear or is_head):
            continue
        weight = getattr(child, "weight", None)
        if weight is None or weight.dim() != 2:
            continue
        shape = (weight.shape[0], weight.shape[1])
        if shape not in SM86_SKINNY_PLANS:
            continue
        child.quant_method = Qwen4ExpSkinnyLinearMethodSM86() if is_linear else Qwen4ExpSkinnyEmbeddingMethodSM86()
        routed.append(shape)
    return routed
