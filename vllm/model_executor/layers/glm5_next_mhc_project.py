# SPDX-License-Identifier: Apache-2.0
"""Owned mHC transition + consumer projection with deferred coefficients.

For SM80 GLM-5.3 decode, overlap post/Sinkhorn work with urgent normalized
input and its BF16 or FP32 projection. The caller owns the side stream.
Fresh outputs and an explicit join keep all side work inside this opaque op.
Larger batches retain the original transition and projection.
"""

from dataclasses import dataclass

import torch

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.glm5_next_mhc_ops import glm5_mhc_fused_post_pre
from vllm.model_executor.layers.glm5_next_mhc_triton import _mhc_partials
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _finalize_phase(
    PART,
    RES,
    SCALE,
    BASE,
    POST,
    COMB,
    OUT,
    NORM,
    RMS_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    POST_MULT: tl.constexpr,
    ITERATIONS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    DEFERRED: tl.constexpr,
):
    t = tl.program_id(0)
    s, n = tl.arange(0, 64), tl.arange(0, 32)
    values = tl.load(
        PART + t * 64 * 32 + s[:, None] * 32 + n[None, :], n[None, :] < 25, 0
    )
    mixes = tl.sum(values, axis=0)
    sq = tl.sum(tl.where(n == 24, mixes, 0), axis=0)
    inv = tl.rsqrt(sq / 16384 + RMS_EPS)
    h = tl.arange(0, 4)
    if DEFERRED:
        scale1, scale2 = tl.load(SCALE + 1), tl.load(SCALE + 2)
        post = POST_MULT * tl.sigmoid(
            tl.gather(mixes, h + 4, 0) * inv * scale1 + tl.load(BASE + h + 4)
        )
        tl.store(POST + t * 4 + h, post)
        i = tl.arange(0, 16)
        matrix = (
            tl.gather(mixes, i + 8, 0) * inv * scale2 + tl.load(BASE + i + 8)
        ).reshape(4, 4)
        matrix = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
        matrix = matrix / tl.sum(matrix, axis=1)[:, None] + HC_EPS
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
        for _ in range(ITERATIONS - 1):
            matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + HC_EPS)
            matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + HC_EPS)
        tl.store(COMB + t * 16 + i, matrix.reshape(16))
    else:
        pre = (
            tl.sigmoid(
                tl.gather(mixes, h, 0) * inv * tl.load(SCALE) + tl.load(BASE + h)
            )
            + HC_EPS
        )
        d = tl.arange(0, 4096)
        layer = tl.full((4096,), 0, tl.float32)
        for stream in tl.static_range(4):
            coeff = tl.sum(tl.where(h == stream, pre, 0), axis=0)
            residual = tl.load(RES + t * 16384 + stream * 4096 + d).to(tl.float32)
            layer = tl.fma(coeff, residual, layer)
        rounded = layer.to(tl.bfloat16)
        v = rounded.to(tl.float32)
        norm = tl.rsqrt(tl.sum(v * v, axis=0) / 4096 + NORM_EPS)
        rounded = (v * norm * tl.load(NORM + d).to(tl.float32)).to(tl.bfloat16)
        tl.store(OUT + t * 4096 + d, rounded)


def project_transition(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    stream_handle: int,
    projection_fp32: bool,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    post_mult: float = 2.0,
    iterations: int = 20,
    norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    assert x.shape == (m, 4096) and residual.shape == (m, 4, 4096)
    assert fn.shape == (24, 16384) and norm_weight.shape == (4096,)
    assert projection_weight.ndim == 2 and projection_weight.shape[1] == 4096
    assert (
        x.dtype
        == residual.dtype
        == norm_weight.dtype
        == projection_weight.dtype
        == torch.bfloat16
    )
    assert (
        fn.dtype
        == scale.dtype
        == base.dtype
        == post.dtype
        == comb.dtype
        == torch.float32
    )
    assert all(
        t.is_contiguous() and t.device == x.device
        for t in (
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            norm_weight,
            projection_weight,
        )
    )
    assert post.numel() == m * 4 and comb.numel() == m * 16
    assert scale.shape == (3,) and base.shape == (24,)
    assert stream_handle > 0 and iterations >= 1

    projected = x.new_empty(
        (m, projection_weight.shape[0]),
        dtype=torch.float32 if projection_fp32 else x.dtype,
    )
    if m > 64:
        result = glm5_mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            rms_eps,
            hc_eps,
            post_mult,
            iterations,
            norm_weight,
            norm_eps,
        )
        layer = result[-1]
    else:
        res_out = torch.empty_like(residual)
        partial = x.new_empty((m, 64, 32), dtype=torch.float32)
        post_out = x.new_empty((m, 4, 1), dtype=torch.float32)
        comb_out = x.new_empty((m, 4, 4), dtype=torch.float32)
        layer = torch.empty_like(x)
        result = res_out, post_out, comb_out, layer
        if m == 0:
            return *result, projected
        _mhc_partials[(64, m)](
            x, residual, post, comb, fn, res_out, partial, True, num_warps=4
        )
        main = torch.cuda.current_stream(x.device)
        side = torch.cuda.ExternalStream(stream_handle, device=x.device)
        side.wait_stream(main)
        args = (
            partial,
            res_out,
            scale,
            base,
            post_out,
            comb_out,
            layer,
            norm_weight,
            rms_eps,
            hc_eps,
            post_mult,
            iterations,
            norm_eps,
        )
        with torch.cuda.stream(side):
            _finalize_phase[(m,)](*args, True, num_warps=4)
        _finalize_phase[(m,)](*args, False, num_warps=4)
    if projection_fp32:
        torch.mm(layer, projection_weight.T, out_dtype=torch.float32, out=projected)
    else:
        torch.mm(layer, projection_weight.T, out=projected)
    if m <= 64:
        main.wait_stream(side)
    return *result, projected


def _fake(
    x,
    residual,
    post,
    comb,
    fn,
    scale,
    base,
    norm_weight,
    projection_weight,
    stream_handle,
    projection_fp32,
    rms_eps=1e-5,
    hc_eps=1e-6,
    post_mult=2.0,
    iterations=20,
    norm_eps=1e-5,
):
    m = x.shape[0]
    return (
        torch.empty_like(residual),
        x.new_empty((m, 4, 1), dtype=torch.float32),
        x.new_empty((m, 4, 4), dtype=torch.float32),
        torch.empty_like(x),
        x.new_empty(
            (m, projection_weight.shape[0]),
            dtype=torch.float32 if projection_fp32 else x.dtype,
        ),
    )


direct_register_custom_op(
    op_name="glm5_mhc_project",
    op_func=project_transition,
    mutates_args=[],
    fake_impl=_fake,
)


@dataclass(frozen=True)
class MHCProjectionStream:
    """Runtime-only ownership; never serialize the CUDA handle into a graph."""

    stream: torch.cuda.Stream


def register_projection_stream(compilation_config, key: str, stream):
    if not key or key in compilation_config.static_forward_context:
        raise ValueError(f"Invalid or duplicate mHC stream name: {key!r}")
    compilation_config.static_forward_context[key] = MHCProjectionStream(stream)


def project_transition_runtime(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    stream_key: str,
    projection_fp32: bool,
    rms_eps: float = 1e-5,
    hc_eps: float = 1e-6,
    post_mult: float = 2.0,
    iterations: int = 20,
    norm_eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Like attention/MoE opaque ops, resolve from this engine's forward context
    # at execution/capture time. AOT/Inductor disk caches may retain stream_key,
    # but must never retain cuda_stream (a pointer from another process).
    owner = get_forward_context().no_compile_layers[stream_key]
    if not isinstance(owner, MHCProjectionStream):
        raise TypeError(f"Not an mHC stream owner: {stream_key!r}")
    if owner.stream.device != x.device:
        raise ValueError(f"mHC stream device does not match input: {stream_key!r}")
    return project_transition(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        base,
        norm_weight,
        projection_weight,
        owner.stream.cuda_stream,
        projection_fp32,
        rms_eps,
        hc_eps,
        post_mult,
        iterations,
        norm_eps,
    )


# New operator identity also prevents loading the old raw-pointer contract.
# The old operator remains only for isolated direct-handle kernel diagnostics.
direct_register_custom_op(
    op_name="glm5_mhc_project_runtime",
    op_func=project_transition_runtime,
    mutates_args=[],
    fake_impl=_fake,
)


def projection_enabled(extra, *, sm80, hidden_size, hc_mult, dtype, lora):
    """Strict opt-in; unsupported model/platform contracts retain old code."""
    enabled = (extra or {}).get("glm5_next_mhc_projection_overlap", False)
    if not isinstance(enabled, bool):
        raise ValueError("glm5_next_mhc_projection_overlap must be boolean")
    return (
        enabled
        and sm80
        and hidden_size == 4096
        and hc_mult == 4
        and dtype == torch.bfloat16
        and not lora
    )


def plain_bf16_projection(layer):
    """Do not bypass quantization, transforms, bias or TP output gathering."""
    weight = getattr(layer, "weight", None)
    return (
        type(getattr(layer, "quant_method", None)) is UnquantizedLinearMethod
        and isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.shape[1] == 4096
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and getattr(layer, "bias", None) is None
        and not getattr(layer, "gather_output", False)
    )


def router_projection_enabled(extra):
    enabled = (extra or {}).get("glm5_next_mhc_router_projection_overlap", False)
    if not isinstance(enabled, bool):
        raise ValueError("glm5_next_mhc_router_projection_overlap must be boolean")
    if enabled and not (extra or {}).get("glm5_next_mhc_projection_overlap", False):
        raise ValueError("router projection overlap requires mHC projection overlap")
    return enabled


def prepare_router_projection(moe):
    """Diagnostic opt-in: September9 TP8 A/B showed no serving benefit.

    Kept for overlap-window experiments; registered profiles disable it.
    Never call on an already compiled/captured/running model.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    gate, runner = moe.gate, moe.experts
    if (
        moe.is_sequence_parallel
        or not isinstance(runner, MoERunner)
        or not gate.allow_cublas_router_gemm
        or not plain_bf16_projection(gate)
        or runner.routed_input_transform is not None
        or runner.shared_expert_gate is not None
        or runner._fse_fuse_gate
        or runner.enable_dbo
    ):
        return False
    if runner.gate is not None and runner.gate is not gate:
        return False
    # The parent MoE still owns/loads the unchanged gate weights. Removing
    # only the runner's alias makes its supported external-router branch
    # consume the supplied logits without recomputing them. Shared-expert
    # streams, quantization and collectives retain their normal contracts.
    runner.gate = None
    return True
