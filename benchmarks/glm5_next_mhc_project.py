# SPDX-License-Identifier: Apache-2.0
"""Serving-style opaque mHC+projection experiment; not imported by models.

The caller owns the side stream and passes its handle. Outputs are freshly
allocated, unlike the earlier fixed-buffer microbenchmark. All side-stream
writes join the calling stream before return. This covers fused post/pre
sites only; the first layer's pre-only transition remains unchanged.
"""

import torch

from benchmarks.glm5_next_mhc_deferred import _finalize_phase
from vllm.model_executor.layers.glm5_next_mhc_ops import glm5_mhc_fused_post_pre
from vllm.model_executor.layers.glm5_next_mhc_triton import _mhc_partials
from vllm.utils.torch_utils import direct_register_custom_op


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
    op_name="glm5_mhc_project_candidate",
    op_func=project_transition,
    mutates_args=[],
    fake_impl=_fake,
)
