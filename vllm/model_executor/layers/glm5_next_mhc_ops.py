# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch.compile-opaque wrappers over the quixicore Ampere mHC kernels.

The MHCPreOp/MHCPostOp/MHCFusedPostPreOp CustomOps call the quixicore
pybind entry points directly, which Dynamo cannot trace (DSV4's A100 model
is not compiled, so it never needed to). glm5_next runs under
support_torch_compile, so the same kernels are exposed here as registered
custom ops with fake implementations. Streams are [T, 4, D] bf16; fn is
float32 (or opt-in lossless BF16 storage) [(2+4)*4, 4*D]; scale float32 [3];
base float32 [(2+4)*4]. All fn operands are converted to FP32 inside the
kernels; accumulation and reduction precision do not change.
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_HC = 4


def load_lossless_mhc_fn(param: torch.Tensor, weight: torch.Tensor) -> None:
    """Keep original BF16 fn values without accepting a new quantization.

    Check before copying so a non-representable source cannot partially replace
    the parameter. Native FP32 base/scale tensors do not use this loader.
    """
    if param.dtype != torch.bfloat16:
        raise ValueError("BF16 mHC fn storage requires a BF16 parameter")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("BF16 mHC fn storage requires BF16/FP32 source weights")
    if param.shape != weight.shape:
        raise ValueError("mHC fn checkpoint and parameter shapes differ")
    narrow = weight.to(torch.bfloat16)
    if not torch.isfinite(weight).all() or not torch.equal(
        weight, narrow.to(weight.dtype)
    ):
        raise ValueError("BF16 mHC fn storage would alter checkpoint values")
    param.data.copy_(narrow)


def _qc():
    from vllm.quixicore import quixicore_ops

    return quixicore_ops


def glm5_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, hc, D = residual.shape
    post, comb, layer_input = _qc().dsv4_mhc_pre(
        residual.view(-1, hc, D), fn, hc_scale, hc_base, rms_eps, hc_eps,
        hc_eps, post_mult, sinkhorn_iters, None, 0.0,
    )
    return post.reshape(T, hc, 1), comb.reshape(T, hc, hc), layer_input.reshape(T, D)


def _glm5_mhc_pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_eps,
                       post_mult, sinkhorn_iters):
    T, hc, D = residual.shape
    return (
        residual.new_empty((T, hc, 1), dtype=torch.float32),
        residual.new_empty((T, hc, hc), dtype=torch.float32),
        residual.new_empty((T, D)),
    )


def glm5_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, hc, D = residual.shape
    res, post, comb, layer_input = _qc().dsv4_mhc_fused_post_pre(
        x.view(-1, D), residual.view(-1, hc, D),
        post_mix.view(-1, hc).contiguous(), comb_mix.view(-1, hc, hc).contiguous(),
        fn, hc_scale, hc_base, rms_eps, hc_eps, hc_eps, post_mult,
        sinkhorn_iters, None, 0.0,
    )
    return (
        res.reshape(T, hc, D),
        post.reshape(T, hc, 1),
        comb.reshape(T, hc, hc),
        layer_input.reshape(T, D),
    )


def _glm5_mhc_fused_post_pre_fake(x, residual, post_mix, comb_mix, fn, hc_scale,
                                  hc_base, rms_eps, hc_eps, post_mult,
                                  sinkhorn_iters):
    T, hc, D = residual.shape
    return (
        residual.new_empty((T, hc, D)),
        residual.new_empty((T, hc, 1), dtype=torch.float32),
        residual.new_empty((T, hc, hc), dtype=torch.float32),
        residual.new_empty((T, D)),
    )


def glm5_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
) -> torch.Tensor:
    T, hc, D = residual.shape
    out = _qc().dsv4_mhc_post(
        x.view(-1, D), residual.view(-1, hc, D),
        post_mix.view(-1, hc).contiguous(), comb_mix.view(-1, hc, hc).contiguous(),
    )
    return out.reshape(T, hc, D)


def _glm5_mhc_post_fake(x, residual, post_mix, comb_mix):
    return residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="glm5_mhc_pre", op_func=glm5_mhc_pre, mutates_args=[],
    fake_impl=_glm5_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="glm5_mhc_fused_post_pre", op_func=glm5_mhc_fused_post_pre,
    mutates_args=[], fake_impl=_glm5_mhc_fused_post_pre_fake,
)
direct_register_custom_op(
    op_name="glm5_mhc_post", op_func=glm5_mhc_post, mutates_args=[],
    fake_impl=_glm5_mhc_post_fake,
)
