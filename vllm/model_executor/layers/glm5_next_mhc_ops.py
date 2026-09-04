# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch.compile-opaque wrappers over the quixicore Ampere mHC kernels.

The MHCPreOp/MHCPostOp/MHCFusedPostPreOp CustomOps call the quixicore
pybind entry points directly, which Dynamo cannot trace (DSV4's A100 model
is not compiled, so it never needed to). glm5_next runs under
support_torch_compile, so the same kernels are exposed here as registered
custom ops with fake implementations. Streams are [T, 4, D] bf16; fn is
float32 [(2+4)*4, 4*D]; scale float32 [3]; base float32 [(2+4)*4].
"""

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_HC = 4


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
