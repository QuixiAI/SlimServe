# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""torch.compile-opaque dispatch for owned GLM mHC kernels.

The MHCPreOp/MHCPostOp/MHCFusedPostPreOp CustomOps call the quixicore
pybind entry points directly, which Dynamo cannot trace (DSV4's A100 model
is not compiled, so it never needed to). glm5_next runs under
support_torch_compile, so the same kernels are exposed here as registered
custom ops with fake implementations. Streams are [T, 4, D] bf16; fn is
float32 (or opt-in lossless BF16 storage) [(2+4)*4, 4*D]; scale float32 [3];
base float32 [(2+4)*4]. All fn operands are converted to FP32 inside the
kernels; accumulation and reduction precision do not change.
On A100, small batches use the split SIMT Triton transition; larger prefill
uses the QuixiCore CUDA implementation. Both can fuse the following RMSNorm.
"""

import torch

from slimserve.model_journal import instrument_mhc
from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

_HC = 4
# The new dispatch has been measured on A100 only. Preserve the existing
# native implementation on other CUDA architectures and on ROCm/Metal.
_USE_SM80_SIMT = current_platform.is_cuda() and current_platform.is_device_capability(
    (8, 0)
)


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


def validate_glm53_mhc_prefill_tc(config, bf16_fn: bool) -> None:
    """Fail before serving if an explicitly requested native path cannot run."""
    if not bf16_fn:
        raise ValueError("mHC tensor-core prefill requires lossless BF16 fn storage")
    expected = {
        "hidden_size": 4096, "hc_mult": 4, "rms_norm_eps": 1e-5,
        "hc_eps": 1e-6, "hc_sinkhorn_iters": 20,
    }
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise ValueError(
            "mHC tensor-core prefill requires the qualified GLM53 settings"
        )
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("mHC tensor-core prefill is qualified only on SM120")
    native = _qc()
    if not native.has_glm53_mhc_prefill_tc():
        raise RuntimeError("rebuild the native library for mHC tensor-core prefill")
    if native.get_glm53_mhc_prefill_tc() != 1:
        raise RuntimeError(
            "mHC tensor-core prefill is not enabled in the native runtime; "
            "restart the process"
        )


def glm5_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    post_mult: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T, hc, D = residual.shape
    if (
        _USE_SM80_SIMT
        and 0 < T <= 64
        and hc == 4
        and D == 4096
        and fn.dtype == torch.float32
    ):
        _, post, comb, layer_input = mhc_transition(
            None,
            residual,
            None,
            None,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )
    else:
        post, comb, layer_input = _qc().dsv4_mhc_pre(
            residual.view(-1, hc, D),
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )
    return post.reshape(T, hc, 1), comb.reshape(T, hc, hc), layer_input.reshape(T, D)


def _glm5_mhc_pre_fake(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_eps,
    post_mult,
    sinkhorn_iters,
    norm_weight=None,
    norm_eps=0.0,
):
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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T, hc, D = residual.shape
    if (
        _USE_SM80_SIMT
        and 0 < T <= 64
        and hc == 4
        and D == 4096
        and fn.dtype == torch.float32
    ):
        res, post, comb, layer_input = mhc_transition(
            x.view(-1, D),
            residual,
            post_mix.contiguous(),
            comb_mix.contiguous(),
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )
    else:
        res, post, comb, layer_input = _qc().dsv4_mhc_fused_post_pre(
            x.view(-1, D),
            residual.view(-1, hc, D),
            post_mix.view(-1, hc).contiguous(),
            comb_mix.view(-1, hc, hc).contiguous(),
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )
    return (
        res.reshape(T, hc, D),
        post.reshape(T, hc, 1),
        comb.reshape(T, hc, hc),
        layer_input.reshape(T, D),
    )


def _glm5_mhc_fused_post_pre_fake(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_eps,
    post_mult,
    sinkhorn_iters,
    norm_weight=None,
    norm_eps=0.0,
):
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
        x.view(-1, D),
        residual.view(-1, hc, D),
        post_mix.view(-1, hc).contiguous(),
        comb_mix.view(-1, hc, hc).contiguous(),
    )
    return out.reshape(T, hc, D)


def _glm5_mhc_post_fake(x, residual, post_mix, comb_mix):
    return residual.new_empty(residual.shape)


direct_register_custom_op(
    op_name="glm5_mhc_pre", op_func=instrument_mhc(glm5_mhc_pre), mutates_args=[],
    fake_impl=_glm5_mhc_pre_fake,
)
direct_register_custom_op(
    op_name="glm5_mhc_fused_post_pre", op_func=instrument_mhc(glm5_mhc_fused_post_pre),
    mutates_args=[], fake_impl=_glm5_mhc_fused_post_pre_fake,
)
direct_register_custom_op(
    op_name="glm5_mhc_post", op_func=instrument_mhc(glm5_mhc_post), mutates_args=[],
    fake_impl=_glm5_mhc_post_fake,
)
