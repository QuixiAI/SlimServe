# SPDX-License-Identifier: Apache-2.0
"""GLM mHC transition fused with the producer's tensor-parallel all-reduce.

``glm5_mhc_ar_transition`` takes the tensor-parallel partial of the
projection feeding a transition site (o_proj or the MoE output, ``[T, D]``
bf16, not yet reduced), reduces it across the ranks and runs the transition
in the same launch (``quixicore/serving/glm5_mhc_allreduce.cuh`` through the
custom all-reduce). Batches beyond the fused kernel's limit, eager steps
and inputs the custom all-reduce does not serve fall back to the plain
all-reduce followed by the split transition, after joining the previous
site's deferred sinkhorn. Opt-in per profile through ``additional_config``
``glm5_next_mhc_allreduce_fusion``; a build without the
``glm5_mhc_allreduce`` binding rejects the option.
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.glm5_next_mhc_ops import (
    glm5_mhc_fused_post_pre,
    join_glm5_mhc_deferred,
)
from vllm.utils.torch_utils import direct_register_custom_op

_KEY = "glm5_next_mhc_allreduce_fusion"


def has_binding() -> bool:
    from vllm.quixicore.ops import quixicore_ops

    return quixicore_ops.has_glm5_mhc_allreduce()


def mhc_allreduce_fusion_enabled(extra) -> bool:
    """Strict opt-in; a record that asks for it must get it."""
    enabled = (extra or {}).get(_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"{_KEY} must be boolean")
    if not enabled:
        return False
    if get_tensor_model_parallel_world_size() == 1:
        return False
    if not has_binding():
        raise ValueError(
            f"{_KEY} is set but this build has no glm5_mhc_allreduce binding"
        )
    return True


def _ca_comm():
    comm = get_tp_group().device_communicator
    ca_comm = getattr(comm, "ca_comm", None) if comm is not None else None
    if ca_comm is None or ca_comm.disabled:
        return None
    return ca_comm


def glm5_mhc_ar_transition(
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
    ca_comm = _ca_comm()
    if ca_comm is not None:
        fused = ca_comm.fused_all_reduce_glm5_mhc(
            x.view(-1, D),
            residual,
            post_mix.view(-1, hc).contiguous(),
            comb_mix.view(-1, hc, hc).contiguous(),
            fn,
            hc_scale,
            hc_base,
            norm_weight,
            rms_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            norm_eps,
        )
        if fused is not None:
            res, post, comb, layer_input = fused
            return (
                res.view(T, hc, D),
                post.view(T, hc, 1),
                comb.view(T, hc, hc),
                layer_input.view(T, D),
            )
    # The previous site may have left its comb output on the side stream.
    join_glm5_mhc_deferred()
    x = tensor_model_parallel_all_reduce(x)
    return glm5_mhc_fused_post_pre(
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
        norm_weight,
        norm_eps,
    )


def _glm5_mhc_ar_transition_fake(
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


direct_register_custom_op(
    op_name="glm5_mhc_ar_transition",
    op_func=glm5_mhc_ar_transition,
    mutates_args=[],
    fake_impl=_glm5_mhc_ar_transition_fake,
)
