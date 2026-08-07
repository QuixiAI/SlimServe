# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from . import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger


def _moe_vec_row_limit(default: int, env: str, cuda_default: int = 64) -> int:
    override = os.environ.get(env)
    if override:
        return int(override)
    if not current_platform.is_rocm():
        return cuda_default
    return default


def _use_quixi_weighted_sum(
    out: torch.Tensor, topk_weights: torch.Tensor, out_hidden: torch.Tensor
) -> bool:
    """One-launch weighted reduce instead of mul_ + moe_sum (CUDA only)."""
    if current_platform.is_rocm():
        return False
    if not (
        out.dtype == torch.bfloat16
        and out_hidden.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and out.is_contiguous()
        and out_hidden.is_contiguous()
    ):
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available()


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(
            activation_enum,
            out,
            inp,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    mmq_ok = qweight_type in MMQ_QUANT_TYPES and qweight_type2 in MMQ_QUANT_TYPES
    vec_ok = qweight_type in MMVQ_QUANT_TYPES and qweight_type2 in MMVQ_QUANT_TYPES
    if current_platform.is_metal():
        # QuixiCore-Metal has no grouped GEMM over GGUF-quantized experts --
        # its moe_grouped_gemm is bf16/fp32 only, and base_qmoe_gemm wants the
        # BaseQN packing, not raw GGUF blocks. Fall through to the per-expert
        # loop below, which reaches the same qgemv/qgemm kernels one expert at
        # a time. Correct, and the throughput ceiling for MoE on Metal until a
        # grouped kernel exists.
        mmq_ok = vec_ok = False
    if mmq_ok or vec_ok:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        global_num_experts = expert_map.shape[0] if expert_map is not None else E
        local_topk_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
        top_k = topk_ids.shape[1]
        # ggml_moe_a8_vec has no weight reuse across tokens -- it reloads the
        # expert row for every (token, k) pair -- so its cost is linear in rows
        # while ggml_moe_a8 is flat.  w2 sees num_tokens*top_k rows, so it
        # crosses over 8x sooner than w1; gating both on one request count (the
        # old `x.shape[0] > 64`) leaves w2 on the GEMV kernel until it is 3.5x
        # slower than the GEMM. Crossovers measured on MI300X at GLM-5.2 TP2
        # shapes are between 32 and 64 input rows for w1 and between 128 and
        # 256 routed rows for w2 after grouping four adjacent output rows in
        # the Q2_K vector kernel and using a 4x64 Q2_K MMQ tile. Keep
        # separate overrides because other GGUF model shapes can cross over at
        # different points.
        # A100 w1 crossover at spec verify width (4 query tokens/request):
        # vec still wins at 4 tokens (40.2 vs 41.3 ms/step; real routing is
        # near-uncorrelated so MMQ streams as many experts as vec) but loses
        # from 32 tokens up (720 vs 604 us kernel-level, -10% ms/step e2e at
        # bs=8). Once the MMQ tile stopped computing its padding columns it
        # also won at 16 tokens (331 vs 360 us kernel-level, -10.3% ms/step at
        # bs=4), so the limit moved 16 -> 8: bs=1 verify stays on vec, bs>=2
        # verify goes to MMQ.
        w1_vec = vec_ok and (
            not mmq_ok
            or num_tokens <= _moe_vec_row_limit(32, "VLLM_GGUF_MOE_VEC_W1", 8)
        )
        # On A100 the w2 crossover sits below any batch this serves: forcing the
        # MMQ tile kernel measured neutral at 8 routed rows and +2.9% / +6.7% at
        # 32 / 64 (GLM-5.2 Q2_K, TP8, CUDA graphs), so vec never wins for w2.
        # w1 is left alone -- the same sweep showed MMQ costs it ~3% at batch 1.
        w2_rows = num_tokens * top_k
        w2_vec = vec_ok and (
            not mmq_ok or w2_rows <= _moe_vec_row_limit(128, "VLLM_GGUF_MOE_VEC_W2", 0)
        )

        sorted_token_ids = num_tokens_post_padded = None
        w1_expert_ids = w2_expert_ids = None
        if not (w1_vec and w2_vec):
            # w1 and w2 need not be the same quant -- DeepSeek-V4 ships IQ2_XXS
            # gate/up with Q2_K down -- and each tile kernel has its own mmq_x.
            # A kernel reads expert_ids[blockIdx.y] with blockIdx.y counting its
            # own tiles, so one shared array is only correct when the widths
            # match; otherwise every w2 tile picks the wrong expert and the
            # output is quietly wrong.
            #
            # So align the rows to a width both agree on (their LCM, which each
            # mmq_x divides) and hand each kernel an expert_ids expanded to its
            # own tile count. The alignment is a property of the layout, the
            # tile width a property of the kernel; conflating them is what tied
            # a type's mmq_x to whatever it was paired with.
            w1_block = ops.ggml_moe_get_block_size(qweight_type)
            w2_block = ops.ggml_moe_get_block_size(qweight_type2)
            align = math.lcm(w1_block, w2_block)
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids,
                align,
                global_num_experts,
                expert_map=expert_map,
            )
            # One entry per align columns -> one per mmq_x columns.
            w1_expert_ids = (
                expert_ids
                if align == w1_block
                else expert_ids.repeat_interleave(align // w1_block)
            )
            w2_expert_ids = (
                expert_ids
                if align == w2_block
                else expert_ids.repeat_interleave(align // w2_block)
            )

        # Both kernels emit rows in flat (token, k) order, so either can feed
        # the other.
        if w1_vec:
            out = ops.ggml_moe_a8_vec(
                x,
                w1,
                local_topk_ids,
                top_k,
                qweight_type,
                N,
                num_tokens,
                expert_parallel=expert_map is not None,
            )
        else:
            out = ops.ggml_moe_a8(
                x,
                w1,
                sorted_token_ids,
                w1_expert_ids,
                num_tokens_post_padded,
                qweight_type,
                N,
                top_k,
                num_tokens,
            )
        out = act(out)
        if w2_vec:
            out = ops.ggml_moe_a8_vec(
                out, w2, local_topk_ids, 1, qweight_type2, w2.shape[1], w2_rows
            )
        else:
            out = ops.ggml_moe_a8(
                out,
                w2,
                sorted_token_ids,
                w2_expert_ids,
                num_tokens_post_padded,
                qweight_type2,
                w2.shape[1],
                1,
                w2_rows,
            )
        out = out.reshape(num_tokens, top_k, w2.shape[1])
        if _use_quixi_weighted_sum(out, topk_weights, out_hidden_states):
            from vllm.quixicore import quixicore_ops

            quixicore_ops.moe_weighted_sum(
                out, topk_weights.contiguous(), out_hidden_states
            )
        else:
            out = out.mul_(topk_weights.view(num_tokens, top_k, 1))
            ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        local_topk_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
        if current_platform.is_metal():
            # Iterating MPS scalar tensors performs a device/host sync for each
            # `ii < 0` test and again when the tensor is used as a Python
            # index.  DSV4 has six routes in 43 layers, making those tiny
            # synchronizations dominate the actual matvecs.  Transfer the
            # 6*token integer routing table once per layer and keep route
            # weights on-device.
            local_topk_ids_host = local_topk_ids.to("cpu").tolist()
        else:
            local_topk_ids_host = local_topk_ids
        for tok, idx in enumerate(local_topk_ids_host):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for slot, ii in enumerate(idx):
                if ii < 0:
                    continue
                ww = topk_weights[tok, slot]
                out = fused_mul_mat_gguf_op(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], qweight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            if current_hidden_state is None:
                out_hidden_states[tok].zero_()
            else:
                out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    del (
        w1,
        w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        activation_situ_beta,
        activation_situ_linear_beta,
        expert_map,
    )
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
            self.moe.activation_situ_beta,
            self.moe.activation_situ_linear_beta,
            layer.global_to_local_expert_map,
        )
