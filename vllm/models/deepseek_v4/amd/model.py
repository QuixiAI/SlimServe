# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import os
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_pp_group,
    get_tp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    GateLinear,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mhc import (
    HAS_AITER_MHC,
    HAS_QUIXICORE_MHC,
    HAS_TILELANG_MHC,
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
    use_quixicore_mhc,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    SupportsEagle3,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform

if current_platform.is_metal():
    from vllm.models.deepseek_v4.metal import (
        DeepseekV4MetalAttention as DeepseekV4PlatformAttention,
    )
elif current_platform.is_cuda():
    from vllm.models.deepseek_v4.ampere import (
        DeepseekV4AmpereMLAAttention as DeepseekV4PlatformAttention,
    )
else:
    from vllm.models.deepseek_v4.amd.rocm import (
        DeepseekV4ROCMAiterMLAAttention as DeepseekV4PlatformAttention,
    )
from vllm.sequence import IntermediateTensors


def _use_quixicore_fused_mhc_norm(tensor: torch.Tensor) -> bool:
    return use_quixicore_mhc(tensor) and os.getenv(
        "VLLM_DSV4_MHC_FUSED_NORM", "0"
    ).lower() not in {"0", "false", "off", "no"}


def _mhc_fn_dtype(vllm_config: VllmConfig) -> torch.dtype:
    """Keep the GGUF F16 hyper-connection matrices packed on Ampere."""
    capability = (
        current_platform.get_device_capability()
        if current_platform.is_cuda()
        else None
    )
    quant_config = vllm_config.quant_config
    if (
        capability is not None
        and capability.major == 8
        and vllm_config.model_config.hf_config.hidden_size == 4096
        and quant_config is not None
        and quant_config.get_name() == "gguf"
    ):
        return torch.float16
    return torch.float32


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        self._ampere_shared_q8_fusion = (
            current_platform.is_cuda()
            and prefix.endswith(".shared_experts")
            and os.getenv("VLLM_DSV4_SHARED_Q8_FUSION", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
        prequant_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qweight_type = getattr(self.gate_up_proj, "qweight_type", None)
        qweight = getattr(self.gate_up_proj, "qweight", None)
        q8_types = (
            list(qweight_type.shard_weight_type.values())
            if qweight_type is not None and qweight_type.shard_weight_type
            else [getattr(qweight_type, "weight_type", -1)]
        )
        use_ampere_fusion = (
            self._ampere_shared_q8_fusion
            and x.ndim == 2
            and x.shape[0] == 1
            and qweight is not None
            and q8_types
            and all(weight_type == 8 for weight_type in q8_types)
            and isinstance(self.act_fn, SiluAndMulWithClamp)
            and self.act_fn.alpha == 1.0
            and self.act_fn.beta == 0.0
        )
        output_owned_down = getattr(self.down_proj, "_dsv4_output_owned", False)
        if output_owned_down:
            if use_ampere_fusion:
                from vllm.model_executor.layers.quantization.gguf import ops

                local_x = ops.ggml_dsv4_shared_gate_up_swiglu(
                    qweight, x, self.act_fn.swiglu_limit, prequant_input
                )
            else:
                gate_up, _ = self.gate_up_proj(x)
                local_x = self.act_fn(gate_up)
            full_x = tensor_model_parallel_all_gather(local_x, dim=-1)
            local_output, _ = self.down_proj(full_x)
            partial_output = local_output.new_zeros(
                (local_output.shape[0], self.down_proj.output_size)
            )
            row_start = self.down_proj.tp_rank * local_output.shape[1]
            partial_output[
                :, row_start : row_start + local_output.shape[1]
            ].copy_(local_output)
            return partial_output

        if use_ampere_fusion:
            from vllm.model_executor.layers.quantization.gguf import ops

            down_qweight_type = getattr(self.down_proj, "qweight_type", None)
            down_type = getattr(down_qweight_type, "weight_type", -1)
            down_quant_method = getattr(self.down_proj, "quant_method", None)
            apply_prequant = getattr(down_quant_method, "apply_prequant", None)
            fuse_down_quant = (
                down_type == 8
                and apply_prequant is not None
                and self.down_proj.input_size_per_partition <= 512
                and os.getenv("VLLM_DSV4_SHARED_DOWN_PREQUANT", "1").lower()
                not in {"0", "false", "off", "no"}
            )
            if fuse_down_quant:
                x, down_quant = ops.ggml_dsv4_shared_gate_up_swiglu_q8_1(
                    qweight, x, self.act_fn.swiglu_limit, prequant_input
                )
                return apply_prequant(self.down_proj, x, down_quant)
            x = ops.ggml_dsv4_shared_gate_up_swiglu(
                qweight, x, self.act_fn.swiglu_limit, prequant_input
            )
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


def _shared_experts_are_fp4(config, layer_idx: int | None = None) -> bool:
    """Whether the shared experts are MXFP4 and thus fusable.

    ``layer_idx=None`` resolves the model-wide default (global scheme), used by
    the main-model weight loader / mapper callers that operate per-model.
    """
    quant_cfg = getattr(config, "quantization_config", None)
    if quant_cfg is None:
        return False
    if layer_idx is None:
        base = None
    elif layer_idx >= config.num_hidden_layers:
        base = f"mtp.{layer_idx - config.num_hidden_layers}.ffn.shared_experts"
    else:
        base = f"layers.{layer_idx}.ffn.shared_experts"
    if base and any(e.startswith(base) for e in (quant_cfg.get("exclude") or [])):
        return False
    entry = (
        (quant_cfg.get("layer_quant_config") or {}).get(f"{base}.w1") if base else None
    )
    if entry is None:
        entry = quant_cfg.get("global_quant_config")
    return ((entry or {}).get("weight") or {}).get("dtype") == "fp4"


def _fuse_shared_experts_enabled(config, prefix: str = "") -> bool:
    """Whether to fuse the shared expert into the routed MXFP4 grouped GEMM.

    Fusion fuses the shared expert into the routed experts' MXFP4 grouped GEMM,
    so it only applies where the shared expert is the same precision as the
    routed experts. Some layers may carry a shared expert in a different quantization
    than the routed experts; when so, it runs as its own linear and must not be fused.
    """
    if not (
        current_platform.is_rocm()
        and getattr(config, "n_shared_experts", None)
        and envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS
        and not get_current_vllm_config().parallel_config.enable_expert_parallel
    ):
        return False
    return _shared_experts_are_fp4(
        config, extract_layer_index(prefix) if prefix else None
    )


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.defer_tp_reduce = (
            current_platform.is_cuda()
            and self.tp_size > 1
            and os.getenv("VLLM_DSV4_DEFER_TP_REDUCE", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.prefix = prefix

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hidden_size = config.hidden_size

        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size
        self.swiglu_limit = config.swiglu_limit
        self.renormalize = config.norm_topk_prob
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")

        self.gate = GateLinear(
            input_size=config.hidden_size,
            output_size=config.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None
        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        self.hash_indices_dtype = torch.int32
        if is_hash_moe:
            # hash MoE doesn't use e_score_correction_bias
            # Use randint instead of empty to avoid garbage values causing
            # invalid memory access in dummy mode (--load-format="dummy")
            self.gate.tid2eid = nn.Parameter(
                torch.randint(
                    0,
                    config.n_routed_experts,
                    (config.vocab_size, config.num_experts_per_tok),
                    dtype=self.hash_indices_dtype,
                ),
                requires_grad=False,
            )
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        self.n_shared_experts = config.n_shared_experts

        self.fuse_shared_experts = _fuse_shared_experts_enabled(config, prefix)

        if config.n_shared_experts is None or self.fuse_shared_experts:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.tp_rank = get_tensor_model_parallel_rank()
        assert config.n_routed_experts % self.tp_size == 0

        self.n_local_experts = config.n_routed_experts // self.tp_size
        self.experts_start_idx = self.tp_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            n_shared_experts=(
                config.n_shared_experts if self.fuse_shared_experts else None
            ),
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            swiglu_limit=self.swiglu_limit,
            router_logits_dtype=torch.float32,
            reduce_results=not self.defer_tp_reduce,
        )
        self.experts.defer_shared_expert_add = self.defer_tp_reduce

    def _output_owned_decode(
        self,
        kernel_input: torch.Tensor,
        full_quant: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        ca_comm,
    ) -> torch.Tensor | None:
        routed = self.experts.routed_experts
        shared = self.shared_experts
        if (
            shared is None
            or not getattr(routed, "_dsv4_w1_repacked", False)
            or not getattr(routed, "_dsv4_w2_repacked", False)
            or not getattr(routed, "_dsv4_w2_output_sharded", False)
            or not getattr(shared.down_proj, "_dsv4_output_owned", False)
        ):
            return None
        shared_down = getattr(shared.down_proj, "_dsv4_q8_aligned", None)
        if shared_down is None or self.swiglu_limit is None:
            return None

        topk_weights, topk_ids = self.experts.router.select_experts(
            hidden_states=kernel_input,
            router_logits=router_logits,
            topk_indices_dtype=torch.int32,
            input_ids=input_ids,
        )
        expert_map = routed.global_to_local_expert_map
        if expert_map is not None:
            topk_ids = expert_map[topk_ids]

        from vllm.model_executor.layers.quantization.gguf import ops

        _, shared_quant = ops.ggml_dsv4_shared_gate_up_swiglu_q8_1(
            shared.gate_up_proj.qweight,
            kernel_input,
            self.swiglu_limit,
            full_quant,
        )
        return ca_comm.dsv4_output_owned_moe(
            full_quant,
            routed.w13_qweight,
            routed.w2_qweight,
            topk_weights.contiguous(),
            topk_ids.contiguous(),
            shared_quant,
            shared_down,
            self.swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        prequant_input: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")

        org_shape = hidden_states.shape
        local_hidden = 4096 // self.tp_size
        owned_decode = (
            os.getenv("VLLM_DSV4_TP_OWNERSHIP", "0").lower()
            in {"1", "true", "on", "yes"}
            and hidden_states.shape == (1, local_hidden)
            and prequant_input is not None
            and prequant_input.numel() == local_hidden // 32 * 9
        )
        if owned_decode:
            communicator = get_tp_group().device_communicator
            ca_comm = (
                getattr(communicator, "ca_comm", None)
                if communicator is not None
                else None
            )
            if ca_comm is None:
                raise RuntimeError("DSV4 TP ownership requires custom all-reduce")
            full_quant = ca_comm.dsv4_gather_owned_q8(prequant_input)
            if self.gate.weight.dtype != torch.bfloat16:
                raise RuntimeError(
                    "DSV4 input-owned router requires BF16 gate weights"
                )
            router_logits = ca_comm.dsv4_owned_router(
                hidden_states, self.gate.weight
            )
            # Native GGUF kernels consume full Q8_1 directly. The BF16 tensor
            # carries shape/dtype metadata only and is never dequantized.
            kernel_input = torch.empty(
                (hidden_states.shape[0], 4096),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            output_owned = self._output_owned_decode(
                kernel_input,
                full_quant,
                router_logits,
                input_ids,
                ca_comm,
            )
            if output_owned is not None:
                return output_owned
            org_shape = kernel_input.shape
            if self.experts.is_internal_router:
                final_hidden_states = self.experts(
                    hidden_states=kernel_input,
                    router_logits=router_logits,
                    input_ids=input_ids,
                    prequant_input=full_quant,
                )
            else:
                final_hidden_states = self.experts(
                    hidden_states=kernel_input,
                    router_logits=router_logits,
                    input_ids=input_ids,
                    prequant_input=full_quant,
                )
            if isinstance(final_hidden_states, tuple):
                return tuple(output.view(org_shape) for output in final_hidden_states)
            return final_hidden_states.view(org_shape)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=hidden_states,
                input_ids=input_ids,
                prequant_input=prequant_input,
            )
        else:
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                input_ids=input_ids,
                prequant_input=prequant_input,
            )

        def restore_prefill_partial(output: torch.Tensor) -> torch.Tensor:
            if output.numel() == math.prod(org_shape):
                return output.view(org_shape)
            if (
                org_shape[-1] == self.hidden_size
                and output.shape[-1] == self.hidden_size // self.tp_size
                and output.numel()
                == math.prod(org_shape[:-1]) * output.shape[-1]
            ):
                partial = output.new_zeros(org_shape)
                row_start = self.tp_rank * output.shape[-1]
                partial[..., row_start : row_start + output.shape[-1]].copy_(
                    output.view(*org_shape[:-1], output.shape[-1])
                )
                return partial
            raise RuntimeError(
                f"DSV4 MoE output shape {tuple(output.shape)} does not match "
                f"the {org_shape} layer contract"
            )

        if isinstance(final_hidden_states, tuple):
            return tuple(
                restore_prefill_partial(output) for output in final_hidden_states
            )
        return restore_prefill_partial(final_hidden_states)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        attention_factory: Callable[[VllmConfig, str], nn.Module] | None = None,
    ):
        super().__init__()

        # Lazy import to avoid top-level tilelang dependency.
        # Registers both torch.ops.vllm.mhc_pre and mhc_post
        import vllm.model_executor.layers.mhc  # noqa: F401

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.rms_norm_eps = config.rms_norm_eps
        attention_prefix = f"{prefix}.attn"
        if attention_factory is None:
            self.attn = DeepseekV4PlatformAttention(
                vllm_config,
                prefix=attention_prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
        else:
            self.attn = attention_factory(vllm_config, attention_prefix)
        self.ffn = DeepseekV4MoE(vllm_config, prefix=f"{prefix}.ffn")

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        capability = (
            current_platform.get_device_capability()
            if current_platform.is_cuda()
            else None
        )
        self._ampere_prequant_fusion = bool(
            current_platform.is_cuda()
            and capability is not None
            and capability.major == 8
            and self.hidden_size == 4096
            and quant_config is not None
            and quant_config.get_name() == "gguf"
            and os.getenv("VLLM_DSV4_PREQUANT_FUSION", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        self._ampere_prequant_attention = bool(
            self._ampere_prequant_fusion
            and os.getenv("VLLM_DSV4_PREQUANT_ATTN", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        self._ampere_prequant_moe = bool(
            self._ampere_prequant_fusion
            and os.getenv("VLLM_DSV4_PREQUANT_MOE", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        self._ampere_prequant_standalone = bool(
            self._ampere_prequant_fusion
            and os.getenv("VLLM_DSV4_PREQUANT_STANDALONE", "0").lower()
            not in {"0", "false", "off", "no"}
        )
        self._ampere_prequant_transition = bool(
            self._ampere_prequant_fusion
            and os.getenv("VLLM_DSV4_PREQUANT_TRANSITION", "1").lower()
            not in {"0", "false", "off", "no"}
        )
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        hc_fn_dtype = _mhc_fn_dtype(vllm_config)
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=hc_fn_dtype,
            ),
            requires_grad=False,
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=hc_fn_dtype,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()
        self.use_fused_mhc = (HAS_TILELANG_MHC or HAS_QUIXICORE_MHC) and not (
            HAS_AITER_MHC and self.hidden_size % 256 == 0
        )
        self.defer_tp_reduce = bool(
            self.use_fused_mhc
            and getattr(self.attn, "defer_tp_reduce", False)
            and self.ffn.defer_tp_reduce
        )
        if not self.defer_tp_reduce:
            self.attn.wo_b.reduce_results = True
            self.ffn.experts.moe_config.skip_final_all_reduce = False
            self.ffn.experts.defer_shared_expert_add = False
            self.ffn.defer_tp_reduce = False
        layer_match = re.search(r"\.layers\.(\d+)$", prefix)
        layer_index = int(layer_match.group(1)) if layer_match is not None else -1
        ownership_requested = os.getenv(
            "VLLM_DSV4_TP_OWNERSHIP", "0"
        ).lower() in {"1", "true", "on", "yes"}
        if ownership_requested and os.getenv("VLLM_DSV4_MHC_SCHEDULE") != "async":
            raise RuntimeError(
                "VLLM_DSV4_TP_OWNERSHIP requires "
                "VLLM_DSV4_MHC_SCHEDULE=async"
            )
        self._tp_owned_mhc = bool(
            ownership_requested
            and current_platform.is_cuda()
            and self.defer_tp_reduce
            and layer_index >= 0
        )
        deferred_ownership_requested = os.getenv(
            "VLLM_DSV4_TP_DEFERRED_OWNERSHIP", "0"
        ).lower() in {"1", "true", "on", "yes"}
        self._tp_deferred_owned_mhc = bool(
            self._tp_owned_mhc and deferred_ownership_requested
        )
        self._layer_index = layer_index
        self._is_last_layer = layer_index + 1 == config.num_hidden_layers
        q2_progress_enabled = os.getenv(
            "VLLM_DSV4_Q2_MHC_PROGRESS", "0"
        ).lower() in {"1", "true", "on", "yes"}
        self._ampere_q2_mhc = bool(
            q2_progress_enabled
            and self._ampere_prequant_transition
            and self.defer_tp_reduce
            and layer_index >= 0
        )
        # The final decoder output is consumed by hc_post rather than another
        # fused transition, so it must remain materialized. Every earlier
        # decode layer hands its native Q8_1 intermediate to the next layer's
        # custom-allreduce-owned Q2_K producer.
        self.ffn.experts.routed_experts._dsv4_defer_down = bool(
            self._ampere_q2_mhc and layer_index + 1 < config.num_hidden_layers
        )

    def reduce_deferred_output(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if self.defer_tp_reduce:
            device_communicator = get_tp_group().device_communicator
            ca_comm = (
                getattr(device_communicator, "ca_comm", None)
                if device_communicator is not None
                else None
            )
            if ca_comm is not None:
                ca_comm.wait_dsv4_mhc(x[0] if isinstance(x, tuple) else x)
        if isinstance(x, tuple):
            x = x[0] + x[1]
        if self.defer_tp_reduce:
            return tensor_model_parallel_all_reduce(x)
        return x

    def reconstruct_deferred_output(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
    ) -> torch.Tensor:
        local_hidden = 4096 // get_tensor_model_parallel_world_size()
        primary = x[0] if isinstance(x, tuple) else x
        if (
            self._tp_owned_mhc
            and residual.shape[-1] == 4096
            and primary.shape[-1] == local_hidden
        ):
            if isinstance(x, tuple):
                raise RuntimeError(
                    "DSV4 output ownership requires fused routed/shared output"
                )
            communicator = get_tp_group().device_communicator
            ca_comm = (
                getattr(communicator, "ca_comm", None)
                if communicator is not None
                else None
            )
            if ca_comm is None:
                raise RuntimeError("DSV4 TP ownership requires custom all-reduce")
            ca_comm.wait_dsv4_mhc(primary)
            return self.hc_post(
                ca_comm.dsv4_gather_owned_bf16(primary),
                residual,
                post_mix,
                res_mix,
            )
        if self._tp_owned_mhc and residual.shape[-1] == local_hidden:
            communicator = get_tp_group().device_communicator
            ca_comm = (
                getattr(communicator, "ca_comm", None)
                if communicator is not None
                else None
            )
            if ca_comm is None:
                raise RuntimeError("DSV4 TP ownership requires custom all-reduce")
            primary, addend = (
                (x[0], x[1]) if isinstance(x, tuple) else (x, None)
            )
            local_x = ca_comm.dsv4_owned_reduce_scatter(primary, addend)
            ca_comm.wait_dsv4_mhc(local_x)
            local_reconstructed = self.hc_post(
                local_x, residual, post_mix, res_mix
            )
            return ca_comm.dsv4_gather_owned_bf16(local_reconstructed)
        reduced = self.reduce_deferred_output(x)
        return self.hc_post(reduced, residual, post_mix, res_mix)

    def _fused_post_pre_with_optional_reduce(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: torch.Tensor | None,
        reduce_x: bool,
        input_prepared: bool = False,
        own_projections: bool = False,
        publish_prepared: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if reduce_x:
            device_communicator = get_tp_group().device_communicator
            ca_comm = (
                getattr(device_communicator, "ca_comm", None)
                if device_communicator is not None
                else None
            )
            if ca_comm is not None:
                owned_decode = (
                    self._tp_owned_mhc
                    and norm_weight is not None
                    and residual.shape == (1, 4, 4096)
                    and (
                        (
                            isinstance(x, tuple)
                            and x[0].shape
                            in ((1, 4096), (1, 4096 // self.ffn.tp_size))
                        )
                        or (
                            isinstance(x, torch.Tensor)
                            and x.shape
                            in ((1, 4096), (1, 4096 // self.ffn.tp_size))
                        )
                    )
                )
                if (
                    self._ampere_q2_mhc
                    and isinstance(x, tuple)
                    and len(x) == 2
                    and x[1].shape == (1, 4096)
                    and norm_weight is not None
                    and not (
                        input_prepared or own_projections or publish_prepared
                    )
                ):
                    fused = ca_comm.fused_all_reduce_dsv4_q2_mhc(
                        x[1],
                        x[0],
                        residual,
                        post_mix.contiguous(),
                        res_mix.contiguous(),
                        fn,
                        scale,
                        base,
                        self.rms_norm_eps,
                        self.hc_eps,
                        self.hc_eps,
                        self.hc_post_alpha,
                        self.hc_sinkhorn_iters,
                        norm_weight,
                        self.rms_norm_eps,
                    )
                    if fused is not None:
                        return fused
                fused_fn = (
                    ca_comm.fused_all_reduce_dsv4_mhc_add
                    if isinstance(x, tuple)
                    else ca_comm.fused_all_reduce_dsv4_mhc
                )
                fused_args = x if isinstance(x, tuple) else (x,)
                fused = fused_fn(
                    *fused_args,
                    residual,
                    post_mix.contiguous(),
                    res_mix.contiguous(),
                    fn,
                    scale,
                    base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    norm_weight,
                    self.rms_norm_eps,
                    input_prepared=input_prepared,
                    own_projections=own_projections,
                    publish_prepared=publish_prepared,
                    local_input_owned=owned_decode,
                )
                if fused is not None:
                    return fused
                if owned_decode:
                    raise RuntimeError(
                        "DSV4 TP-owned decode transition did not match the "
                        "native mHC contract"
                    )
                # Memory profiling, prefill, and batched eager probes still
                # carry the static ownership flags, but remain full-width.
        if isinstance(x, tuple):
            x = x[0] + x[1]
        if reduce_x:
            x = tensor_model_parallel_all_reduce(x)
        return self.mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight=None,
            norm_eps=0.0,
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor | None = None,
    ):
        post_mix, res_mix, layer_input = self.mhc_pre(
            residual=x,
            fn=hc_fn,
            hc_scale=hc_scale,
            hc_base=hc_base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=self.rms_norm_eps,
        )
        return layer_input, post_mix, res_mix

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):
        return self.mhc_post(x, residual, post, comb)

    def _norm_with_prequant(
        self,
        x: torch.Tensor,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            self._ampere_prequant_standalone
            and x.ndim == 2
            and 0 < x.shape[0] <= 8
            and x.shape[1] == 4096
            and x.dtype == torch.bfloat16
        ):
            from vllm.model_executor.layers.quantization.gguf import ops

            return ops.ggml_dsv4_rms_norm_q8_1(
                x, norm.weight, self.rms_norm_eps
            )
        return norm(x), None

    def _forward_fused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            # Run standalone hc_pre on first layer
            residual = x
            fused_norm = False
            x, post_mix, res_mix = self.hc_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
            prequant_input = None
        else:
            transition = self._fused_post_pre_with_optional_reduce(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.attn_norm.weight
                if self._ampere_prequant_transition
                else None,
                self.defer_tp_reduce,
                input_prepared=self._tp_deferred_owned_mhc,
                own_projections=self._tp_owned_mhc,
                publish_prepared=self._tp_deferred_owned_mhc,
            )
            fused_norm = len(transition) == 5
            residual, post_mix, res_mix, x = transition[:4]
            prequant_input = transition[4] if fused_norm else None

        if not fused_norm:
            x, prequant_input = self._norm_with_prequant(x, self.attn_norm)
        x = self.attn(
            positions,
            x,
            None,
            prequant_input=(
                prequant_input if self._ampere_prequant_attention else None
            ),
        )

        transition = self._fused_post_pre_with_optional_reduce(
            x,
            residual,
            post_mix,
            res_mix,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight if self._ampere_prequant_transition else None,
            self.defer_tp_reduce,
            input_prepared=(
                self._tp_deferred_owned_mhc and self._layer_index > 0
            ),
            own_projections=self._tp_owned_mhc,
            publish_prepared=(
                self._tp_deferred_owned_mhc and not self._is_last_layer
            ),
        )
        fused_norm = len(transition) == 5
        residual, post_mix, res_mix, x = transition[:4]
        prequant_input = transition[4] if fused_norm else None
        if not fused_norm:
            x, prequant_input = self._norm_with_prequant(x, self.ffn_norm)
        x = self.ffn(
            x,
            input_ids,
            prequant_input=(prequant_input if self._ampere_prequant_moe else None),
        )
        return x, residual, post_mix, res_mix

    def _forward_unfused_post_pre(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        x, prequant_input = self._norm_with_prequant(x, self.attn_norm)
        x = self.attn(
            positions,
            x,
            None,
            prequant_input=(
                prequant_input if self._ampere_prequant_attention else None
            ),
        )
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(
            x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        x, prequant_input = self._norm_with_prequant(x, self.ffn_norm)
        x = self.ffn(
            x,
            input_ids,
            prequant_input=(prequant_input if self._ampere_prequant_moe else None),
        )
        x = self.hc_post(x, residual, post, comb)
        return x, None, None, None

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if not self.use_fused_mhc:
            return self._forward_unfused_post_pre(
                x, positions, input_ids, post_mix, res_mix, residual
            )
        return self._forward_fused_post_pre(
            x, positions, input_ids, post_mix, res_mix, residual
        )


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention.attn_gemm_parallel_execute
        # (compressor kv_score, indexer.weights_proj, indexer.compressor
        # kv_score). fused_wqa_wkv stays on the default stream.
        # Disable them on ROCm because of hang issues.
        aux_streams_enabled = os.getenv("VLLM_DSV4_AUX_STREAMS", "1").lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        aux_stream_list = (
            None
            if (
                current_platform.is_rocm()
                or current_platform.is_metal()
                or not aux_streams_enabled
            )
            else [torch.cuda.Stream() for _ in range(3)]
        )

        self.device = current_platform.device_type
        # Reserved topk indices buffer for all Indexer layers to reuse.
        # Seeded with the -1 "no token" sentinel rather than left uninitialized:
        # the per-step reset only covers [:hidden_states.shape[0]], while the
        # decode path slices [:num_padded_tokens], so padded rows would
        # otherwise carry whatever was in the allocation and index real KV
        # slots.
        self.topk_indices_buffer = torch.full(
            (
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
            ),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.hc_head_fn = nn.Parameter(
            torch.empty(
                self.hc_mult,
                self.hc_dim,
                dtype=_mhc_fn_dtype(vllm_config),
            ),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(
                self.hc_mult,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_op = HCHeadOp()
        # Pre-hc_head residual stream buffer for the MTP draft. Stable
        # address (outside the cudagraph pool) so the copy_ in forward()
        # refreshes it correctly across captured shapes.
        # refreshes it correctly across captured shapes. Only allocated on
        # the last PP rank — that's where MTP target hidden states are
        # produced.
        if get_pp_group().is_last_rank:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until hc_head() collapses it.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        residual, post_mix, res_mix = None, None, None
        # EAGLE3 / DSpark / DFlash aux hidden states: reconstructed (post-mhc)
        # hidden state at the configured target layers, averaged over the
        # hc_mult streams to [T, hidden_size]. Empty unless a draft model set
        # aux_hidden_state_layers.
        aux_hidden_states: list[torch.Tensor] = []
        # On the fused path the final layer's hc_post output is reused below
        # (avoids computing hc_post twice when the last layer is also an aux
        # layer).
        final_aux_recon: torch.Tensor | None = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            if (idx + 1) in self.aux_hidden_state_layers:
                # On the unfused (aiter) path the layer already applied hc_post,
                # so hidden_states is the reconstructed stream; on the fused
                # path reconstruct it via hc_post before averaging.
                if layer.use_fused_mhc:
                    aux_recon = layer.reconstruct_deferred_output(
                        hidden_states, residual, post_mix, res_mix
                    )
                    final_aux_recon = aux_recon
                else:
                    aux_recon = hidden_states
                aux_hidden_states.append(aux_recon.mean(dim=1))
        if layer is not None and layer.use_fused_mhc:
            # Reuse the last layer's hc_post output if it was already computed
            # for the aux hidden state above; otherwise compute it now.
            if (
                final_aux_recon is not None
                and self.end_layer in self.aux_hidden_state_layers
            ):
                hidden_states = final_aux_recon
            else:
                hidden_states = layer.reconstruct_deferred_output(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        # Stash pre-hc_head residual for the MTP draft (captured copy_).
        num_tokens = hidden_states.shape[0]
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = self.hc_head_op(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Use each MoE's own per-layer fusion decision (computed with its prefix
        # at init) as the single source of truth, so the redirect below cannot
        # diverge from how the module was built if per-layer quantization ever
        # mixes fused and non-fused layers.
        fuse_by_layer = {
            extract_layer_index(mod_name): mod.fuse_shared_experts
            for mod_name, mod in self.named_modules()
            if isinstance(mod, DeepseekV4MoE)
        }
        n_routed = self.config.n_routed_experts
        # The redirect below maps the single shared-expert tensor group to one
        # appended slot; multiple shared experts would need per-expert slicing
        # (see deepseek_v2.py). DeepSeek-V4 has n_shared_experts == 1.
        if any(fuse_by_layer.values()) and self.config.n_shared_experts != 1:
            raise NotImplementedError(
                "deepseek-v4 fused shared-expert loading supports only "
                f"n_shared_experts == 1, got {self.config.n_shared_experts}"
            )

        for name, loaded_weight in weights:
            # Shared-expert fusion: redirect ``.ffn.shared_experts.w{1,2,3}``
            # into appended routed-expert slot ``.ffn.experts.{n_routed}``
            # so the MXFP4-quantized shared expert loads through the routed
            # expert loader (grouped GEMM). Single shared expert only.
            if ".ffn.shared_experts.w" in name and fuse_by_layer.get(
                extract_layer_index(name), False
            ):
                name = name.replace(
                    ".ffn.shared_experts.w",
                    f".ffn.experts.{n_routed}.w",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        # When fusing shared experts, include the appended slots
        # (ids n_routed_experts .. n_routed_experts + n_shared - 1) so the
        # redirected shared-expert weights route through the expert loader.
        n_shared = getattr(self.config, "n_shared_experts", 0) or 0
        num_experts = self.config.n_routed_experts + (
            n_shared if _fuse_shared_experts_enabled(self.config) else 0
        )
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=num_experts,
        )


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, fuse_shared_experts: bool = False
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). FP8 linear and
        # (non-fused) shared experts use Fp8LinearMethod's block scales,
        # which register as ``weight_scale_inv``.
        #
        #  - DeepSeek native ``.scale``: expert scales -> ``.weight_scale``,
        #    everything else -> ``.weight_scale_inv``.
        #  - AMD-Quark ``.weight_scale``: linear/attn scales ->
        #    ``.weight_scale_inv``. Expert and shared-expert
        #    ``w{1,2,3}.weight_scale`` are left untouched (consumed as-is by
        #    the MXFP4 expert loader, which produces ``w{13,2}_weight_scale``);
        scale_regex = {
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
            re.compile(r"(?<!\.w[123])\.weight_scale$"): ".weight_scale_inv",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    # When shared experts are fused into the routed MXFP4 grouped GEMM, the
    # shared_experts tensors are redirected to routed expert slots ; leave
    # their names untouched here.
    substr_map = (
        {}
        if fuse_shared_experts
        else {".shared_experts.w2": ".shared_experts.down_proj"}
    )
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr=substr_map,
    )


class DeepseekV4ForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        fuse_shared_experts = _fuse_shared_experts_enabled(config)
        if expert_dtype != "fp4" or fuse_shared_experts:
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
                expert_dtype, fuse_shared_experts=fuse_shared_experts
            )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-hc_head residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        return loaded_params

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
