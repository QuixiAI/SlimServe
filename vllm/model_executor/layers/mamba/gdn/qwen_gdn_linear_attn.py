# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3-Next/Qwen3.5 model."""

import os
from typing import Literal

import torch
from einops import rearrange
from torch import nn

from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    divide,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp, PluggableLayer
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.inc import INCConfig
from vllm.model_executor.model_loader.weight_utils import (
    sharded_weight_loader,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
)
from vllm.third_party.flash_linear_attention.ops import (
    fused_post_conv_prep,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.third_party.flash_linear_attention.ops.chunk import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Optional ROCm AITER Triton kernels for the GDN decode path.
# Availability is checked centrally via rocm_aiter_ops; the actual function
# references are imported here so that they can be called without per-call
# import overhead.
GDN_AITER_TRITON_AVAILABLE = rocm_aiter_ops.are_gdn_triton_kernels_available()


if GDN_AITER_TRITON_AVAILABLE:
    from aiter.ops.triton.causal_conv1d_update_single_token import (
        fused_reshape_causal_conv1d_update_single_token as gdn_aiter_fused_reshape_causal_conv1d_update_single_token,  # noqa: E501
    )
    from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
        fused_rearrange_sigmoid_gated_delta_rule as gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule,  # noqa: E501
    )

logger = init_logger(__name__)


# Metal GDN dispatch-fusion kill switches (see _forward_core_metal): the
# fused prepare collapses conv/qkv_prepare/gate_beta plus the host-side fp32
# cast and b/a copies into one dispatch on pure-decode batches; the fused
# norm reads the fp32 recurrence output and the strided z view directly,
# replacing the cast, gather, and container copies around the gated RMSNorm.
_QC_GDN_FUSEPREP = os.environ.get("VLLM_QC_GDN_FUSEPREP", "1") != "0"
_QC_GDN_FUSENORM = os.environ.get("VLLM_QC_GDN_FUSENORM", "1") != "0"


def _resolve_gdn_prefill_backend(
    vllm_config: VllmConfig,
) -> tuple[str, Literal["triton", "flashinfer", "cutedsl"]]:
    """Resolve GDN prefill backend.

    FlashInfer's GDN prefill kernel is chosen when:
    * ``requested in ["flashinfer", "auto"]``;
    * ``platform == cuda``;
    * one of the following:
      - Hopper (SM90) — no further constraints;
      - Blackwell (SM10.x) with ``head_k_dim == 128``, ``cuda_runtime >= 13``.

    In-tree CuteDSL GDN prefill kernel is chosen when:
    * "cutedsl" is requested; (opt-in only)
    * Blackwell (SM10.x) with ``head_k_dim == 128``;
    """
    additional_config = vllm_config.additional_config
    backend_cfg = (
        additional_config.get("gdn_prefill_backend", "auto")
        if isinstance(additional_config, dict)
        else "auto"
    )
    backend = str(backend_cfg).strip().lower()

    if not current_platform.is_cuda():
        return backend, "triton"

    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )

    supports_flashinfer = False
    supports_cutedsl = False

    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True

    if backend in ["flashinfer", "auto"] and supports_flashinfer:
        return backend, "flashinfer"
    if backend == "cutedsl" and supports_cutedsl:
        return backend, "cutedsl"
    return backend, "triton"


def _log_gdn_backend_decision(
    vllm_config: VllmConfig,
    requested_backend: str,
    active_backend: str,
) -> None:
    """Log the GDN prefill backend choice in the attention-selector style."""
    head_k_dim = getattr(
        vllm_config.model_config.hf_text_config, "linear_key_head_dim", None
    )
    chosen = {
        "flashinfer": "FlashInfer",
        "cutedsl": "CuteDSL",
        "triton": "Triton/FLA",
    }[active_backend]
    logger.info_once(
        "Using %s GDN prefill kernel (requested=%s, head_k_dim=%s).",
        chosen,
        requested_backend,
        head_k_dim,
    )
    if active_backend == "flashinfer" and current_platform.is_device_capability(90):
        logger.warning_once(
            "FlashInfer GDN prefill is JIT-compiled; first run may take a "
            "while. Set --gdn-prefill-backend triton to skip JIT.",
        )


def fi_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    from flashinfer.gdn_prefill import (
        chunk_gated_delta_rule as chunk_gated_delta_rule_fi,
    )

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # use flashinfer implementation
    q = q.squeeze(0).contiguous()
    k = k.squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()

    g = g.squeeze(0).contiguous()
    beta = beta.squeeze(0).contiguous()
    fi_state = initial_state.to(torch.float32)
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q,
        k=k,
        v=v,
        g=torch.exp(fi_g),
        beta=fi_beta,
        initial_state=fi_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # FlashInfer returns (output, state) when output_final_state=True,
    # or just output when output_final_state=False.
    # Unsqueeze back to 4D (1, L, H, D) to match fla output format
    if output_final_state:
        output, final_state = result
        return output.unsqueeze(0), final_state
    else:
        return result.unsqueeze(0), None


@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self) -> None:
        super().__init__()
        vllm_config = get_current_vllm_config()
        backend, active_backend = _resolve_gdn_prefill_backend(vllm_config)
        self.gdn_prefill_backend = active_backend

        if backend in ("flashinfer", "cutedsl") and active_backend != backend:
            logger.warning_once(
                "GDN prefill backend '%s' is selected but cannot use this "
                "kernel on the current platform. Falling back to Triton/FLA.",
                backend,
            )
        _log_gdn_backend_decision(vllm_config, backend, active_backend)

        if active_backend == "flashinfer":
            self._forward_method = self.forward_cuda
        elif active_backend == "cutedsl":
            self._forward_method = self.forward_cutedsl
        else:
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        o, final_state = fi_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)
        return o, final_state

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )

    def forward_cutedsl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            chunk_gated_delta_rule_cutedsl,
        )

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        assert cu_seqlens is not None
        assert chunk_indices is not None
        assert chunk_offsets is not None

        o, final_state = chunk_gated_delta_rule_cutedsl(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        if not output_final_state:
            final_state = None
        return o, final_state


@PluggableLayer.register("qwen_gated_delta_net_attention")
class QwenGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def __init__(
        self,
        config: Qwen3NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        gqa_interleaved_layout=False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.gqa_interleaved_layout = gqa_interleaved_layout
        # llama.cpp-converted GGUFs store every per-V-head GDN tensor in ggml
        # tiled-broadcast order (see build_qwen35_config_from_gguf); the q/k
        # heads must then be expanded with tile semantics (i_k = i_hv % H)
        # instead of HF's repeat_interleave (i_k = i_hv // (HV // H)).
        # Currently honored by the MPS core; the CUDA/ROCm FLA kernels assume
        # the HF grouped layout.
        self.tiled_v_head_layout = bool(
            getattr(config, "gdn_tiled_v_head_layout", False)
        )
        # Which core honors the tiled layout natively decides how we serve it,
        # and that is a property of the platform's kernels, not of the file.
        # The MPS core threads `tiled_gqa` all the way into its scan. The
        # CUDA/ROCm FLA kernels have no such switch and always expand q/k with
        # repeat_interleave, so on those platforms the v-head axis must be
        # normalized to the HF grouped order around the recurrence instead.
        self.gdn_core_honors_tiled = current_platform.is_metal()
        self.normalize_v_heads_to_grouped = (
            self.tiled_v_head_layout and not self.gdn_core_honors_tiled
        )
        self._v_head_perms: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
        if current_platform.is_xpu():
            self._forward_method = self.forward_xpu
        elif current_platform.is_cpu():
            from vllm.model_executor.layers.mamba.ops.cpu.gdn_attention import (
                register_cpu_gdn_attention_ops,
            )

            register_cpu_gdn_attention_ops()
            self._forward_method = self.forward_cpu
        elif current_platform.is_rocm():
            self._forward_method = self.forward_hip
        elif current_platform.is_metal():
            self._forward_method = self.forward_mps
            # Metal GDN kernel gate: resolved lazily on the first core call
            # (the state pools are not bound yet here). None = unresolved.
            self._metal_gdn: object | None = None
        else:
            self._forward_method = self.forward_cuda

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        # Qwen3-Next and Qwen3.5 has a different qkv_proj layout,
        # we need to create qkvz_proj adaptively here.
        # When create_in_proj_qkvz is False (e.g. LoRA enabled in Qwen3.5),
        # in_proj_qkv and in_proj_z are created separately instead.
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvz",
        )

        # ba_proj doesn't support blockwise fp8 quantization.
        # Qwen3-Next and Qwen3.5 have different in_proj_ba checkpoint
        # layouts, so we use a factory method to create the projection.
        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_ba",
        )
        self.disable_tp_for_ba_proj = self.maybe_disable_tp(self.quant_config)

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                query_key_settings,
                query_key_settings,
                value_settings,
            ],
            self.tp_size,
            self.tp_rank,
        )

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            )
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        assert output_gate_type in ["silu", "swish", "sigmoid"], (
            f"unsupported {output_gate_type=}"
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            activation=output_gate_type,
            device=current_platform.current_device(),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=reduce_results,
            quant_config=self.quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
        self.gdn_prefill_backend = self.chunk_gated_delta_rule.gdn_prefill_backend
        self._prefill_kernels_warmed_up = False
        self.enable_packed_recurrent_decode = (
            envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), qkvz weights are
        # stored as a single fused tensor with interleaved GQA layout, so we
        # use one output shard to preserve the interleaving across TP ranks.
        # When gqa_interleaved_layout=False (Qwen3.5), the checkpoint has
        # separate q, k, v, z weights, so we use 4 independent output sizes.
        output_sizes = (
            [sum((key_dim, key_dim, value_dim, value_dim))]
            if self.gqa_interleaved_layout
            else [key_dim, key_dim, value_dim, value_dim]
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # When gqa_interleaved_layout=True (Qwen3-Next), in_proj_ba is stored
        # as a single fused weight [b_g0, a_g0, b_g1, a_g1, ...] interleaved
        # by key-head group; a single output shard preserves this across TP.
        # When gqa_interleaved_layout=False (Qwen3.5), in_proj_b and in_proj_a
        # are separate checkpoint weights, so we use 2 independent output sizes.
        output_sizes = (
            [num_v_heads * 2] if self.gqa_interleaved_layout else [num_v_heads] * 2
        )
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=self.maybe_disable_tp(quant_config),
        )

    def maybe_disable_tp(self, quant_config: QuantizationConfig | None) -> bool:
        """Whether to replicate ba_proj instead of TP-sharding it.

        Marlin requires output_size_per_partition >= MIN_THREAD_N=64, which
        the Qwen3.5 non-interleaved [num_v_heads]*2 layout violates at TP>=2
        (e.g. num_v_heads=64, TP=4 -> 16). Replicating the projection keeps
        each rank above the Marlin threshold; forward() then slices b/a to
        the local TP partition. Qwen3-Next's interleaved [num_v_heads*2]
        layout is unaffected and stays TP-sharded.

        See https://github.com/vllm-project/vllm/issues/35924
        """
        return (
            current_platform.is_cuda()
            and not self.gqa_interleaved_layout
            and isinstance(quant_config, (AutoAWQConfig, AutoGPTQConfig, INCConfig))
        )

    def split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.chunk(2, dim=-1)
        if self.disable_tp_for_ba_proj and self.tp_size > 1:
            # ba_proj is replicated for Marlin; slice b/a to local TP rank.
            ba_chunk = self.num_v_heads // self.tp_size
            ba_start = self.tp_rank * ba_chunk
            b = b[:, ba_start : ba_start + ba_chunk]
            a = a[:, ba_start : ba_start + ba_chunk]
        return b, a

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    @torch.compile(fullgraph=True)
    def prepare_gdn_attention_core_inputs(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
        num_tokens: int,
    ):
        """
        Derives mixed_qkv, z, b, a from projected qkvz/ba for the GDN custom op.

        For gqa_interleaved_layout (Qwen3-Next): unpack the interleaved
        [ng, (hk + hk + np/ng*hv + np/ng*hv)] layout into contiguous qkv.
        For non-interleaved layout (Qwen3.5): simple split along last dim.
        """
        if not self.gqa_interleaved_layout:
            # Qwen3.5: weights are in [q, k, v, z] order
            assert num_tokens == mixed_qkvz.shape[0]
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z_flat = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            n = mixed_qkvz.shape[0]
            z_out = z_flat.reshape(n, -1, self.head_v_dim)
            b, a = mixed_ba.chunk(2, dim=-1)
            return mixed_qkv, z_out, b, a

        # Qwen3-Next: interleaved GQA layout
        base_shape_qkvz = mixed_qkvz.size()[:-1]
        base_shape_ba = mixed_ba.size()[:-1]
        ng = self.num_k_heads // self.tp_size

        new_tensor_shape_qkvz = base_shape_qkvz + (
            ng,
            (
                self.head_k_dim
                + self.head_k_dim
                + (self.head_v_dim + self.head_v_dim)
                * self.num_v_heads
                // self.num_k_heads
            ),
        )
        new_tensor_shape_ba = base_shape_ba + (
            ng,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads,
        ]

        (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=-1)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=-1)

        mixed_qkv_logical = torch.cat(
            [
                query.reshape(num_tokens, -1),
                key.reshape(num_tokens, -1),
                value.reshape(num_tokens, -1),
            ],
            dim=-1,
        )

        # The split above produces non-contiguous views into the interleaved
        # buffer.  Concatenating everything into a single flat tensor forces a
        # contiguous copy, then slicing back out gives contiguous q/k/v/z/b/a
        # tensors that downstream kernels require.  Doing this in one cat+slice
        # keeps torch.compile in a single Triton graph instead of emitting
        # separate copy kernels per tensor.  The original code used
        # rearrange(...).contiguous() on each tensor individually.
        fused = torch.cat(
            [
                mixed_qkv_logical.reshape(-1),
                z.reshape(-1),
                b.reshape(-1),
                a.reshape(-1),
            ],
            dim=0,
        )

        curr = 0
        qkv_numel = mixed_qkv_logical.numel()
        z_numel = z.numel()
        b_numel = b.numel()
        a_numel = a.numel()

        mixed_qkv_out = fused[curr : curr + qkv_numel].view(num_tokens, -1)
        curr += qkv_numel

        z_out = fused[curr : curr + z_numel].view(
            num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim
        )
        curr += z_numel

        b_out = fused[curr : curr + b_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )
        curr += b_numel

        a_out = fused[curr : curr + a_numel].view(
            num_tokens, self.num_v_heads // self.tp_size
        )

        return mixed_qkv_out, z_out, b_out, a_out

    def rearrange_mixed_qkv(self, mixed_qkv):
        """Split packed qkv into contiguous (1, seq, heads, dim) tensors.

        The original code used ``rearrange(x, "l (h d) -> 1 l h d", d=...)``
        followed by ``.contiguous()`` on each tensor.  This version flattens
        all three splits into a single buffer via ``torch.cat`` so that
        torch.compile emits one Triton copy kernel instead of three separate
        contiguous() calls.
        """
        if mixed_qkv is None:
            return None, None, None

        seq_len = mixed_qkv.shape[0]
        q_dim = self.key_dim // self.tp_size
        k_dim = self.key_dim // self.tp_size
        v_dim = self.value_dim // self.tp_size

        query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)

        fused = torch.cat(
            [query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0
        )

        q_size = seq_len * q_dim
        k_size = seq_len * k_dim

        q_contig = fused[0:q_size]
        k_contig = fused[q_size : q_size + k_size]
        v_contig = fused[q_size + k_size :]

        query = q_contig.view(1, seq_len, -1, self.head_k_dim)
        key = k_contig.view(1, seq_len, -1, self.head_k_dim)
        value = v_contig.view(1, seq_len, -1, self.head_v_dim)

        return query, key, value

    def _tiled_grouped_perms(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(tiled -> grouped, grouped -> tiled) index maps over the v-head axis.

        ggml tiled order places value head ``hv`` under key head ``hv % H``;
        the HF grouped order the FLA kernels assume places it under
        ``hv // (HV // H)``.
        """
        cached = self._v_head_perms.get(device)
        if cached is not None:
            return cached
        hv = self.num_v_heads // self.tp_size
        h = self.num_k_heads // self.tp_size
        rep = hv // h
        idx = torch.arange(hv, device=device)
        to_grouped = idx.reshape(rep, h).T.reshape(-1)
        to_tiled = idx.reshape(h, rep).T.reshape(-1)
        self._v_head_perms[device] = (to_grouped, to_tiled)
        return to_grouped, to_tiled

    def _regroup_v_heads(self, x: torch.Tensor, dim: int, inverse: bool = False):
        """Reorder one v-head axis between the tiled and grouped layouts."""
        to_grouped, to_tiled = self._tiled_grouped_perms(x.device)
        return x.index_select(dim, to_tiled if inverse else to_grouped)

    def _regroup_qkv_value_channels(self, mixed_qkv: torch.Tensor | None):
        """Reorder just the value block of a packed [q | k | v] tensor."""
        if mixed_qkv is None or not self.normalize_v_heads_to_grouped:
            return mixed_qkv
        v_off = 2 * (self.key_dim // self.tp_size)
        hv = self.num_v_heads // self.tp_size
        qk, v = mixed_qkv[..., :v_off], mixed_qkv[..., v_off:]
        v = self._regroup_v_heads(
            v.reshape(*v.shape[:-1], hv, self.head_v_dim), -2
        ).reshape(v.shape)
        return torch.cat([qk, v], dim=-1)

    def _kernel_order(self, name: str) -> torch.Tensor:
        """A per-v-head parameter in the order this platform's core expects.

        Cached rather than written back into the parameter: rewriting weights
        on the first forward frees/replaces storage inside the traced graph,
        which is what broke compiled GGUF startup once already.
        """
        param = getattr(self, name)
        if not self.normalize_v_heads_to_grouped:
            return param
        cache = self.__dict__.setdefault("_kernel_order_cache", {})
        hit = cache.get(name)
        if hit is None or hit.device != param.device or hit.dtype != param.dtype:
            hit = self._regroup_v_heads(param.detach(), 0)
            cache[name] = hit
        return hit

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self._forward_method(hidden_states)

    def _gated_norm(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """RMSNormGated over the core output; returns [..., Hv*Dv] rows."""
        z_shape_og = z.shape
        if (
            core_attn_out.dim() == 3
            and z.dim() == 3
            and self._fused_gdn_norm_ok(core_attn_out, z)
        ):
            # Fused RMSNormGated (norm_before_gate + silu) in one Metal
            # dispatch straight off the (tokens, Hv, D) views; verified
            # against forward_native (the oracle; same kill-switch as the
            # fused GDN step).
            from vllm.quixicore.ops import quixicore_ops

            core_attn_out = quixicore_ops.qwen_gdn_gated_norm(
                core_attn_out.contiguous(),
                z,
                self._fused_gdn_norm_weight(),
                self.norm.eps,
            )
            return core_attn_out.flatten(-2)
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        return core_attn_out.flatten(-2)  # ... h d -> ... (h d)

    def _output_projection(
        self,
        core_attn_out: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Part 3: RMSNormGated + output linear projection.

        The RMSNormGated + quant sequence is eligible for fusion
        by the compilation pass when fuse_norm_quant is enabled.
        """
        output, _ = self.out_proj(self._gated_norm(core_attn_out, z))
        return output

    def _fused_gdn_norm_ok(self, x: torch.Tensor, z: torch.Tensor) -> bool:
        ok = self.__dict__.get("_fused_gdn_norm_flag")
        if ok is None:
            norm = self.norm
            ok = (
                self._fused_gdn_enabled()
                and _FUSED_GDN_NORM_ENV
                and norm.group_size is None
                and norm.norm_before_gate
                and norm.activation in ("silu", "swish")
                and norm.bias is None
                and self.head_v_dim % 4 == 0
            )
            self.__dict__["_fused_gdn_norm_flag"] = ok
        return (
            ok
            and x.dtype == z.dtype
            and z.stride(-1) == 1
            and z.stride(-2) == z.shape[-1]
        )

    def _fused_gdn_norm_weight(self) -> torch.Tensor:
        w = self.__dict__.get("_fused_gdn_norm_w")
        if w is None:
            w = self.norm.weight.detach().to(torch.float32).contiguous()
            self.__dict__["_fused_gdn_norm_w"] = w
        return w

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """ROCm forward using AITER Triton fused projection+attention when
        available, otherwise falling back to the generic CUDA path."""
        if GDN_AITER_TRITON_AVAILABLE:
            num_tokens = hidden_states.size(0)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
            projected_states_qkvz = projected_states_qkvz.view(num_tokens, -1)
            projected_states_ba = projected_states_ba.view(num_tokens, -1)
            core_attn_out = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            z = torch.empty(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=projected_states_qkvz.dtype,
                device=projected_states_qkvz.device,
            )

            torch.ops.vllm.qwen_gdn_attention_core(
                projected_states_qkvz,
                projected_states_ba,
                z,
                core_attn_out,
                layer_name=_encode_layer_name(self.prefix),
                use_aiter=True,
            )

            return self._output_projection(core_attn_out, z)
        else:
            return self.forward_cuda(hidden_states)

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)
        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = self.split_ba(ba)
            b = b.contiguous()
            a = a.contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        # Note: we should not use torch.empty here like other attention backends,
        # see discussions in https://github.com/vllm-project/vllm/pull/28182
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.qwen_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        out = self._output_projection(core_attn_out, z)
        return out

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
        projected_states_ba, _ = self.in_proj_ba(hidden_states)

        # ============================================================
        # Part 2: Core Attention
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        z = torch.empty_like(core_attn_out)

        torch.ops.vllm.gdn_attention_core_xpu(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        z_shape_og = z.shape
        # Reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        assert not hasattr(self, "in_proj_qkv"), "lora isn't supported on CPU."

        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        if self.gqa_interleaved_layout:
            # Qwen3-Next: unpack the interleaved GQA layout
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                mixed_qkvz, ba
            )
            query, key, value = map(
                lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

        num_tokens = hidden_states.size(0)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.cpu_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            _encode_layer_name(self.prefix),
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
        out, _ = self.out_proj(core_attn_out)
        return out

    def forward_mps(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Metal forward: input projections, then the GDN core (fused Metal
        kernels when eligible, torch-native fallbacks otherwise; see
        gdn_mps_fallback.py for the numerics contract). The core returns the
        gated+normed rows, so only the output projection remains here."""
        assert not self.gqa_interleaved_layout, (
            "The MPS GDN path supports the Qwen3.5 non-interleaved layout only."
        )
        num_tokens = hidden_states.size(0)
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)

        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        # Strided view into the projection rows; the Metal norm kernel reads
        # it in place, the fallbacks gather it when they need it.
        z = z.view(num_tokens, -1, self.head_v_dim)

        core = self._forward_core_mps(mixed_qkv, ba, z, num_tokens)
        output, _ = self.out_proj(core)
        return output

    def _forward_core_mps(
        self,
        mixed_qkv: torch.Tensor,
        ba: torch.Tensor,
        z: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Non-speculative GDN core on Metal. Returns the gated+normed rows
        [num_tokens, Hv*Dv] in the activation dtype, ready for out_proj.
        Padded rows (warmup, or num_actual_tokens < num_tokens) are zeros,
        matching the previous zeros-container semantics (rmsnorm(0)*silu(z)
        is zero)."""
        out_width = (self.num_v_heads // self.tp_size) * self.head_v_dim

        def _zeros() -> torch.Tensor:
            return torch.zeros(
                (num_tokens, out_width),
                dtype=mixed_qkv.dtype,
                device=mixed_qkv.device,
            )

        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            # Profile/warmup run: nothing to prime (no Triton autotuners).
            return _zeros()

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            # Zero-length batches must never reach the kernels (degenerate
            # grids have produced silent hangs in other GDN stacks).
            return _zeros()

        if attn_metadata.spec_sequence_masks is not None:
            if self._metal_gdn_enabled():
                normed = self._forward_core_metal_spec(mixed_qkv, ba, z, attn_metadata)
            else:
                # The in-place native core carries its own spec-verify route
                # (fused Metal step when eligible, torch otherwise).
                normed = self._forward_core_native(
                    mixed_qkv, ba, z, num_tokens, num_actual_tokens
                )
        elif self._metal_gdn_enabled():
            normed = self._forward_core_metal(mixed_qkv, ba, z, attn_metadata)
        elif self._fused_gdn_enabled():
            normed = self._forward_core_native(
                mixed_qkv, ba, z, num_tokens, num_actual_tokens
            )
        else:
            normed = self._forward_core_torch(mixed_qkv, ba, z, attn_metadata)

        if num_actual_tokens == num_tokens:
            return normed
        out = _zeros()
        out[:num_actual_tokens] = normed
        return out

    def _forward_core_native(
        self,
        mixed_qkv: torch.Tensor,
        ba: torch.Tensor,
        z: torch.Tensor,
        num_tokens: int,
        num_actual_tokens: int,
    ) -> torch.Tensor:
        """Adapter onto ``_forward_core_mps_native`` (the in-place container
        core): split ``ba``, allocate the [tokens, Hv, Dv] container, run the
        core (fused Metal GDN step when eligible, torch scans otherwise, spec
        verify included), then the gated norm. Returns
        [num_actual_tokens, Hv*Dv]."""
        b, a = self.split_ba(ba)
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        self._forward_core_mps_native(mixed_qkv, b, a, core_attn_out)
        return self._gated_norm(
            core_attn_out[:num_actual_tokens], z[:num_actual_tokens]
        )

    def _forward_core_torch(
        self,
        mixed_qkv: torch.Tensor,
        ba: torch.Tensor,
        z: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        """Torch-native MPS fallbacks: conv (varlen prefill / single-token
        decode), the gated delta rule (sequential fp32 scan for prefill, one
        vectorized step for decode), then the gated RMSNorm. Returns
        [num_actual_tokens, Hv*Dv]."""
        from vllm.model_executor.layers.mamba.gdn.gdn_mps_fallback import (
            causal_conv1d_fn_native,
            causal_conv1d_update_native,
            gated_delta_rule_decode_native,
            gated_delta_rule_prefill_native,
            post_conv_prep_native,
        )

        num_actual_tokens = attn_metadata.num_actual_tokens
        b, a = self.split_ba(ba)
        b = b.contiguous()
        a = a.contiguous()

        # conv_state must be (..., dim, width-1) for the conv fallbacks.
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self.kv_cache[1]

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # 1. Convolution sequence transformation
        if attn_metadata.num_prefills > 0:
            mixed_qkv = causal_conv1d_fn_native(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state,
                attn_metadata.has_initial_state,
                attn_metadata.non_spec_state_indices_tensor,
                attn_metadata.non_spec_query_start_loc,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv = causal_conv1d_update_native(
                mixed_qkv,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                attn_metadata.non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            )
        else:
            return torch.zeros(
                (num_actual_tokens, z.shape[1] * z.shape[2]),
                dtype=z.dtype,
                device=z.device,
            )

        # 2. Gated delta rule. The non-spec batch is decode-first sorted;
        # peel the single-token decode sequences off the front.
        num_decode_tokens = attn_metadata.num_decode_tokens
        split_non_spec = (
            attn_metadata.num_prefills > 0 and attn_metadata.num_decodes > 0
        )

        core_attn_out_decode = None
        if attn_metadata.num_decodes > 0:
            decode_qkv = mixed_qkv[:num_decode_tokens] if split_non_spec else mixed_qkv
            query_d, key_d, value_d = self.rearrange_mixed_qkv(decode_qkv)
            core_attn_out_decode = gated_delta_rule_decode_native(
                A_log=self.A_log,
                a=a[:num_decode_tokens] if split_non_spec else a,
                b=b[:num_decode_tokens] if split_non_spec else b,
                dt_bias=self.dt_bias,
                q=query_d,
                k=key_d,
                v=value_d,
                ssm_state=ssm_state,
                state_indices=attn_metadata.non_spec_state_indices_tensor,  # type: ignore[arg-type]
                use_qk_l2norm=True,
            )

        core_attn_out_prefill = None
        if attn_metadata.num_prefills > 0:
            conv_output_prefill = mixed_qkv[num_decode_tokens:]
            a_prefill = a[num_decode_tokens:]
            b_prefill = b[num_decode_tokens:]
            query_p, key_p, value_p, g_p, beta_p = post_conv_prep_native(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
            )

            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0

            core_attn_out_prefill, last_recurrent_state = (
                gated_delta_rule_prefill_native(
                    q=query_p.unsqueeze(0),
                    k=key_p.unsqueeze(0),
                    v=value_p.unsqueeze(0),
                    g=g_p.unsqueeze(0),
                    beta=beta_p.unsqueeze(0),
                    initial_state=initial_state,
                    cu_seqlens=attn_metadata.prefill_query_start_loc,  # type: ignore[arg-type]
                )
            )
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

        # 3. Merge (decode-first order, mirroring _forward_core)
        if core_attn_out_decode is not None and core_attn_out_prefill is not None:
            merged = torch.cat([core_attn_out_decode, core_attn_out_prefill], dim=1)
        elif core_attn_out_prefill is not None:
            merged = core_attn_out_prefill
        else:
            assert core_attn_out_decode is not None
            merged = core_attn_out_decode

        # 4. Gated RMSNorm (previously in _output_projection; the cast to the
        # activation dtype matches the old container copy).
        core2d = merged.squeeze(0).to(z.dtype).reshape(-1, self.head_v_dim)
        z2d = z[:num_actual_tokens].reshape(-1, self.head_v_dim)
        normed = self.norm(core2d, z2d)
        return normed.view(num_actual_tokens, -1)

    def _metal_gdn_enabled(self) -> bool:
        state = self._metal_gdn
        if state is None:
            state = self._resolve_metal_gdn()
            self._metal_gdn = state
        return state is not False

    def _resolve_metal_gdn(self) -> object:
        """Decide once, at the first core call (the state pools are bound by
        then), whether the five-kernel Metal GDN path can serve this layer,
        and cache the fp32 parameter copies it needs. Any failed contract
        check logs its reason and pins the torch fallback."""
        from types import SimpleNamespace

        def off(reason: str) -> bool:
            logger.warning_once("Metal GDN kernels disabled: %s", reason)
            return False

        if os.environ.get("VLLM_METAL_GDN", "1") == "0":
            return off("VLLM_METAL_GDN=0")
        from vllm.quixicore.ops import _qc, quixicore_ops

        if not quixicore_ops.is_available():
            return off("vllm._quixicore_C unavailable")
        if not hasattr(_qc(), "gdn_recur"):
            return off("extension predates the GDN bindings")
        if self.activation not in ("silu", "swish"):
            return off(f"unsupported conv activation {self.activation!r}")
        if self.conv1d.bias is not None:
            return off("gdn_short_conv has no bias buffer")
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        head_k_dim, head_v_dim = self.head_k_dim, self.head_v_dim
        if head_k_dim not in (64, 128) or head_v_dim not in (64, 128):
            return off(f"head dims ({head_k_dim},{head_v_dim}) not instantiated")
        if num_v_heads % num_k_heads != 0:
            return off("num_v_heads not a multiple of num_k_heads")
        conv_state, ssm_state = self.kv_cache[0], self.kv_cache[1]
        conv_dim = self.conv_dim // self.tp_size
        width = self.conv_kernel_size
        # bind_kv_cache slices each state out of a per-block byte page, so
        # the pools are contiguous WITHIN a slot with a wider slot stride;
        # the kernels take that stride explicitly.
        # With speculation the pool carries width-1+num_spec columns per
        # channel; the non-spec ring occupies the first width-1 of them.
        if not (
            conv_state.dtype == torch.float32
            and conv_state.dim() == 3
            and conv_state.shape[1] == conv_dim
            and conv_state.shape[2] >= width - 1
            and conv_state.stride(2) == 1
            and conv_state.stride(1) == conv_state.shape[2]
        ):
            return off(
                "conv state pool must be fp32 [slots, dim, >=width-1] with "
                "contiguous slot rows; set VLLM_SSM_CONV_STATE_LAYOUT=DS and "
                f"mamba_cache_dtype=float32 (got {conv_state.dtype}, "
                f"{tuple(conv_state.shape)}, strides "
                f"{tuple(conv_state.stride())})"
            )
        if not (
            ssm_state.dtype == torch.float32
            and ssm_state.dim() == 4
            and tuple(ssm_state.shape[1:]) == (num_v_heads, head_v_dim, head_k_dim)
            and ssm_state.stride(3) == 1
            and ssm_state.stride(2) == head_k_dim
            and ssm_state.stride(1) == head_v_dim * head_k_dim
        ):
            return off(
                "ssm state pool must be fp32 [slots, Hv, Dv, Dk] with "
                f"contiguous slot rows (got {ssm_state.dtype}, "
                f"{tuple(ssm_state.shape)}, strides {tuple(ssm_state.stride())})"
            )
        state = SimpleNamespace(
            conv_w=self.conv1d.weight.view(conv_dim, width).float().contiguous(),
            A_log=self.A_log.float().contiguous(),
            dt_bias=self.dt_bias.float().contiguous(),
            # The kernel norm is rms-form rsqrt(ss/Dk + eps); these constants
            # recover l2norm exactly and fold the q * Dk^-0.5 output scale:
            # eps = 1e-6/Dk, k_scale = Dk^-0.5, q_scale = Dk^-1.
            eps=1e-6 / head_k_dim,
            q_scale=float(head_k_dim) ** -1.0,
            k_scale=float(head_k_dim) ** -0.5,
            checked_indices=False,
            # Dispatch-fusion routes (kill switches + extension capability).
            # The fused prep reads ba as packed [b|a] rows, which the
            # replicated-ba TP slicing would break (CUDA-only today).
            fused_prep=(
                _QC_GDN_FUSEPREP
                and hasattr(_qc(), "gdn_fused_prepare")
                and not (self.disable_tp_for_ba_proj and self.tp_size > 1)
            ),
            # Speculative verify route: fused prep in rewind mode + the
            # per-position-checkpointing recurrence.
            spec=hasattr(_qc(), "gdn_recur_spec"),
            fused_norm=(_QC_GDN_FUSENORM and hasattr(_qc(), "gdn_gated_rmsnorm_f32")),
        )
        logger.info_once("Metal GDN kernels enabled (%s)", self.prefix)
        return state

    def _forward_core_metal(
        self,
        mixed_qkv: torch.Tensor,
        ba: torch.Tensor,
        z: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        """The Metal GDN chain (gdn.metal through _quixicore_C): on
        pure-decode batches one fused prepare dispatch (conv + qkv
        split/norm + gating, reading the qkvz/ba projection rows in place),
        otherwise the varlen conv / prepare / gate kernels, then the
        delta-rule scan against the persistent pools and the gated RMSNorm
        (reading z in place on the fused path) — fp32 activations throughout
        the recurrence chain. Kernel outputs were oracle-gated bit-for-bit
        against the torch fallbacks in this file before promotion; the
        fused prep/norm are gated on
        VLLM_QC_GDN_FUSEPREP/FUSENORM. Per-request fresh state is handled by
        pre-zeroing those pool rows, then always loading initial state.
        Returns the gated+normed rows [num_actual_tokens, Hv*Dv]."""
        from vllm.quixicore.ops import quixicore_ops as qc

        st = self._metal_gdn
        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_seqs = num_decodes + num_prefills

        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        head_k_dim, head_v_dim = self.head_k_dim, self.head_v_dim

        cu_all = attn_metadata.non_spec_query_start_loc
        state_indices = attn_metadata.non_spec_state_indices_tensor[:num_seqs]
        if not st.checked_indices:
            # One-time host check: the kernels index the pools directly, so
            # a PAD_SLOT_ID (-1) entry would read out of bounds in gdn_recur.
            # A hard raise (not assert): the consequence is a silent OOB
            # read, so this must survive python -O.
            if int(state_indices.min().item()) < 0:
                raise RuntimeError("negative state index reached the Metal GDN path")
            st.checked_indices = True

        if num_prefills > 0:
            # Fresh sequences start from empty conv history / S = 0: zero
            # their pool rows sync-free, then the kernels always
            # load_initial. where() rather than a masked multiply — a
            # never-written pool row can hold Inf/NaN bit patterns and
            # 0 * Inf would poison the state instead of clearing it.
            keep = attn_metadata.prefill_has_initial_state
            rows = attn_metadata.prefill_state_indices.long()
            zero = torch.zeros((), dtype=torch.float32, device=conv_state.device)
            conv_state[rows] = torch.where(keep[:, None, None], conv_state[rows], zero)
            ssm_state[rows] = torch.where(
                keep[:, None, None, None], ssm_state[rows], zero
            )

        if (
            st.fused_prep
            and num_prefills == 0
            and mixed_qkv.stride(-1) == 1
            and ba.stride(-1) == 1
        ):
            # One dispatch for conv+silu, q/k norm, v, and decay/beta,
            # reading the projection rows in place — replaces the three
            # kernels below plus the fp32 cast and the b/a copies.
            q, k, v, decay, beta = qc.gdn_fused_prepare(
                mixed_qkv[:num_actual_tokens],
                ba[:num_actual_tokens],
                st.conv_w,
                conv_state,
                cu_all,
                state_indices,
                st.A_log,
                st.dt_bias,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                st.eps,
                st.q_scale,
                st.k_scale,
                True,
            )
        else:
            x = mixed_qkv[:num_actual_tokens].float()
            b, a = self.split_ba(ba[:num_actual_tokens])
            a = a.contiguous()
            b = b.contiguous()
            conv_out = qc.gdn_short_conv(
                x, st.conv_w, conv_state, cu_all, state_indices, True, True
            )
            q, k, v = qc.gdn_qkv_prepare(
                conv_out,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                st.eps,
                st.q_scale,
                st.k_scale,
            )
            decay, beta = qc.gdn_gate_beta(a, b, st.A_log, st.dt_bias)

        out_decode = None
        if num_decodes > 0:
            out_decode = qc.gdn_recur(
                q[:num_decode_tokens],
                k[:num_decode_tokens],
                v[:num_decode_tokens],
                decay[:num_decode_tokens],
                beta[:num_decode_tokens],
                ssm_state,
                cu_all[: num_decodes + 1],
                state_indices[:num_decodes],
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                True,
            )
        out_prefill = None
        if num_prefills > 0:
            out_prefill = qc.gdn_recur(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                decay[num_decode_tokens:],
                beta[num_decode_tokens:],
                ssm_state,
                attn_metadata.prefill_query_start_loc,
                attn_metadata.prefill_state_indices,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                True,
            )

        norm_w = self.norm.weight
        norm_eps = self.norm.eps
        if (
            st.fused_norm
            and z.stride(-1) == 1
            and z.stride(1) == head_v_dim
            and norm_w.dtype == z.dtype
        ):
            # Norm straight off the fp32 recurrence output with z read in
            # place — replaces the bf16 cast, the z gather, and the container
            # copy (the in-kernel rounding is bitwise the same cast).
            parts = []
            if out_decode is not None:
                parts.append(
                    qc.gdn_gated_rmsnorm_f32(
                        out_decode.view(-1, head_v_dim),
                        z[:num_decode_tokens],
                        norm_w,
                        norm_eps,
                    )
                )
            if out_prefill is not None:
                parts.append(
                    qc.gdn_gated_rmsnorm_f32(
                        out_prefill.view(-1, head_v_dim),
                        z[num_decode_tokens:num_actual_tokens],
                        norm_w,
                        norm_eps,
                    )
                )
            normed = parts[0] if len(parts) == 1 else torch.cat(parts)
            return normed

        if out_decode is not None and out_prefill is not None:
            merged = torch.cat([out_decode, out_prefill])
        elif out_prefill is not None:
            merged = out_prefill
        else:
            assert out_decode is not None
            merged = out_decode
        y2d = merged.to(z.dtype).view(-1, head_v_dim)
        z2d = z[:num_actual_tokens].reshape(-1, head_v_dim)
        normed = qc.gdn_gated_rmsnorm(y2d, z2d.contiguous(), norm_w, norm_eps)
        return normed.view(num_actual_tokens, -1)

    def _forward_core_metal_spec(
        self,
        mixed_qkv: torch.Tensor,
        ba: torch.Tensor,
        z: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ) -> torch.Tensor:
        """Speculative-verify GDN core on Metal (target-side DFlash verify).
        Mirrors the CUDA spec branch: spec rows run the fused prep in
        conv-rewind mode plus the per-position-checkpointing recurrence
        (gdn_recur_spec); non-spec rows of a mixed batch (always prefills —
        the builder reclassifies decodes) run the varlen-general fused prep +
        gdn_recur; outputs merge back into token order before one gated
        RMSNorm. Returns [num_actual_tokens, Hv*Dv] in the activation
        dtype."""
        from vllm.quixicore.ops import quixicore_ops as qc

        st = self._metal_gdn
        if not (st.fused_prep and st.spec):
            raise NotImplementedError(
                "Speculative GDN on MPS needs gdn_fused_prepare + "
                "gdn_recur_spec (VLLM_QC_GDN_FUSEPREP=1 and a current "
                "quixicore .so)"
            )
        conv_state = self.kv_cache[0]
        ssm_state = self.kv_cache[1]

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_prefills = attn_metadata.num_prefills
        num_spec_decodes = attn_metadata.num_spec_decodes
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        head_k_dim, head_v_dim = self.head_k_dim, self.head_v_dim

        # Per-step derived tensors are identical for every GDN layer; compute
        # them once on the shared metadata object.
        cache = getattr(attn_metadata, "_mps_spec_cache", None)
        if cache is None:
            from types import SimpleNamespace

            spec_tab = attn_metadata.spec_state_indices_tensor
            accepted = attn_metadata.num_accepted_tokens
            if accepted.dtype != torch.int32:
                accepted = accepted.to(torch.int32)
            cache = SimpleNamespace(
                slot_table=spec_tab,
                conv_slots=spec_tab[:, 0].contiguous(),
                num_accepted=accepted.contiguous(),
                spec_cu=attn_metadata.spec_query_start_loc[: num_spec_decodes + 1],
                spec_indx=(
                    attn_metadata.spec_token_indx.long() if num_prefills > 0 else None
                ),
                non_spec_indx=(
                    attn_metadata.non_spec_token_indx.long()
                    if num_prefills > 0
                    else None
                ),
            )
            attn_metadata._mps_spec_cache = cache

        pure_spec = num_prefills == 0
        if pure_spec:
            qkvz_spec = mixed_qkv[:num_actual_tokens]
            ba_spec = ba[:num_actual_tokens]
        else:
            qkvz_spec = mixed_qkv[:num_actual_tokens].index_select(0, cache.spec_indx)
            ba_spec = ba[:num_actual_tokens].index_select(0, cache.spec_indx)

        q, k, v, decay, beta = qc.gdn_fused_prepare(
            qkvz_spec,
            ba_spec,
            st.conv_w,
            conv_state,
            cache.spec_cu,
            cache.conv_slots,
            st.A_log,
            st.dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            st.eps,
            st.q_scale,
            st.k_scale,
            True,
            cache.num_accepted,
        )
        out_spec = qc.gdn_recur_spec(
            q,
            k,
            v,
            decay,
            beta,
            ssm_state,
            cache.spec_cu,
            cache.slot_table,
            cache.num_accepted,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        )

        if pure_spec:
            merged = out_spec
        else:
            # Non-spec rows (all prefills here). Fresh sequences pre-zero
            # their pool rows exactly as the non-spec path does.
            keep = attn_metadata.prefill_has_initial_state
            rows = attn_metadata.prefill_state_indices.long()
            zero = torch.zeros((), dtype=torch.float32, device=conv_state.device)
            conv_state[rows] = torch.where(keep[:, None, None], conv_state[rows], zero)
            ssm_state[rows] = torch.where(
                keep[:, None, None, None], ssm_state[rows], zero
            )

            qkvz_ns = mixed_qkv[:num_actual_tokens].index_select(0, cache.non_spec_indx)
            ba_ns = ba[:num_actual_tokens].index_select(0, cache.non_spec_indx)
            cu_ns = attn_metadata.non_spec_query_start_loc
            slots_ns = attn_metadata.non_spec_state_indices_tensor[:num_prefills]
            qn, kn, vn, decayn, betan = qc.gdn_fused_prepare(
                qkvz_ns,
                ba_ns,
                st.conv_w,
                conv_state,
                cu_ns,
                slots_ns,
                st.A_log,
                st.dt_bias,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                st.eps,
                st.q_scale,
                st.k_scale,
                True,
            )
            out_ns = qc.gdn_recur(
                qn,
                kn,
                vn,
                decayn,
                betan,
                ssm_state,
                cu_ns[: num_prefills + 1],
                slots_ns,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                True,
            )
            merged = torch.empty(
                (num_actual_tokens, out_spec.shape[1]),
                dtype=out_spec.dtype,
                device=out_spec.device,
            )
            merged.index_copy_(0, cache.spec_indx, out_spec)
            merged.index_copy_(0, cache.non_spec_indx, out_ns)

        norm_w = self.norm.weight
        norm_eps = self.norm.eps
        if (
            st.fused_norm
            and z.stride(-1) == 1
            and z.stride(1) == head_v_dim
            and norm_w.dtype == z.dtype
        ):
            return qc.gdn_gated_rmsnorm_f32(
                merged.view(-1, head_v_dim),
                z[:num_actual_tokens],
                norm_w,
                norm_eps,
            )
        y2d = merged.to(z.dtype).view(-1, head_v_dim)
        z2d = z[:num_actual_tokens].reshape(-1, head_v_dim)
        normed = qc.gdn_gated_rmsnorm(y2d, z2d.contiguous(), norm_w, norm_eps)
        return normed.view(num_actual_tokens, -1)

    def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:
        """Warm up GDN prefill kernels during V1 profiling.

        During V1 profile runs, ``_forward_core`` returns early because
        ``attn_metadata`` is ``None``, so the autotuned kernels used by
        ``chunk_gated_delta_rule`` (e.g. ``solve_tril``,
        ``chunk_scaled_dot_kkt``) are never invoked.  After profiling,
        vLLM allocates KV cache using most of the remaining GPU memory.
        When the first real inference triggers the autotuner it OOMs
        because there is not enough memory left for benchmarking.

        This method runs minimal forward passes through
        ``chunk_gated_delta_rule`` with small dummy tensors to force
        autotuning while GPU memory is still plentiful.  The autotuner
        results are cached globally, so only the first layer incurs
        actual benchmarking cost.

        All kernels including ``chunk_fwd_kernel_o`` now use a fixed
        ``BT = chunk_size`` (64).  A single warmup pass with T = 64
        is sufficient to populate the autotuner cache.

        The decode path uses ``gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule``
        which has fixed kernel parameters (no autotuning), so only the
        prefill (chunked) path needs warming up.
        """
        if self._prefill_kernels_warmed_up:
            return
        self._prefill_kernels_warmed_up = True

        device = qkv_or_qkvz.device
        dtype = qkv_or_qkvz.dtype
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        _, state_dtype = self.get_state_dtype()

        # All kernels use BT = chunk_size, so a single pass with T = chunk_size
        # is sufficient to populate every autotuner cache. Mirror the real
        # prefill path here: build q/k/v/g/beta via fused_post_conv_prep and
        # then run chunk_gated_delta_rule with in-kernel L2 norm disabled.
        T = FLA_CHUNK_SIZE
        dummy_mixed_qkv = torch.randn(
            T, qkv_or_qkvz.shape[-1] - v_dim, device=device, dtype=dtype
        )
        dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=dummy_mixed_qkv,
            a=dummy_a,
            b=dummy_b,
            A_log=self._kernel_order("A_log"),
            dt_bias=self._kernel_order("dt_bias"),
            num_k_heads=num_k_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            apply_l2norm=True,
            output_g_exp=False,
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)
        state = torch.zeros(
            1,
            num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            device=device,
            dtype=state_dtype,
        )
        cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)

        # CuteDSL kernels require metadata
        chunk_indices = None
        chunk_offsets = None
        if self.gdn_prefill_backend == "cutedsl":
            from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
                prepare_metadata_cutedsl,
            )

            chunk_indices, chunk_offsets = prepare_metadata_cutedsl(cu_seqlens, T)

        try:
            self.chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
        except Exception:
            logger.warning(
                "GDN prefill kernel warmup (T=%d) failed for "
                "layer %s. First inference may OOM due to "
                "autotuner.",
                T,
                self.prefix,
                exc_info=True,
            )
        else:
            logger.debug(
                "GDN prefill kernel warmup (T=%d) completed for layer %s",
                T,
                self.prefix,
            )
        finally:
            del (
                dummy_mixed_qkv,
                q,
                k,
                v,
                dummy_a,
                dummy_b,
                g,
                beta,
                state,
                cu_seqlens,
                chunk_indices,
                chunk_offsets,
            )

        torch.accelerator.empty_cache()

    def _forward_core_rocm(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """ROCm AITER fast path: conv1d + recurrent attention from packed
        qkvz/ba layout.

        For decode-only (no spec, no prefill) interleaved-GQA layouts,
        dispatches directly to ``_forward_core_decode_aiter``. Otherwise unpacks
        the packed layout and falls through to ``_forward_core``.

        Args:
            qkvz: packed [q, k, v, z] projection (num_tokens, qkvz_dim)
            ba:   packed [b, a] gating vectors    (num_tokens, 2*num_heads)
            z_out: **output** buffer for z        (num_tokens, num_heads,
                   head_dim); mutated in-place.
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            v_dim = core_attn_out.shape[-1] * core_attn_out.shape[-2]
            self._warmup_prefill_kernels(qkvz, v_dim)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # The AITER fused reshape/conv kernel expects Qwen3-Next's interleaved
        # GQA layout. Qwen3.5 uses a non-interleaved q/k/v/z layout and must use
        # the generic path below to split/rearrange inputs correctly.
        if (
            self.gqa_interleaved_layout
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_aiter(
                qkvz=qkvz,
                ba=ba,
                z_out=z_out,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        core_attn_out.zero_()
        num_tokens_all = qkvz.shape[0]
        mixed_qkv, z, b, a = self.prepare_gdn_attention_core_inputs(
            qkvz, ba, num_tokens_all
        )
        z_out[:] = z
        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Core conv1d + recurrent attention (standard path).

        Args:
            mixed_qkv: packed [q, k, v] projection (num_tokens, qkv_dim)
            b: beta gating vector                   (num_tokens, num_heads)
            a: alpha gating vector                  (num_tokens, num_heads)
            core_attn_out: Pre-allocated output buffer for attention results.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if self.normalize_v_heads_to_grouped:
            # Ahead of the spec/non-spec split so every view derived below
            # inherits the grouped order (one branch aliases these tensors
            # rather than slicing them, so reordering later would double-apply).
            a = self._regroup_v_heads(a, -1)
            b = self._regroup_v_heads(b, -1)

        if (
            self.enable_packed_recurrent_decode
            and attn_metadata.spec_sequence_masks is None
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
        ):
            return self._forward_core_decode_non_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                a_spec = a
                b_spec = b
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                a_spec = a.index_select(0, spec_token_indx)
                b_spec = b.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            # spec_state_indices_tensor is always set when spec_sequence_masks is set
            assert spec_state_indices_tensor is not None
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][  # type: ignore[index]
                    : attn_metadata.num_spec_decodes  # type: ignore[attr-defined]
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec_T,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            assert mixed_qkv_non_spec is not None
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        if self.normalize_v_heads_to_grouped:
            # The conv is per-channel and its weights stay in the stored
            # (tiled) order, so the value channels are reordered here: after
            # the conv, before anything pairs value heads with key heads.
            # `a`/`b` were already reordered above, ahead of the spec split.
            mixed_qkv_spec = self._regroup_qkv_value_channels(mixed_qkv_spec)
            mixed_qkv_non_spec = self._regroup_qkv_value_channels(mixed_qkv_non_spec)

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)

        # Split mixed non-spec-decode+prefill to process independently
        split_non_spec = (
            spec_sequence_masks is None
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes > 0
        )
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_metadata.num_prefills > 0:
            assert mixed_qkv_non_spec is not None, (
                "mixed_qkv_non_spec must be provided for prefill path"
            )
            if spec_sequence_masks is not None:
                a_non_spec = a.index_select(0, non_spec_token_indx)
                b_non_spec = b.index_select(0, non_spec_token_indx)
            else:
                a_non_spec = a
                b_non_spec = b

            if split_non_spec:
                conv_output_prefill = mixed_qkv_non_spec[num_decode_tokens:]
                a_prefill = a_non_spec[num_decode_tokens:]
                b_prefill = b_non_spec[num_decode_tokens:]
            else:
                conv_output_prefill = mixed_qkv_non_spec
                a_prefill = a_non_spec
                b_prefill = b_non_spec

            (
                query_non_spec,
                key_non_spec,
                value_non_spec,
                g_non_spec,
                beta_non_spec,
            ) = fused_post_conv_prep(
                conv_output=conv_output_prefill,
                a=a_prefill,
                b=b_prefill,
                A_log=self._kernel_order("A_log"),
                dt_bias=self._kernel_order("dt_bias"),
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                apply_l2norm=True,
                output_g_exp=False,
            )
            query_non_spec = query_non_spec.unsqueeze(0)
            key_non_spec = key_non_spec.unsqueeze(0)
            value_non_spec = value_non_spec.unsqueeze(0)
            g_non_spec = g_non_spec.unsqueeze(0)
            beta_non_spec = beta_non_spec.unsqueeze(0)
        else:
            query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec
            )
            g_non_spec = None
            beta_non_spec = None

        # 2. Recurrent attention

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self._kernel_order("A_log"),
                    a=a_spec,
                    b=b_spec,
                    dt_bias=self._kernel_order("dt_bias"),
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_spec_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process non-spec-decode part
        if split_non_spec:
            query_decode, key_decode, value_decode = self.rearrange_mixed_qkv(
                mixed_qkv_non_spec[:num_decode_tokens]  # type: ignore[index]
            )
            core_attn_out_decode, _ = fused_sigmoid_gating_delta_rule_update(
                A_log=self._kernel_order("A_log"),
                a=a[:num_decode_tokens],
                b=b[:num_decode_tokens],
                dt_bias=self._kernel_order("dt_bias"),
                q=query_decode,
                k=key_decode,
                v=value_decode,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                    : attn_metadata.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_decode = None

        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)
        if attn_metadata.num_prefills > 0:
            # State indices, initial-state mask and cu_seqlens for the chunk
            # kernel are precomputed by the metadata builder (the prefill tail
            # when decodes are peeled off, else the full non-spec batch), so they
            # don't need to be re-derived per layer.
            prefill_state_indices = attn_metadata.prefill_state_indices
            prefill_has_initial_state = attn_metadata.prefill_has_initial_state
            assert prefill_state_indices is not None
            assert prefill_has_initial_state is not None
            initial_state = ssm_state[prefill_state_indices]
            initial_state[~prefill_has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = self.chunk_gated_delta_rule(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.prefill_query_start_loc,
                chunk_indices=attn_metadata.chunk_indices,
                chunk_offsets=attn_metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
            )
            # Init cache
            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)

            if split_non_spec:
                # Stitch the peeled decode outputs in front of the prefill
                # outputs (decode-first order).
                core_attn_out_non_spec = torch.cat(
                    [core_attn_out_decode, core_attn_out_non_spec], dim=1
                )
        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self._kernel_order("A_log"),
                    a=a,
                    b=b,
                    dt_bias=self._kernel_order("dt_bias"),
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[  # type: ignore[index]
                        : attn_metadata.num_decodes
                        + 1  # type: ignore[attr-defined]
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)

        if self.normalize_v_heads_to_grouped:
            # Back to the stored order: `out_proj` and the `z` gate were never
            # reordered, so the recurrence output has to meet them there.
            core_attn_out[:num_actual_tokens] = self._regroup_v_heads(
                core_attn_out[:num_actual_tokens], -2, inverse=True
            )

    def _forward_core_decode_aiter(
        self,
        qkvz: torch.Tensor,
        ba: torch.Tensor,
        z_out: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        mixed_qkv_non_spec, b, a = (
            gdn_aiter_fused_reshape_causal_conv1d_update_single_token(
                qkvz,
                attn_metadata.num_actual_tokens,
                self.num_k_heads // self.tp_size,
                self.num_v_heads // self.tp_size,
                self.head_k_dim,
                self.head_v_dim,
                ba,
                z_out,
                core_attn_out,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[  # type: ignore[index]
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        )

        # 2. Recurrent attention
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(
            A_log=self._kernel_order("A_log"),
            a=a,
            b=b,
            dt_bias=self._kernel_order("dt_bias"),
            qkv=mixed_qkv_non_spec,
            key_dim=self.key_dim // self.tp_size,
            value_dim=self.value_dim // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            initial_state=ssm_state,
            inplace_final_state=True,
            cu_seqlens=non_spec_query_start_loc[: attn_metadata.num_decodes + 1],  # type: ignore[index]
            ssm_state_indices=non_spec_state_indices_tensor,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=core_attn_out.reshape(-1),
        )

    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """
        Core attention computation with a packed non-spec decode fast path.
        """
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            validate_data=False,
        )
        # Same layout normalization the general path applies: this fast path
        # returns before `_forward_core` reaches it, so without this the value
        # channels stay in the stored (tiled) order while `a`/`b` and the
        # per-head parameters have already been regrouped -- a mix that
        # corrupts every decoded token while prefill stays correct.
        mixed_qkv_non_spec = self._regroup_qkv_value_channels(mixed_qkv_non_spec)

        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self._kernel_order("A_log"),
            dt_bias=self._kernel_order("dt_bias"),
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],  # type: ignore[index]
            use_qk_l2norm_in_kernel=True,
        )
        if self.normalize_v_heads_to_grouped:
            # Back to the stored order for `out_proj` and the `z` gate.
            core_attn_out[:num_actual_tokens] = self._regroup_v_heads(
                core_attn_out[:num_actual_tokens], -2, inverse=True
            )
        return

    def _split_conved_qkv_mps(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split conved (N, T, conv_dim) tokens into per-head q/k/v tensors.

        Returns q, k of shape (N, T, H, head_k_dim) and v of shape
        (N, T, HV, head_v_dim).
        """
        key_dim = self.key_dim // self.tp_size
        value_dim = self.value_dim // self.tp_size
        q, k, v = torch.split(tokens, [key_dim, key_dim, value_dim], dim=-1)
        q = q.reshape(*tokens.shape[:2], -1, self.head_k_dim)
        k = k.reshape(*tokens.shape[:2], -1, self.head_k_dim)
        v = v.reshape(*tokens.shape[:2], -1, self.head_v_dim)
        return q, k, v

    def _forward_core_mps_native(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """Torch-native core for Apple Metal (MPS), filling ``core_attn_out``
        in place; reached through the ``_forward_core_native`` adapter when
        the five-kernel Metal path is off.

        Mirrors ``_forward_core`` semantics (causal_conv1d +
        fused_recurrent/chunk gated delta rule Triton kernels) with plain
        torch ops: fp32 state and math, in-place per-slot conv/SSM cache
        updates. Decode sequences (T=1) run as one batched recurrent step;
        each prefill sequence runs a token scan with per-token work batched
        across all heads.
        """
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            # V1 profiling run. There is no Triton autotuner to warm up on
            # MPS and no persistent state to touch; the zero-filled
            # core_attn_out is the expected output.
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]  # type: ignore[index]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        if _GDN_MPS_DEBUG:
            _gdn_mps_debug_dump(self.prefix, attn_metadata, attn_metadata_raw)

        if attn_metadata.spec_sequence_masks is not None:
            return self._forward_core_mps_spec(
                mixed_qkv=mixed_qkv,
                b=b,
                a=a,
                core_attn_out=core_attn_out,
                attn_metadata=attn_metadata,
            )

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_seqs = num_decodes + num_prefills
        if num_seqs == 0:
            return
        num_actual_tokens = attn_metadata.num_actual_tokens

        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1) for the conv math below.
        # DS layout stores it that way directly; SD layout needs a transpose.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        scale = self.head_k_dim**-0.5
        fused = num_decodes > 0 and self._fused_gdn_enabled()
        if fused:
            # Fused Metal step (kernels/serving_glue/gdn_step.metal): conv
            # window update + delta-rule scan for all decode sequences and
            # heads in two dispatches, fp32 state in place. Verified against
            # the torch path below (the oracle; VLLM_QWEN38_FUSED_GDN=0
            # restores it).
            plan = _fused_gdn_decode_plan(attn_metadata, num_decodes)
            self._fused_gdn_step(
                x=mixed_qkv,
                a=a,
                b=b,
                conv_state=conv_state,
                ssm_state=ssm_state,
                out=core_attn_out,
                num_seqs=num_decodes,
                seq_len=1,
                token_map=plan[0],
                conv_slot=plan[1],
                resume_slot=plan[1],
                store_slots=plan[2],
                num_accepted=None,
            )
            if num_prefills == 0:
                return

        state_indices = attn_metadata.non_spec_state_indices_tensor[  # type: ignore[index]
            :num_seqs
        ].to(torch.long)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).to(torch.float32)
        conv_bias = (
            self.conv1d.bias.to(torch.float32) if self.conv1d.bias is not None else None
        )

        # Gating terms for all tokens at once (fp32), matching
        # fused_sigmoid_gating_delta_rule_update / fused_post_conv_prep:
        # g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
        g_all = -self.A_log.to(torch.float32).exp() * nn.functional.softplus(
            a.to(torch.float32) + self.dt_bias.to(torch.float32),
            beta=1.0,
            threshold=20.0,
        )
        beta_all = torch.sigmoid(b.to(torch.float32))

        # 1. Decode sequences (single token each, batch-first order):
        # one fully batched conv update + recurrent step.
        if num_decodes > 0 and not fused:
            idx_d = state_indices[:num_decodes]
            # NULL_BLOCK_ID=0 marks padded entries; route their reads and
            # writes to the null block (slot 0) and zero their outputs,
            # matching the Triton kernels' skip semantics.
            valid_d = idx_d > 0
            idx_d = torch.where(valid_d, idx_d, torch.zeros_like(idx_d))

            # Non-spec semantics use only the first width-1 conv-state
            # columns (the slot may be wider when spec decode is configured).
            conv_width = conv_weights.size(-1) - 1
            x_d = mixed_qkv[:num_decodes].to(torch.float32).unsqueeze(-1)
            if _GDN_STATE_PROBE:
                x_nan = int(torch.isnan(x_d).sum().item())
                x_max = float(x_d.abs().max().item())
                _sprobe_check_and_record(
                    self.prefix,
                    conv_state,
                    ssm_state,
                    [int(s) for s in idx_d[valid_d].tolist()],
                    conv_width,
                    "dec-pre",
                    extras=f" in_nan={x_nan} in_absmax={x_max:.3e}",
                )
            conv_init_d = conv_state[idx_d, :, :conv_width].to(torch.float32)
            conv_out_d, conv_final_d = _causal_conv1d_native(
                x_d, conv_init_d, conv_weights, conv_bias, self.activation
            )
            conv_state[idx_d, :, :conv_width] = conv_final_d.to(conv_state.dtype)

            q_d, k_d, v_d = self._split_conved_qkv_mps(conv_out_d.transpose(1, 2))
            ssm_init_d = ssm_state[idx_d].to(torch.float32)
            o_d, ssm_final_d = _gdn_recurrent_scan_native(
                q_d,
                k_d,
                v_d,
                g_all[:num_decodes].unsqueeze(1),
                beta_all[:num_decodes].unsqueeze(1),
                scale,
                ssm_init_d,
                tiled_gqa=self.tiled_v_head_layout,
            )
            ssm_state[idx_d] = ssm_final_d.to(ssm_state.dtype)
            if _GDN_STATE_PROBE:
                o_nan = int(torch.isnan(o_d).sum().item())
                _sprobe_check_and_record(
                    self.prefix,
                    conv_state,
                    ssm_state,
                    [int(s) for s in idx_d[valid_d].tolist()],
                    conv_width,
                    "dec-post",
                    extras=f" out_nan={o_nan}",
                )
            o_d = torch.where(valid_d.view(-1, 1, 1, 1), o_d, torch.zeros_like(o_d))
            core_attn_out[:num_decodes] = o_d.squeeze(1).to(core_attn_out.dtype)

        # 2. Prefill sequences (varlen): per-sequence scan.
        if num_prefills > 0:
            qsl_cpu = attn_metadata.non_spec_query_start_loc[  # type: ignore[index]
                : num_seqs + 1
            ].cpu()
            has_initial_state = attn_metadata.has_initial_state
            has_init_cpu = (
                has_initial_state[:num_seqs].cpu()
                if has_initial_state is not None
                else torch.ones(num_seqs, dtype=torch.bool)
            )
            idx_cpu = state_indices.cpu()
            for i in range(num_decodes, num_seqs):
                start = int(qsl_cpu[i])
                end = int(qsl_cpu[i + 1])
                seq_len = end - start
                if seq_len <= 0:
                    continue
                slot = int(idx_cpu[i])
                if slot <= 0:
                    # NULL_BLOCK_ID: padded sequence, nothing to compute.
                    continue

                conv_width = conv_weights.size(-1) - 1
                x_i = mixed_qkv[start:end].to(torch.float32).T.unsqueeze(0)
                if _GDN_STATE_PROBE:
                    _sprobe_check_and_record(
                        self.prefix,
                        conv_state,
                        ssm_state,
                        [slot],
                        conv_width,
                        "pre-pre",
                        extras=(
                            f" has_init={bool(has_init_cpu[i])} "
                            f"seq_len={seq_len} in_nan="
                            f"{int(torch.isnan(x_i).sum().item())}"
                        ),
                    )
                if bool(has_init_cpu[i]):
                    conv_init = (
                        conv_state[slot, :, :conv_width].to(torch.float32).unsqueeze(0)
                    )
                    ssm_init = ssm_state[slot].to(torch.float32).unsqueeze(0)
                else:
                    conv_init = torch.zeros(
                        (1, conv_state.shape[1], conv_width),
                        dtype=torch.float32,
                        device=x_i.device,
                    )
                    ssm_init = torch.zeros(
                        (1, *ssm_state.shape[1:]),
                        dtype=torch.float32,
                        device=x_i.device,
                    )

                conv_out_i, conv_final_i = _causal_conv1d_native(
                    x_i, conv_init, conv_weights, conv_bias, self.activation
                )
                conv_state[slot, :, :conv_width] = conv_final_i[0].to(conv_state.dtype)

                q_i, k_i, v_i = self._split_conved_qkv_mps(conv_out_i.transpose(1, 2))
                o_i, ssm_final_i = _gdn_recurrent_scan_native(
                    q_i,
                    k_i,
                    v_i,
                    g_all[start:end].unsqueeze(0),
                    beta_all[start:end].unsqueeze(0),
                    scale,
                    ssm_init,
                    tiled_gqa=self.tiled_v_head_layout,
                )
                ssm_state[slot] = ssm_final_i[0].to(ssm_state.dtype)
                if _GDN_STATE_PROBE:
                    _sprobe_check_and_record(
                        self.prefix,
                        conv_state,
                        ssm_state,
                        [slot],
                        conv_width,
                        "pre-post",
                        extras=f" out_nan={int(torch.isnan(o_i).sum().item())}",
                    )
                core_attn_out[start:end] = o_i[0].to(core_attn_out.dtype)

    def _fused_gdn_enabled(self) -> bool:
        """Fused Metal GDN step available for this layer's geometry?

        Kill-switch: VLLM_QWEN38_FUSED_GDN=0 keeps the torch-native path
        (the verification oracle).
        """
        ok = self.__dict__.get("_fused_gdn_ok")
        if ok is None:
            ok = (
                _FUSED_GDN_ENV
                and self.head_k_dim == 128
                and self.head_v_dim % 8 == 0
                and self.conv_kernel_size <= 4
                and self.num_spec + 1 <= 16
                and _fused_gdn_available()
            )
            self.__dict__["_fused_gdn_ok"] = ok
            if ok:
                logger.info_once(
                    "Qwen GDN: fused Metal decode/verify step engaged "
                    "(VLLM_QWEN38_FUSED_GDN=0 restores the torch-native path)."
                )
        return ok

    def _fused_gdn_consts(self):
        consts = self.__dict__.get("_fused_gdn_const_cache")
        if consts is None:
            w = self.conv1d.weight
            conv_weight = w.view(w.size(0), w.size(-1)).to(torch.float32).contiguous()
            conv_bias = (
                self.conv1d.bias.to(torch.float32).contiguous()
                if self.conv1d.bias is not None
                else None
            )
            A_log = self.A_log.detach().to(torch.float32).contiguous()
            dt_bias = self.dt_bias.detach().to(torch.float32).contiguous()
            consts = (conv_weight, conv_bias, A_log, dt_bias)
            self.__dict__["_fused_gdn_const_cache"] = consts
        return consts

    def _fused_gdn_step(
        self,
        x: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        out: torch.Tensor,
        num_seqs: int,
        seq_len: int,
        token_map: torch.Tensor,
        conv_slot: torch.Tensor,
        resume_slot: torch.Tensor,
        store_slots: torch.Tensor,
        num_accepted: torch.Tensor | None,
    ) -> None:
        from vllm.quixicore.ops import quixicore_ops

        conv_weight, conv_bias, A_log, dt_bias = self._fused_gdn_consts()
        quixicore_ops.qwen_gdn_step(
            x,
            a,
            b,
            conv_state,
            ssm_state,
            conv_weight,
            conv_bias,
            A_log,
            dt_bias,
            token_map,
            conv_slot,
            resume_slot,
            store_slots,
            num_accepted,
            out,
            num_seqs,
            seq_len,
            self.num_k_heads // self.tp_size,
            self.tiled_v_head_layout,
            self.activation is not None,
            self.head_k_dim**-0.5,
        )

    def _forward_core_mps_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        """Spec-decode MPS core: verify-step forward with state rollback.

        Mirrors the spec branches of ``_forward_core`` (sections 1.1/2.1
        there): spec tokens run ``_gdn_spec_state_step_native`` (per-position
        SSM state stores + rolling conv window, resuming from the last
        accepted position), while non-spec sequences in the same batch run
        the ordinary per-sequence prefill scan on their gathered token
        stream (the metadata builder reclassifies non-spec decodes as
        prefills whenever spec decodes are present).
        """
        num_actual_tokens = attn_metadata.num_actual_tokens

        self_kv_cache = self.kv_cache
        # conv_state must be (..., dim, width-1+num_spec) for the conv math.
        conv_state = (
            self_kv_cache[0]
            if is_conv_state_dim_first()
            else self_kv_cache[0].transpose(-1, -2)
        )
        ssm_state = self_kv_cache[1]

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # num_decodes and num_spec_decodes are mutually exclusive (builder
        # invariant): mixed batches only carry prefills alongside spec.
        assert attn_metadata.num_decodes == 0
        mixed = attn_metadata.num_prefills > 0

        fused_spec = self._fused_gdn_enabled()
        if fused_spec:
            # Fused Metal verify step (kernels/serving_glue/gdn_step.metal):
            # the per-position loop (T <= block) runs inside the kernel with
            # the rollback contract (resume from slot[num_accepted-1], running
            # state stored to slot[t] after every position, NULL slots
            # skipped). Group plans are built once per step and cached on the
            # metadata; the torch path below is the oracle
            # (VLLM_QWEN38_FUSED_GDN=0).
            for plan in _fused_gdn_spec_plans(attn_metadata, self.num_spec):
                self._fused_gdn_step(
                    x=mixed_qkv,
                    a=a,
                    b=b,
                    conv_state=conv_state,
                    ssm_state=ssm_state,
                    out=core_attn_out,
                    num_seqs=plan[0],
                    seq_len=plan[1],
                    token_map=plan[2],
                    conv_slot=plan[3],
                    resume_slot=plan[4],
                    store_slots=plan[5],
                    num_accepted=plan[6],
                )
            if not mixed:
                return

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        ).to(torch.float32)
        conv_bias = (
            self.conv1d.bias.to(torch.float32) if self.conv1d.bias is not None else None
        )

        g_all = -self.A_log.to(torch.float32).exp() * nn.functional.softplus(
            a.to(torch.float32) + self.dt_bias.to(torch.float32),
            beta=1.0,
            threshold=20.0,
        )
        beta_all = torch.sigmoid(b.to(torch.float32))
        scale = self.head_k_dim**-0.5

        if mixed:
            spec_token_indx = attn_metadata.spec_token_indx.to(torch.long)  # type: ignore[union-attr]
            non_spec_token_indx = attn_metadata.non_spec_token_indx.to(  # type: ignore[union-attr]
                torch.long
            )

        if not fused_spec:
            self._forward_core_mps_spec_torch(
                mixed_qkv,
                g_all,
                beta_all,
                conv_state,
                ssm_state,
                conv_weights,
                conv_bias,
                scale,
                core_attn_out,
                attn_metadata,
                mixed,
                spec_token_indx if mixed else None,
            )
            if not mixed:
                return

        # 2. Non-spec sequences (prefills, incl. reclassified decodes):
        # per-sequence scan over the gathered non-spec token stream.
        qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        g_non_spec = g_all.index_select(0, non_spec_token_indx)
        beta_non_spec = beta_all.index_select(0, non_spec_token_indx)
        num_prefills = attn_metadata.num_prefills
        assert attn_metadata.non_spec_query_start_loc is not None
        qsl_cpu = attn_metadata.non_spec_query_start_loc[: num_prefills + 1].cpu()
        has_initial_state = attn_metadata.has_initial_state
        has_init_cpu = (
            has_initial_state[:num_prefills].cpu()
            if has_initial_state is not None
            else torch.ones(num_prefills, dtype=torch.bool)
        )
        assert attn_metadata.non_spec_state_indices_tensor is not None
        idx_cpu = attn_metadata.non_spec_state_indices_tensor[:num_prefills].cpu()
        out_non_spec = torch.zeros(
            (qkv_non_spec.size(0), *core_attn_out.shape[1:]),
            dtype=torch.float32,
            device=core_attn_out.device,
        )
        for i in range(num_prefills):
            start = int(qsl_cpu[i])
            end = int(qsl_cpu[i + 1])
            seq_len = end - start
            if seq_len <= 0:
                continue
            slot = int(idx_cpu[i])
            if slot <= 0:
                # NULL_BLOCK_ID: padded sequence, nothing to compute.
                continue

            x_i = qkv_non_spec[start:end].to(torch.float32).T.unsqueeze(0)
            if bool(has_init_cpu[i]):
                conv_init = (
                    conv_state[slot, :, : conv_weights.size(-1) - 1]
                    .to(torch.float32)
                    .unsqueeze(0)
                )
                ssm_init = ssm_state[slot].to(torch.float32).unsqueeze(0)
            else:
                conv_init = torch.zeros(
                    (1, conv_state.shape[1], conv_weights.size(-1) - 1),
                    dtype=torch.float32,
                    device=x_i.device,
                )
                ssm_init = torch.zeros(
                    (1, *ssm_state.shape[1:]),
                    dtype=torch.float32,
                    device=x_i.device,
                )

            conv_out_i, conv_final_i = _causal_conv1d_native(
                x_i, conv_init, conv_weights, conv_bias, self.activation
            )
            conv_state[slot, :, : conv_weights.size(-1) - 1] = conv_final_i[0].to(
                conv_state.dtype
            )

            q_i, k_i, v_i = self._split_conved_qkv_mps(conv_out_i.transpose(1, 2))
            o_i, ssm_final_i = _gdn_recurrent_scan_native(
                q_i,
                k_i,
                v_i,
                g_non_spec[start:end].unsqueeze(0),
                beta_non_spec[start:end].unsqueeze(0),
                scale,
                ssm_init,
                tiled_gqa=self.tiled_v_head_layout,
            )
            ssm_state[slot] = ssm_final_i[0].to(ssm_state.dtype)
            out_non_spec[start:end] = o_i[0]

        core_attn_out[:num_actual_tokens].index_copy_(
            0, non_spec_token_indx, out_non_spec.to(core_attn_out.dtype)
        )

    def _forward_core_mps_spec_torch(
        self,
        mixed_qkv: torch.Tensor,
        g_all: torch.Tensor,
        beta_all: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        conv_weights: torch.Tensor,
        conv_bias: torch.Tensor | None,
        scale: float,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
        mixed: bool,
        spec_token_indx: torch.Tensor | None,
    ) -> None:
        """Torch-native spec verify core (the fused kernel's oracle).

        Spec tokens run ``_gdn_spec_state_step_native`` grouped by query
        length; outputs land in ``core_attn_out`` through ``spec_token_indx``
        when the batch also carries prefills.
        """
        num_spec_decodes = attn_metadata.num_spec_decodes
        num_actual_tokens = attn_metadata.num_actual_tokens
        spec_state_indices = attn_metadata.spec_state_indices_tensor
        assert spec_state_indices is not None
        spec_state_indices = spec_state_indices[:num_spec_decodes].to(torch.long)
        num_accepted = attn_metadata.num_accepted_tokens
        assert num_accepted is not None
        num_accepted = num_accepted[:num_spec_decodes].to(
            device=spec_state_indices.device, dtype=torch.long
        )
        assert attn_metadata.spec_query_start_loc is not None
        spec_qsl_cpu = attn_metadata.spec_query_start_loc[: num_spec_decodes + 1].cpu()

        if mixed:
            assert spec_token_indx is not None
            qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            g_spec = g_all.index_select(0, spec_token_indx)
            beta_spec = beta_all.index_select(0, spec_token_indx)
        else:
            qkv_spec = mixed_qkv
            g_spec = g_all
            beta_spec = beta_all

        # Spec sequences, grouped by query length so the common
        # uniform-block case runs one batched conv + scan.
        seq_lens_cpu = (spec_qsl_cpu[1:] - spec_qsl_cpu[:-1]).tolist()
        starts_cpu = spec_qsl_cpu[:-1].tolist()
        accepted_cpu = num_accepted.cpu().tolist()
        slots_cpu = spec_state_indices.cpu()
        num_spec_tokens = int(spec_qsl_cpu[-1])
        out_spec = torch.zeros(
            (num_spec_tokens, *core_attn_out.shape[1:]),
            dtype=torch.float32,
            device=core_attn_out.device,
        )

        groups: dict[int, list[int]] = {}
        for i in range(num_spec_decodes):
            s = int(seq_lens_cpu[i])
            if s <= 0:
                continue
            if int(slots_cpu[i, 0]) <= 0 or int(slots_cpu[i, accepted_cpu[i] - 1]) <= 0:
                # NULL_BLOCK_ID conv/resume slot: padded entry, both Triton
                # kernels return without touching state or output.
                continue
            groups.setdefault(s, []).append(i)

        for s, idx_list in groups.items():
            sel = torch.tensor(
                idx_list, dtype=torch.long, device=spec_state_indices.device
            )
            tok_idx = torch.tensor(
                [starts_cpu[i] + t for i in idx_list for t in range(s)],
                dtype=torch.long,
                device=qkv_spec.device,
            )
            num_g = len(idx_list)
            x_g = qkv_spec.index_select(0, tok_idx).view(num_g, s, -1)
            g_g = g_spec.index_select(0, tok_idx).view(num_g, s, -1)
            beta_g = beta_spec.index_select(0, tok_idx).view(num_g, s, -1)

            o_g = _gdn_spec_state_step_native(
                x=x_g,
                g=g_g,
                beta=beta_g,
                conv_state=conv_state,
                ssm_state=ssm_state,
                slot_rows=spec_state_indices.index_select(0, sel),
                num_accepted=num_accepted.index_select(0, sel),
                conv_weights=conv_weights,
                conv_bias=conv_bias,
                activation=self.activation,
                num_k_heads=self.num_k_heads // self.tp_size,
                head_k_dim=self.head_k_dim,
                head_v_dim=self.head_v_dim,
                scale=scale,
                tiled_gqa=self.tiled_v_head_layout,
            )
            out_spec.index_copy_(0, tok_idx, o_g.reshape(-1, *o_g.shape[2:]))

        if mixed:
            assert spec_token_indx is not None
            core_attn_out[:num_actual_tokens].index_copy_(
                0,
                spec_token_indx[:num_spec_tokens],
                out_spec.to(core_attn_out.dtype),
            )
        else:
            core_attn_out[:num_spec_tokens] = out_spec.to(core_attn_out.dtype)


import os as _os  # noqa: E402

# Env-gated diagnostic (QWEN38_GDN_DEBUG=1): per step, dump the GDN
# metadata this layer received plus one full-attention layer's metadata
# (block table shape / occupancy / seq_lens) from the same forward context.
_GDN_MPS_DEBUG = _os.environ.get("QWEN38_GDN_DEBUG") == "1"
_GDN_DEBUG_STATE: dict = {"first_prefix": None, "calls": 0}

# Env-gated diagnostic (QWEN38_STATE_PROBE=1): write-vs-readback checksums of
# the persistent conv/ssm state, per layer per step. Any mismatch between the
# state read at step N+1 and the state written at step N means an external
# clobber or a lost/reordered write. Also reports NaN/absmax of the layer's
# inputs so corruption arriving from upstream layers is distinguishable from
# corruption born in this layer's state. Diagnostic only: syncs every step.
_GDN_STATE_PROBE = _os.environ.get("QWEN38_STATE_PROBE") == "1"

# Fused Metal GDN step (kernels/serving_glue/gdn_step.metal). Default on;
# VLLM_QWEN38_FUSED_GDN=0 keeps the torch-native oracle path.
_FUSED_GDN_ENV = _os.environ.get("VLLM_QWEN38_FUSED_GDN", "1") != "0"
# Fused gated RMS norm on the GDN output (same kernel file); separately
# switchable for A/B.
_FUSED_GDN_NORM_ENV = _os.environ.get("VLLM_QWEN38_FUSED_GDN_NORM", "1") != "0"
_FUSED_GDN_AVAILABLE: bool | None = None
_FUSED_GDN_ARANGE: dict = {}


def _fused_gdn_available() -> bool:
    global _FUSED_GDN_AVAILABLE
    if _FUSED_GDN_AVAILABLE is None:
        ok = False
        if current_platform.is_metal():
            try:
                from vllm.quixicore.ops import _qc

                ok = hasattr(_qc(), "qwen_gdn_step")
            except ImportError:
                ok = False
        _FUSED_GDN_AVAILABLE = ok
    return _FUSED_GDN_AVAILABLE


def _fused_gdn_arange(n: int, device: torch.device) -> torch.Tensor:
    key = (n, str(device))
    t = _FUSED_GDN_ARANGE.get(key)
    if t is None:
        t = torch.arange(n, dtype=torch.int32, device=device)
        _FUSED_GDN_ARANGE[key] = t
    return t


def _fused_gdn_decode_plan(attn_metadata: GDNAttentionMetadata, num_decodes: int):
    """Per-step decode index tensors (shared by all GDN layers of the step)."""
    plan = getattr(attn_metadata, "_mps_fused_decode_plan", None)
    if plan is None:
        idx = attn_metadata.non_spec_state_indices_tensor[:num_decodes]  # type: ignore[index]
        idx = idx.to(torch.int32).contiguous()
        plan = (_fused_gdn_arange(num_decodes, idx.device), idx, idx.view(-1, 1))
        attn_metadata._mps_fused_decode_plan = plan  # type: ignore[attr-defined]
    return plan


def _fused_gdn_spec_plans(attn_metadata: GDNAttentionMetadata, num_spec: int):
    """Per-step spec group plans: (N, S, token_map, conv_slot, resume_slot,
    store_slots, num_accepted) per query-length group.

    NULL semantics follow the Triton kernels / torch oracle: a sequence whose
    conv slot (row[0]) or resume slot (row[num_accepted-1]) is NULL is
    skipped entirely (zero output); per-position NULL store slots are
    skipped by the kernel. The uniform full-block case needs no host sync.
    """
    plans = getattr(attn_metadata, "_mps_fused_spec_plans", None)
    if plans is not None:
        return plans
    n = attn_metadata.num_spec_decodes
    slots = attn_metadata.spec_state_indices_tensor[:n]  # type: ignore[index]
    device = slots.device
    slots = slots.to(torch.int32)
    accepted = attn_metadata.num_accepted_tokens[:n].to(  # type: ignore[index]
        device=device, dtype=torch.int32
    )
    resume = slots.gather(1, (accepted - 1).clamp_min(0).to(torch.long).view(-1, 1))
    resume = resume.view(-1)
    conv_slot = slots[:, 0]
    valid = (conv_slot > 0) & (resume > 0)
    zero = torch.zeros_like(conv_slot)
    conv_slot = torch.where(valid, conv_slot, zero).contiguous()
    resume = torch.where(valid, resume, zero).contiguous()
    mixed = attn_metadata.num_prefills > 0
    spec_token_indx = attn_metadata.spec_token_indx
    block = num_spec + 1
    plans = []
    if attn_metadata.num_spec_decode_tokens == n * block:
        # Uniform full blocks: sequence i owns spec rows [i*block, (i+1)*block).
        token_map = _fused_gdn_arange(n * block, device)
        if mixed:
            token_map = spec_token_indx[: n * block].to(torch.int32).contiguous()  # type: ignore[index]
        plans.append((n, block, token_map, conv_slot, resume, slots, accepted))
    else:
        qsl = attn_metadata.spec_query_start_loc[: n + 1].cpu().tolist()  # type: ignore[index]
        groups: dict[int, list[int]] = {}
        for i in range(n):
            s = qsl[i + 1] - qsl[i]
            if s > 0:
                groups.setdefault(s, []).append(i)
        for s, idx_list in groups.items():
            sel = torch.tensor(idx_list, dtype=torch.long, device=device)
            tok = torch.tensor(
                [qsl[i] + t for i in idx_list for t in range(s)],
                dtype=torch.long,
                device=device,
            )
            token_map = (
                (
                    spec_token_indx[tok] if mixed else tok  # type: ignore[index]
                )
                .to(torch.int32)
                .contiguous()
            )
            plans.append(
                (
                    len(idx_list),
                    s,
                    token_map,
                    conv_slot[sel].contiguous(),
                    resume[sel].contiguous(),
                    slots[sel].contiguous(),
                    accepted[sel].contiguous(),
                )
            )
    attn_metadata._mps_fused_spec_plans = plans  # type: ignore[attr-defined]
    return plans


# prefix -> slot -> (conv_sum, ssm_sum, step_tag)
_SPROBE_STORE: dict = {}
_SPROBE_STEP: dict = {"n": 0, "first_prefix": None}


def _sprobe_sums(conv_state, ssm_state, slot, conv_width):
    c = conv_state[slot, :, :conv_width].to(torch.float32)
    s = ssm_state[slot].to(torch.float32)
    return (
        float(c.sum().item()),
        float(s.sum().item()),
        int(torch.isnan(c).sum().item()) + int(torch.isnan(s).sum().item()),
        float(s.abs().max().item()),
    )


def _sprobe_check_and_record(
    prefix, conv_state, ssm_state, slots, conv_width, phase, extras=""
) -> None:
    """Compare the current state of `slots` against the last recorded write."""
    st = _SPROBE_STEP
    if st["first_prefix"] is None:
        st["first_prefix"] = prefix
    if prefix == st["first_prefix"] and phase.endswith("pre"):
        st["n"] += 1
    layer_store = _SPROBE_STORE.setdefault(prefix, {})
    for slot in slots:
        cur = _sprobe_sums(conv_state, ssm_state, slot, conv_width)
        prev = layer_store.get(slot)
        if phase.endswith("pre") and prev is not None:
            dc = abs(cur[0] - prev[0])
            ds = abs(cur[1] - prev[1])
            if dc > 1e-3 or ds > 1e-3 or cur[2] != prev[2]:
                print(
                    f"[SPROBE step {st['n']}] {prefix} slot {slot} CLOBBER "
                    f"{phase}: conv {prev[0]:.6f}->{cur[0]:.6f} "
                    f"ssm {prev[1]:.6f}->{cur[1]:.6f} nan {prev[2]}->{cur[2]} "
                    f"(written at {prev[3]}){extras}",
                    flush=True,
                )
        if cur[2] or cur[3] > 1e4:
            print(
                f"[SPROBE step {st['n']}] {prefix} slot {slot} ANOMALY {phase}: "
                f"nan={cur[2]} ssm_absmax={cur[3]:.3e}{extras}",
                flush=True,
            )
        if phase.endswith("post"):
            layer_store[slot] = (cur[0], cur[1], cur[2], f"{phase}@{st['n']}")


def _gdn_mps_debug_dump(prefix, attn_metadata, attn_metadata_raw) -> None:
    st = _GDN_DEBUG_STATE
    if st["first_prefix"] is None:
        st["first_prefix"] = prefix
    if prefix != st["first_prefix"] or st["calls"] >= 60:
        return
    st["calls"] += 1
    idx = attn_metadata.non_spec_state_indices_tensor
    idx_head = idx[:4].cpu().tolist() if idx is not None else None
    line = (
        f"[GDN-DBG step {st['calls']}] {prefix}: dec={attn_metadata.num_decodes} "
        f"pre={attn_metadata.num_prefills} spec={attn_metadata.num_spec_decodes} "
        f"slots={idx_head}"
    )
    for key, meta in attn_metadata_raw.items():
        if isinstance(meta, GDNAttentionMetadata) or meta is attn_metadata:
            continue
        bt = getattr(meta, "block_table", None)
        if bt is None:
            bt = getattr(meta, "block_table_tensor", None)
        sl = getattr(meta, "seq_lens_cpu", None)
        if sl is None:
            sl = getattr(meta, "seq_lens", None)
        bt_desc = None
        if bt is not None:
            row0 = bt[0]
            bt_desc = (
                f"shape={tuple(bt.shape)} row0_nonzero="
                f"{int((row0 != 0).sum())} row0_head={row0[:6].cpu().tolist()}"
            )
        sl_desc = sl[:4].cpu().tolist() if sl is not None else None
        line += f" | {key}: bt {bt_desc} seq_lens={sl_desc}"
        break
    print(line, flush=True)


def _causal_conv1d_native(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native causal depthwise conv1d with persistent state (fp32).

    Matches the semantics of ``causal_conv1d_fn`` / ``causal_conv1d_update``
    (vllm/model_executor/layers/mamba/ops/causal_conv1d.py): the state holds
    the last ``width - 1`` inputs in chronological order (newest last) and

        out[:, :, t] = act(bias + sum_w weight[:, w] * padded[:, :, t + w])

    with ``padded = cat([initial_state, x], dim=-1)``.

    Args:
        x: (N, dim, T) fp32 inputs.
        initial_state: (N, dim, width - 1) fp32 prior inputs (zeros when a
            sequence has no initial state).
        weight: (dim, width) fp32.
        bias: (dim,) fp32 or None.
        activation: None or "silu"/"swish".

    Returns:
        out: (N, dim, T) fp32, final_state: (N, dim, width - 1) fp32.
    """
    width = weight.size(-1)
    seq_len = x.size(-1)
    padded = torch.cat([initial_state, x], dim=-1)
    out = torch.zeros_like(x)
    for w in range(width):
        out += weight[:, w : w + 1] * padded[..., w : w + seq_len]
    if bias is not None:
        out += bias[:, None]
    if activation is not None:
        out = nn.functional.silu(out)
    return out, padded[..., seq_len:]


def _gdn_recurrent_scan_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    tiled_gqa: bool = False,
    output_all_states: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native gated delta rule recurrence (fp32).

    Port of ``fused_recurrent_gated_delta_rule_fwd_kernel``
    (vllm/third_party/flash_linear_attention/ops/fused_recurrent.py) with
    ``USE_QK_L2NORM_IN_KERNEL=True`` semantics; the sequential loop is over
    tokens only, batched across sequences and heads.

    Args:
        q, k: (N, T, H, K) shared q/k heads (head hv uses h = hv // (HV//H)).
        v: (N, T, HV, V).
        g: (N, T, HV) fp32 log-decay.
        beta: (N, T, HV) fp32 write strength.
        scale: query scale (head_k_dim ** -0.5).
        initial_state: (N, HV, V, K) fp32; not mutated.
        output_all_states: when True, the second return value is the running
            state after EVERY position, shape (N, T, HV, V, K) — the
            per-position stores required by spec decoding
            (INPLACE_FINAL_STATE in the Triton kernel).

    Returns:
        o: (N, T, HV, V) fp32, final_state: (N, HV, V, K) fp32
        (or (N, T, HV, V, K) fp32 when ``output_all_states``).
    """
    num_seqs, seq_len, num_k_heads, _ = q.shape
    num_v_heads = v.shape[2]
    rep = num_v_heads // num_k_heads

    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Per-head l2norm along K, then query scaling (b_q/b_k normalization
    # followed by b_q *= scale in the Triton kernel).
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)

    # Expand shared q/k heads to v-head granularity.
    if tiled_gqa:
        # ggml tiled broadcast (llama.cpp-converted GGUF weights, where the
        # per-V-head tensors are stored v-outer/k-inner): i_h = i_hv % H.
        q = q.repeat(1, 1, rep, 1)
        k = k.repeat(1, 1, rep, 1)
    else:
        # HF grouped layout: i_h = i_hv // (HV // H).
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)

    decay = g.exp()

    state = initial_state.to(torch.float32).clone()  # (N, HV, V, K)
    o = torch.empty(
        (num_seqs, seq_len, num_v_heads, v.shape[3]),
        dtype=torch.float32,
        device=q.device,
    )
    all_states: list[torch.Tensor] | None = [] if output_all_states else None
    for t in range(seq_len):
        k_t = k[:, t]  # (N, HV, K)
        state = state * decay[:, t, :, None, None]
        # Delta rule: v_t -= S @ k_t; v_t *= beta_t; S += v_t ⊗ k_t.
        v_t = v[:, t] - torch.einsum("nhvk,nhk->nhv", state, k_t)
        v_t = v_t * beta[:, t, :, None]
        state = state + v_t.unsqueeze(-1) * k_t.unsqueeze(-2)
        o[:, t] = torch.einsum("nhvk,nhk->nhv", state, q[:, t])
        if all_states is not None:
            all_states.append(state)
    if all_states is not None:
        return o, torch.stack(all_states, dim=1)  # (N, T, HV, V, K)
    return o, state


def _gdn_spec_state_step_native(
    x: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    slot_rows: torch.Tensor,
    num_accepted: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_bias: torch.Tensor | None,
    activation: str | None,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    scale: float,
    tiled_gqa: bool,
) -> torch.Tensor:
    """One spec-decode (multi-query) GDN step with state rollback (fp32).

    Torch-native port of the Triton spec-decoding pair:

    - ``causal_conv1d_update`` with ``IS_SPEC_DECODING`` (ops/causal_conv1d.py):
      the conv slot ``slot_rows[i, 0]`` holds a rolling window; with
      ``off = num_accepted[i] - 1`` the initial window is
      ``state[:, off : off + width - 1]`` and after the step the slot is
      rewritten (from column 0) as
      ``[state[:, off + 1 : off + width - 1], x_i]`` — rejected draft inputs
      from the previous step are thereby dropped.
    - ``fused_recurrent_gated_delta_rule_fwd_kernel`` with
      ``IS_SPEC_DECODING + INPLACE_FINAL_STATE`` (ops/fused_recurrent.py):
      the SSM state resumes from slot ``slot_rows[i, num_accepted[i] - 1]``
      and the running state after EVERY position t is stored to slot
      ``slot_rows[i, t]``; slots <= 0 (NULL_BLOCK_ID) skip the store.

    All sequences in the batch must share the same query length ``s`` (the
    caller groups by length) and must have valid (> 0) conv and resume
    slots; per-position NULL slots are still honored on store.

    Args:
        x: (G, s, conv_dim) pre-conv spec tokens (chronological).
        g, beta: (G, s, HV) fp32 gating terms.
        conv_state: (num_slots, conv_dim, L) mutated in place,
            L >= width - 1 + s - 1.
        ssm_state: (num_slots, HV, V, K) mutated in place.
        slot_rows: (G, >= s) long per-position state slots.
        num_accepted: (G,) long accepted-token counts from the previous step.
        conv_weights: (conv_dim, width) fp32; conv_bias: (conv_dim,) or None.

    Returns:
        o: (G, s, HV, V) fp32 core attention outputs.
    """
    num_groups, s, conv_dim = x.shape
    width = conv_weights.size(-1)
    device = x.device

    x_t = x.to(torch.float32).transpose(1, 2)  # (G, conv_dim, s)
    conv_slots = slot_rows[:, 0]
    state_rows = conv_state[conv_slots].to(torch.float32)  # (G, conv_dim, L)
    off = (num_accepted - 1).view(-1, 1, 1)

    # Initial conv window: state[:, off : off + width - 1].
    win_idx = off + torch.arange(width - 1, device=device).view(1, 1, -1)
    conv_init = state_rows.gather(-1, win_idx.expand(num_groups, conv_dim, -1))
    conv_out, _ = _causal_conv1d_native(
        x_t, conv_init, conv_weights, conv_bias, activation
    )

    # Rolled conv state: [state[:, off+1 : off+width-1], x] from column 0.
    if width > 2:
        carry_idx = off + 1 + torch.arange(width - 2, device=device).view(1, 1, -1)
        carry = state_rows.gather(-1, carry_idx.expand(num_groups, conv_dim, -1))
        new_state = torch.cat([carry, x_t], dim=-1)
    else:
        new_state = x_t
    conv_state[conv_slots, :, : width - 2 + s] = new_state.to(conv_state.dtype)

    # Recurrent scan resuming from the last accepted position's SSM slot.
    key_dim = num_k_heads * head_k_dim
    tokens = conv_out.transpose(1, 2)  # (G, s, conv_dim)
    q_g, k_g, v_g = torch.split(
        tokens, [key_dim, key_dim, conv_dim - 2 * key_dim], dim=-1
    )
    q_g = q_g.reshape(num_groups, s, -1, head_k_dim)
    k_g = k_g.reshape(num_groups, s, -1, head_k_dim)
    v_g = v_g.reshape(num_groups, s, -1, head_v_dim)

    resume_slots = slot_rows.gather(1, (num_accepted - 1).view(-1, 1)).squeeze(1)
    ssm_init = ssm_state[resume_slots].to(torch.float32)
    o_g, states_all = _gdn_recurrent_scan_native(
        q_g,
        k_g,
        v_g,
        g.to(torch.float32),
        beta.to(torch.float32),
        scale,
        ssm_init,
        tiled_gqa=tiled_gqa,
        output_all_states=True,
    )

    # Store the running state after every position to its slot; NULL slots
    # (<= 0) are skipped, matching the Triton kernel.
    pos_slots = slot_rows[:, :s].reshape(-1)
    valid = pos_slots > 0
    if bool(valid.any()):
        flat_states = states_all.reshape(-1, *states_all.shape[2:])
        ssm_state[pos_slots[valid]] = flat_states[valid].to(ssm_state.dtype)
    return o_g


def qwen_gdn_attention_core(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Custom op dispatching to _forward_core or _forward_core_rocm.

    Handles conv1d + recurrent attention only; input/output projections
    are performed by the caller.

    When ``use_aiter=False`` (standard path):
        qkv_or_qkvz is [q, k, v], b_or_ba is b, a_or_z_out is a (read-only).
    When ``use_aiter=True`` (AITER Triton path, ROCm only):
        qkv_or_qkvz is [q, k, v, z], b_or_ba is [b, a], a_or_z_out is the
        z output buffer (mutated in-place).

    ``core_attn_out`` is always mutated in-place.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if use_aiter:
        self._forward_core_rocm(
            qkvz=qkv_or_qkvz,
            ba=b_or_ba,
            z_out=a_or_z_out,
            core_attn_out=core_attn_out,
        )
    else:
        self._forward_core(
            mixed_qkv=qkv_or_qkvz,
            b=b_or_ba,
            a=a_or_z_out,
            core_attn_out=core_attn_out,
        )


def gdn_attention_core_fake(
    qkv_or_qkvz: torch.Tensor,
    b_or_ba: torch.Tensor,
    a_or_z_out: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
    use_aiter: bool = False,
) -> None:
    """Fake implementation for torch.compile."""
    return


direct_register_custom_op(
    op_name="qwen_gdn_attention_core",
    op_func=qwen_gdn_attention_core,
    mutates_args=["a_or_z_out", "core_attn_out"],
    fake_impl=gdn_attention_core_fake,
)


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    # compute beta_output = sigmoid(b)
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(
        beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask
    )


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output
