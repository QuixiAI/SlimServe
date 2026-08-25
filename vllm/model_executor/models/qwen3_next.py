# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model (dense subset).

Vendored from upstream vLLM for the Qwen3.5 bring-up. This fork keeps only
what the dense Qwen3.5 chain (models/qwen3_5.py) subclasses:
Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextModel, Qwen3NextMLP,
and the two module-level helpers. Trimmed relative to upstream:

- Qwen3NextSparseMoeBlock / QwenNextMixtureOfExperts / Qwen3NextForCausalLM:
  the fork does not ship upstream's FusedMoEFactory, so the MoE variants are
  unsupported and the decoder layer raises if a MoE layer is requested.
- Qwen3NextMLP is upstream's qwen2_moe.Qwen2MoeMLP inlined verbatim
  (qwen2_moe.py is not vendored; its other contents are MoE-only).
- fused_qk_rmsnorm_rope_gate is imported lazily inside the CUDA-only fused
  path (layers/fused_qk_norm_rope.py is not vendored; the flag guarding it
  is False off-CUDA, so Metal always takes the eager path).
"""

import os
from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.metal_phaseprof import phase as _qc_phase

from .interfaces import EagleModelMixin
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
)

logger = init_logger(__name__)


# Muse single-CB decode step (N14, muse_q38_metal.py): opt-in during
# bringup; the serve-mode flip re-pins the canonical trajectory.
_MUSE_ENABLED = os.environ.get("VLLM_QC_MUSE", "0") in ("1", "shadow")

_QC_QK_ROPE_AVAILABLE: bool | None = None


def _qc_qk_rope_kernel_available() -> bool:
    global _QC_QK_ROPE_AVAILABLE
    if _QC_QK_ROPE_AVAILABLE is None:
        try:
            from vllm.quixicore import quixicore_ops

            if quixicore_ops.is_available():
                import vllm._quixicore_C as qc

                _QC_QK_ROPE_AVAILABLE = hasattr(qc, "qc_qk_norm_rope_gate")
            else:
                _QC_QK_ROPE_AVAILABLE = False
        except ImportError:
            _QC_QK_ROPE_AVAILABLE = False
    return _QC_QK_ROPE_AVAILABLE


def _should_use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    config = vllm_config.model_config.hf_text_config
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.use_sequence_parallel_moe
        and parallel_config.pipeline_parallel_size == 1
        and getattr(config, "num_experts", 0) > 0
        and not getattr(config, "mlp_only_layers", [])
        and getattr(config, "decoder_sparse_step", 1) == 1
    )


class Qwen3NextMLP(nn.Module):
    # Upstream: qwen2_moe.Qwen2MoeMLP (inlined; see module docstring).
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
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
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        return out


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        # Late-interaction retrieval models (e.g. ColQwen3.5) run BIDIRECTIONAL
        # attention on the full_attention layers; they set config.is_causal=False
        # via a VerifyAndUpdateConfig handler. Generation models leave is_causal
        # unset (-> causal/DECODER), so this is a no-op for them. Mirrors qwen3.py.
        attn_type = (
            AttentionType.DECODER
            if getattr(config, "is_causal", True)
            else AttentionType.ENCODER_ONLY
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Fuse the gated split + QK-RMSNorm + (partial) NeoX RoPE + gate copy.
        # TODO: support MRoPE
        mm_config = model_config.multimodal_config if model_config else None
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )
        # Metal twin of the fusion above (qc_qk_norm_rope_gate kernel).
        # Rejected as a default: bit-exact at decode shapes but
        # torch-MPS eager numerics are SIZE-DEPENDENT — large prefill
        # tensors round the rotation chain differently (~5 ppm single-ulp
        # diffs), so the canonical trajectory forks at prefill while the
        # measured win was c4/c8 ~+1% and c1 ~flat-to-negative — not worth
        # the full sha re-pin. Opt-in diagnostic: VLLM_QC_QKROPE=1.
        self.use_fused_qk_rope_metal = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_metal()
            and text_only
            and os.environ.get("VLLM_QC_QKROPE", "0") == "1"
            and _qc_qk_rope_kernel_available()
        )

    def _project_qkv_gate(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return post-norm, post-RoPE (q, k, v) and the pre-sigmoid gate.

        Dispatches between the fused Triton kernel and the eager
        split + QK-RMSNorm + RoPE path. ``gate`` is ``None`` when output
        gating is disabled.
        """
        if self.use_fused_qk_norm_rope_gate:
            # CUDA-only path; the kernel module is not vendored in this fork,
            # so import at use (the guard above is False off-CUDA).
            from vllm.model_executor.layers.fused_qk_norm_rope import (
                fused_qk_rmsnorm_rope_gate,
            )

            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            # mRoPE passes positions as (3, n_tokens) for T/H/W. Fusion is only
            # enabled text-only, where the three rows are identical, so taking
            # the T row is exact. (1D positions pass through.)
            pos = positions[0] if positions.ndim == 2 else positions
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.weight.float() + 1.0,
                self.k_norm.weight.float() + 1.0,
                self.rotary_emb.cos_sin_cache,
                pos,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
            return q, k, v, gate

        if (
            self.use_fused_qk_rope_metal
            # bf16 only: the eager reference is the bf16-gated gemma-norm
            # custom kernel; fp16 falls back to the native ir chain, whose
            # numerics this kernel does not mirror (parity harness showed
            # ulp-level diffs there).
            and qkv.dtype is torch.bfloat16
            and qkv.dim() == 2
            and qkv.is_contiguous()
            and self.q_norm.weight.dtype == qkv.dtype
        ):
            from vllm.quixicore import quixicore_ops

            # Text-only: the three mRoPE rows are identical; the T row is
            # exact (same argument as the CUDA fusion above).
            pos = positions[0] if positions.ndim == 2 else positions
            cache = self.rotary_emb._match_cos_sin_cache_dtype(qkv)
            q, gate, k = quixicore_ops.qc_qk_norm_rope_gate(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cache,
                pos.contiguous(),
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
                self.q_norm.variance_epsilon,
            )
            v = qkv[:, self.q_size * 2 + self.kv_size :]
            return q, k, v, gate

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        attn_output = self.attn(q, k, v)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        is_moe_layer = (self.layer_idx not in mlp_only_layers) and (
            getattr(config, "num_experts", 0) > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        )
        self.use_attn_reduce_scatter_for_moe = _should_use_sequence_parallel(
            vllm_config
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                reduce_results=not self.use_attn_reduce_scatter_for_moe,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        if is_moe_layer:
            raise NotImplementedError(
                "Qwen3-Next MoE layers are not supported by this fork "
                "(Qwen3NextSparseMoeBlock is not vendored); only the dense "
                "Qwen3.5 chain is served."
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs: object,
    ):
        full_num_tokens = positions.shape[-1]

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[:full_num_tokens]

        if self.layer_type == "linear_attention":
            with _qc_phase("gdn_attn"):
                hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            with _qc_phase("full_attn"):
                hidden_states = self.self_attn(
                    hidden_states=hidden_states,
                    positions=positions,
                )
        else:
            raise ValueError("Invalid layer_type")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        if self.use_attn_reduce_scatter_for_moe:
            tp_world_size = get_tensor_model_parallel_world_size()
            # small trick using minus, eg. -17 % 8 = 7
            sp_pad = (-hidden_states.shape[0]) % tp_world_size
            # pad if not divisible by world size
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
            hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # use_attn_reduce_scatter_for_moe requires num_experts > 0, which
        # __init__ rejects — the dense Qwen3NextMLP takes no SP kwarg.
        with _qc_phase("mlp"):
            hidden_states = self.mlp(hidden_states)

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            # weight_name: (param_name, shard_id)
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def use_sequence_parallel(self) -> bool:
        return self.layers[self.start_layer].use_attn_reduce_scatter_for_moe

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        full_num_tokens = positions.shape[-1]
        if self.use_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None

        # Muse single-CB decode step (N14): on eligible uniform pure-spec
        # decode batches the whole 64-layer forward + final norm is encoded
        # into one command buffer, including the drafter's aux hidden-state
        # taps. VLLM_QC_MUSE=1 serves it, =shadow runs both and compares
        # (eager serves). Trajectory re-pin accepted for this path — see
        # csrc/quixicore/metal/muse_qwen38_design.md.
        if (
            _MUSE_ENABLED
            and residual is None
            and not self.use_sequence_parallel
            and get_pp_group().is_last_rank
        ):
            from vllm.model_executor.models import muse_q38_metal

            muse_out = muse_q38_metal.try_step(self, hidden_states, positions)
            if muse_out is not None:
                return muse_out

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0),
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
