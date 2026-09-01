# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (``glm5_next``).

Hybrid text backbone: 34 KDA (Kimi Delta Attention) linear-attention layers
and 11 DeepSeek-sparse-attention full layers (NoPE MLA: ``qk_rope_head_dim
== 0``), joined by mHC (manifold-constrained hyper-connections, ``hc_mult
4``) and a 288-expert sigmoid/noaux_tc MoE with one shared expert. The
vision half is a 24-depth SwiGLU-limit ViT with a 2x2 patch merger.

Composition sources, all already serving in this fork:
- KDA: the shared ``KimiGatedDeltaNetAttention`` layer (Kimi-K3 lineage;
  GLM-5.3's ForgetGate/conv/beta math is identical).
- NoPE MLA: the ``MultiHeadLatentAttentionWrapper`` NoPE path Kimi-K3 uses.
- MoE routing: ``DeepseekV2MoE`` (config field names match verbatim).
- mHC: eager implementation of the reference math (DSV4's fused tilelang
  mHC kernels are the planned optimization; the math is identical, DSV4's
  ``hc_post_alpha=2.0`` == the reference's ``2*sigmoid``).

Bring-up status (perf notebook 2026-09-01): full-attention layers currently
run DENSE NoPE MLA - exact for contexts <= index_topk (2048) and a
diagnostic approximation beyond; the pooled DSA indexer (4-token pools,
learned APE + gate compression) is required before any profile ships.
The MTP head (layer 45) and video inputs are later phases.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mhc import (
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreOp,
)
from vllm.model_executor.layers.mla import (
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA for the DeepSeek-sparse-attention layers.

    Currently dense (see module docstring); the pooled indexer will hang
    off this module and flip is_sparse when it lands.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_local_heads = self.num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5

        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
            disable_tp=True,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank, eps=config.rms_norm_eps
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=None,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj,
            kv_a_proj_with_mqa=None,
            q_a_layernorm=self.q_a_layernorm,
            q_b_proj=self.q_b_proj,
            q_proj=None,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states)


class Glm5NextDecoderLayer(nn.Module):
    """One hybrid layer with mHC at both sites.

    Uses the fork's fused mHC ops (quixicore CUDA kernels on Ampere, the
    same path DSV4 serves with): the first layer runs ``MHCPreOp`` on the
    expanded streams, every later site runs ``MHCFusedPostPreOp`` which
    applies the previous sublayer's post/comb placement and the next
    site's pre in one launch. The layer returns its FFN output with the
    placement deferred to the next layer (or the model's final
    ``MHCPostOp``). hc parameters are flat on the layer, matching the
    checkpoint's ``hc_{attn,ffn}_{fn,base,scale}`` names.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        quant_config = vllm_config.quant_config
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.is_linear = (
            config.layer_types[self.layer_idx] == "linear_attention"
        )

        if self.is_linear:
            self.self_attn = KimiGatedDeltaNetAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                config, vllm_config, prefix=f"{prefix}.self_attn"
            )

        if config.mlp_layer_types[self.layer_idx] == "sparse":
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=vllm_config.parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(self.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            self.hidden_size, config.rms_norm_eps
        )

        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0  # reference: post = 2 * sigmoid(.)
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size

        def _p(*shape):
            return nn.Parameter(
                torch.empty(*shape, dtype=torch.float32), requires_grad=False
            )

        self.hc_attn_fn = _p(mix_hc, hc_dim)
        self.hc_ffn_fn = _p(mix_hc, hc_dim)
        self.hc_attn_base = _p(mix_hc)
        self.hc_ffn_base = _p(mix_hc)
        self.hc_attn_scale = _p(3)
        self.hc_ffn_scale = _p(3)

        self.mhc_pre = MHCPreOp()
        self.mhc_post = MHCPostOp()
        self.mhc_fused_post_pre = MHCFusedPostPreOp()

    def _site_pre(self, residual, fn, scale, base):
        post_mix, res_mix, x = self.mhc_pre(
            residual=residual,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=self.rms_norm_eps,
            hc_pre_eps=self.hc_eps,
            hc_sinkhorn_eps=self.hc_eps,
            hc_post_mult_value=self.hc_post_alpha,
            sinkhorn_repeat=self.hc_sinkhorn_iters,
            norm_weight=None,
            norm_eps=0.0,
        )
        return residual, post_mix, res_mix, x

    def _site_fused(self, x, residual, post_mix, res_mix, fn, scale, base):
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

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: x is the expanded [T, hc_mult, D] stream tensor.
            residual, post_mix, res_mix, x = self._site_pre(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        else:
            residual, post_mix, res_mix, x = self._site_fused(
                x, residual, post_mix, res_mix,
                self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            )
        x = self.input_layernorm(x)
        if self.is_linear:
            attn_out = torch.empty_like(x)
            self.self_attn(x, positions, attn_out)
        else:
            attn_out = self.self_attn(positions, x)

        residual, post_mix, res_mix, x = self._site_fused(
            attn_out, residual, post_mix, res_mix,
            self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
        )
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return x, residual, post_mix, res_mix


class Glm5NextTextModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(
                config, vllm_config, prefix=prefix
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        # Expand to hc_mult parallel residual streams: [T, H, D].
        x = hidden_states.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
        residual = post_mix = res_mix = None
        layer = None
        for layer in self.layers[self.start_layer:self.end_layer]:
            x, residual, post_mix, res_mix = layer(
                x, positions, residual, post_mix, res_mix
            )
        assert layer is not None
        streams = layer.mhc_post(x, residual, post_mix, res_mix)
        # HyperHead: unweighted mean over the streams (reference; DSV4's
        # weighted head does not apply).
        hidden_states = streams.mean(dim=1)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Glm5NextForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsPP):
    """Text-only serving entry for GLM-5.3-Flash (phase 1).

    The checkpoint's ``model.visual.*`` and MTP (layer 45) tensors are
    skipped at load; the multimodal wrapper lands with the vision phase.
    """

    hf_to_vllm_prefix = {
        "model.language_model.": "model.",
        "lm_head.": "lm_head.",
    }

    # fused/stacked parameter mappings: (target, checkpoint_shard, shard_id)
    stacked_params_mapping = [
        # MLA latent projections
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        # KDA merged input projection: [q, k, v, b(beta), f_a]
        ("in_proj_qkvgfab", "q_proj", 0),
        ("in_proj_qkvgfab", "k_proj", 1),
        ("in_proj_qkvgfab", "v_proj", 2),
        ("in_proj_qkvgfab", "b_proj", 3),
        ("in_proj_qkvgfab", "f_a_proj", 4),
        # KDA fused conv over [q, k, v]
        ("conv1d", "q_conv1d", 0),
        ("conv1d", "k_conv1d", 1),
        ("conv1d", "v_conv1d", 2),
        # Dense MLP / shared experts
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        self.config = config
        self.model = Glm5NextTextModel(
            vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            tp_size,
            hf_config.linear_attn_config["num_heads"],
            hf_config.linear_attn_config["head_dim"],
            conv_kernel_size=hf_config.linear_attn_config[
                "short_conv_kernel_size"
            ],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        num_text_layers = self.config.num_hidden_layers
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            # Vision tower and MTP layer: later phases.
            if name.startswith("model.visual."):
                continue
            if f"layers.{num_text_layers}." in name:
                continue
            for pref, new in self.hf_to_vllm_prefix.items():
                if name.startswith(pref):
                    name = new + name[len(pref):]
                    break
            else:
                continue
            # Indexer weights: retained by the sparse phase; skipped while
            # the DSA layers run dense.
            if ".self_attn.indexer." in name:
                continue

            is_expert = ".mlp.experts." in name
            mapped = False
            if not is_expert:
                for target, ckpt_name, shard_id in self.stacked_params_mapping:
                    token = f".{ckpt_name}."
                    if token not in name and not name.endswith(
                        f".{ckpt_name}"
                    ):
                        continue
                    tgt = name.replace(ckpt_name, target)
                    if tgt not in params_dict:
                        continue
                    param = params_dict[tgt]
                    param.weight_loader(param, weight, shard_id)
                    loaded.add(tgt)
                    mapped = True
                    break
            else:
                for (
                    param_name,
                    ckpt_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if ckpt_name not in name:
                        continue
                    tgt = name.replace(ckpt_name, param_name)
                    if is_pp_missing_parameter(tgt, self):
                        continue
                    if tgt not in params_dict:
                        continue
                    param = params_dict[tgt]
                    param.weight_loader(
                        param,
                        weight,
                        tgt,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded.add(tgt)
                    mapped = True
                    break
            if mapped:
                continue
            if name not in params_dict:
                logger.warning_once("glm5_next: unmatched weight %s", name)
                continue
            param = params_dict[name]
            loader = getattr(
                param, "weight_loader", None
            )
            if loader is not None:
                loader(param, weight)
            else:
                param.data.copy_(weight)
            loaded.add(name)
        return loaded
