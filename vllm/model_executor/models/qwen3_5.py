# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights.

Ported from the lazarus vLLM fork onto the SlimServe base.

SlimServe pruned the Qwen VL stack (qwen3_vl / qwen2_5_vl / qwen2_vl), so the
`*ForConditionalGeneration` classes here are TEXT-ONLY adaptations: they keep
the checkpoint architecture name and weight layout (model.language_model.*,
lm_head.*) but do not instantiate the vision tower; `model.visual.*` weights
are skipped at load time. Multimodal (image/video) inputs are NOT supported.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from .interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
    _is_shared_expert_fse_compatible,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

# llama.cpp's GGUF converter (conversion/qwen.py) folds +1 into every
# `*norm.weight` except `linear_attn.norm.weight`, so that plain ggml
# RMS_NORM * w reproduces HF's zero-centered Qwen3.5 norms (x_hat * (1 + w)).
# This model uses GemmaRMSNorm, which re-adds the +1 at runtime, so
# GGUF-sourced norm weights must be shifted back to the zero-centered form
# at load. `linear_attn.norm.weight` (RMSNormGated, x_hat * w) is stored
# unshifted in the GGUF and passes through unchanged.
_GGUF_PLUS_ONE_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "model.norm.weight",
)


def _degemma_gguf_norms(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Shift llama.cpp-convention GGUF norm weights back to zero-centered."""
    for name, weight in weights:
        if name.endswith(_GGUF_PLUS_ONE_NORM_SUFFIXES):
            weight = (weight.to(torch.float32) - 1.0).to(weight.dtype)
        yield name, weight


# --- BEGIN TEMPORARY DIAGNOSTIC (Qwen3.8 forward-divergence hunt) ----------
# Env-gated activation dump used to compare our forward pass against
# llama.cpp's eval-callback output, layer by layer, for one fixed prompt.
# Zero-cost when VLLM_QWEN38_DEBUG_DUMP is unset. Remove after the
# investigation (tracked in perf/optimization_status.md).
import os as _os  # noqa: E402

_QWEN38_DUMP_DIR = _os.environ.get("VLLM_QWEN38_DEBUG_DUMP")
# Prompt prefix that arms the dump (the ESSAY probe prompt).
_QWEN38_DUMP_PREFIX = [760, 2250, 1379]
_QWEN38_DUMP_LEN = 11


class _Qwen38DumpState:
    """Capture per-layer activations on the first forward of the probe
    prompt, then write a single .npz and disarm."""

    def __init__(self, dump_dir: str, model: "Qwen3_5Model") -> None:
        self.dump_dir = dump_dir
        self.armed = False
        self.done = False
        self.seen = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self._install(model)
        print(
            f"[qwen38-dump] installed hooks on {len(model.layers)} layers, "
            f"dir={dump_dir}",
            flush=True,
        )

    def _save(self, name: str, t: torch.Tensor | None) -> None:
        if self.armed and t is not None:
            self.tensors[name] = t.detach().to(torch.float32).cpu().clone()

    def _install(self, model: "Qwen3_5Model") -> None:
        def embed_hook(mod, args, out):
            if self.done:
                return
            input_ids = args[0] if args else None
            if input_ids is None:
                return
            ids = input_ids.flatten().tolist()
            if self.seen < 8:
                self.seen += 1
                print(
                    f"[qwen38-dump] embed #{self.seen}: n={len(ids)} head={ids[:4]}",
                    flush=True,
                )
            n = _QWEN38_DUMP_LEN
            if len(ids) >= n and ids[: len(_QWEN38_DUMP_PREFIX)] == _QWEN38_DUMP_PREFIX:
                self.armed = True
                self._save("input_ids", input_ids)
                self._save("emb", out)

        model.embed_tokens.register_forward_hook(embed_hook)

        for idx, layer in enumerate(model.layers):
            inner = getattr(layer, "linear_attn", None)
            if inner is None:
                inner = getattr(layer, "self_attn", None)

            def attn_hook(mod, args, kwargs, out, idx=idx):
                if not self.armed:
                    return
                hs = kwargs.get("hidden_states", args[0] if args else None)
                self._save(f"L{idx}.attn_in", hs)
                self._save(f"L{idx}.attn_out", out)

            inner.register_forward_hook(attn_hook, with_kwargs=True)

            def layer_hook(mod, args, kwargs, out, idx=idx):
                if not self.armed:
                    return
                hidden, residual = out
                # Equivalent of llama.cpp's l_out: the residual stream after
                # the FFN add (our fused add happens in the next norm).
                self._save(f"L{idx}.out", hidden + residual)

            layer.register_forward_hook(layer_hook, with_kwargs=True)

        def final_norm_hook(mod, args, kwargs, out):
            if not self.armed or self.done:
                return
            final = out[0] if isinstance(out, tuple) else out
            self._save("final_norm", final)
            import numpy as np

            _os.makedirs(self.dump_dir, exist_ok=True)
            path = _os.path.join(self.dump_dir, "our_forward.npz")
            np.savez(path, **{k: v.numpy() for k, v in self.tensors.items()})
            print(
                f"[qwen38-dump] wrote {len(self.tensors)} tensors to {path}",
                flush=True,
            )
            self.armed = False
            self.done = True
            self.tensors.clear()

        model.norm.register_forward_hook(final_norm_hook, with_kwargs=True)


# --- END TEMPORARY DIAGNOSTIC -----------------------------------------------


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
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


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    # Qwen3.5 ships the GDN in_proj checkpoints separately (qwen3-next
    # pre-fuses them); fuse them on top of the qwen3-next QKV/gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        self.quant_config = vllm_config.quant_config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3_5DecoderLayer(
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
            self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.aux_hidden_state_layers: tuple[int, ...] = ()

        # TEMPORARY DIAGNOSTIC: see _Qwen38DumpState above.
        if _QWEN38_DUMP_DIR:
            self._qwen38_dump = _Qwen38DumpState(_QWEN38_DUMP_DIR, self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper
        # FSE must match construction (Qwen3NextSparseMoeBlock): reroute the
        # shared expert into the extra fused slot only when AITER FSE is both
        # requested and compatible with the quant spec.
        is_fse = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled() and (
            _is_shared_expert_fse_compatible(self.quant_config)
        )
        if is_fse:
            num_routed = self.config.num_experts
            mapper = mapper | WeightsMapper(
                orig_to_new_substr={"mlp.shared_expert.": f"mlp.experts.{num_routed}."}
            )
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections.
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3_5Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self.model_config.quantization == "gguf":
            weights = _degemma_gguf_norms(weights)
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase, QwenNextMixtureOfExperts):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # set MoE hyperparameters
        self.set_moe_parameters()


########################################################
# Qwen3_5-Dense (text-only wrapper over the VL checkpoint layout)
########################################################


class Qwen3_5ForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
):
    """Text-only adaptation of the Qwen3.5 checkpoint architecture.

    The upstream (lazarus) implementation wraps Qwen3VLForConditionalGeneration
    and instantiates a vision tower; SlimServe has no Qwen VL stack, so this
    class serves the language model only. `model.visual.*` and `mtp.*`
    checkpoint weights are skipped.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    # Checkpoint layout: model.language_model.*, model.visual.*, lm_head.*
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config = vllm_config.model_config.hf_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config

        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and not multimodal_config.language_model_only:
            logger.warning_once(
                "Qwen3_5ForConditionalGeneration on SlimServe is text-only: "
                "the vision tower is not instantiated and image/video inputs "
                "are not supported. `model.visual.*` weights are skipped."
            )

        self.language_model = Qwen3_5ForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            # "visual." is the post-mapper name of "model.visual.".
            skip_prefixes=["mtp.", "visual.", "model.visual."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


########################################################
# Qwen3_5-MoE
########################################################


class Qwen3_5_MoeMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen3_5DecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen3_5 layer found in the language_model.model.layers."
            )

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3_5MoeForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, Qwen3_5_MoeMixtureOfExperts
):
    # For MoE LoRA weights loading
    is_3d_moe_weight: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config

        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is not None and not multimodal_config.language_model_only:
            logger.warning_once(
                "Qwen3_5MoeForConditionalGeneration on SlimServe is text-only: "
                "the vision tower is not instantiated and image/video inputs "
                "are not supported. `model.visual.*` weights are skipped."
            )

        self.language_model = Qwen3_5MoeForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # set MoE hyperparameters
        self.set_moe_parameters()
