# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5 Series compatible with HuggingFace weights.

Vendored from upstream vLLM ``qwen3_5.py`` and trimmed to the dense
text-only chain for this fork: ``Qwen3_5DecoderLayer``, ``Qwen3_5Model``,
``Qwen3_5ForCausalLMBase``, and ``Qwen3_5ForCausalLM`` are kept. The MoE
classes (``Qwen3_5MoeForCausalLM`` and friends) and the multimodal
``Qwen3_5ForConditionalGeneration`` stack are removed; see the notes at the
bottom of this file.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
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

from .interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
)
from .qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextModel,
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
            # NOTE(slimserve): deferred import, same reason as in
            # Qwen3NextDecoderLayer: the GDN stack currently fails to import
            # because vllm.model_executor.layers.mamba.mamba_mixer2
            # (mamba_v2_sharded_weight_loader) is missing in this fork.
            from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
                QwenGatedDeltaNetAttention,
            )

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

        # NOTE: Determine the MLP type based on the model type.
        # Qwen3.5 uses a dense MLP on every layer; Qwen3.5-MoE
        # (model_type == "qwen3_5_moe_text") uses sparse MoE blocks upstream
        # and is intentionally not vendored in this dense-only fork.
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(
                f"Invalid model_type {config.model_type} "
                "(only dense qwen3_5_text is supported in this fork)"
            )

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
            # Per-shard q/k/v names for sources that cannot split a fused
            # tensor (GGUF: quantized bytes split only on row boundaries,
            # so the adapter slices and emits these). The `_shard` suffix
            # keeps them from substring-matching ".in_proj_qkv(z)" above.
            ".in_proj_q_shard": (".in_proj_qkvz", 0),
            ".in_proj_k_shard": (".in_proj_qkvz", 1),
            ".in_proj_v_shard": (".in_proj_qkvz", 2),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen3_5TextConfig = vllm_config.model_config.hf_text_config

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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
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
    # Maps PEFT embed/lm_head LoRA targets onto vLLM embedding wrappers.
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Some community text-only checkpoints keep the extraneous
    # `model.language_model.` prefix inherited from the VL training stack.
    # Strip it so both prefixed and clean checkpoints load correctly.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."},
    )

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
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
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
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

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
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


# NOTE(slimserve): upstream defines Qwen3_5MoeForCausalLM,
# Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration, and the
# associated Qwen3VL-based ProcessingInfo/MultiModalProcessor classes here.
# The MoE classes are out of scope for this dense-only fork. The multimodal
# Qwen3_5ForConditionalGeneration (vision tower + language model) is
# intentionally deferred: the vision half will be built separately on this
# fork's Muse-Glimmer pattern.
