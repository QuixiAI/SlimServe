# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Muse-Glimmer-30B, from the published GGUF set.

Dense 52-layer decoder. Distinctives over the Qwen3 shape this is otherwise
closest to:

- [local, local, local, global] interleaving: local layers use a 2048-token
  sliding window and RoPE (theta 500k); global layers attend fully and carry
  no position encoding (NoPE).
- Sigmoid-gated attention output: attn_out * sigmoid(W_gate x) before o_proj.
- Sandwich norms: RMSNorm before and after both the attention and FFN halves
  (Gemma2-style), on the residual branch.
- Per-head QK-RMSNorm, SwiGLU FFN, untied 202k-token head, final logits
  scaled by `logit_scale` then soft-capped at `final_logit_softcapping`.

The drafter (`MuseGlimmerDFlashDraftModel`, see muse_glimmer_dflash.py) reuses
these layers with `gated_attention=False, sandwich_norms=False`.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.muse_glimmer import MuseGlimmerConfig

from .interfaces import SupportsEagle3, SupportsPP
from .utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


def layer_is_local(config: MuseGlimmerConfig, layer_idx: int) -> bool:
    """True when the layer slides (window + RoPE); False for global NoPE."""
    pattern = getattr(config, "sliding_window_pattern", None)
    if pattern is None:
        return (layer_idx % 4) != 3
    return bool(pattern[layer_idx])


class MuseGlimmerMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self,
        config: MuseGlimmerConfig,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_local = layer_is_local(config, layer_idx)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if getattr(config, "gated_attention", True):
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.gate_proj",
            )
        else:
            self.gate_proj = None

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # RoPE only on local (sliding) layers; global layers are NoPE.
        # The GGUF conversion un-permutes Q/K into ggml's interleaved (GPT-J)
        # rope layout for the target (the DFlash drafter keeps NeoX), so the
        # target must run interleaved rotation.
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=getattr(config, "rope_is_neox", False),
                rope_parameters=config.rope_parameters,
            )
            if self.is_local
            else None
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=(config.sliding_window if self.is_local else None),
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], -1, self.head_dim)
        q = self.q_norm(q_by_head).view(q.shape)
        k_by_head = k.view(*k.shape[:-1], -1, self.head_dim)
        k = self.k_norm(k_by_head).view(k.shape)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.gate_proj is not None:
            attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        output, _ = self.o_proj(attn_output)
        return output


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseGlimmerConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.self_attn = MuseGlimmerAttention(
            config,
            layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MuseGlimmerMLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        eps = config.rms_norm_eps
        # The reference uses a tighter epsilon on the two post-norms
        # (post_norm_eps = 1e-8); the GGUF has no key for it.
        post_eps = getattr(config, "post_norm_eps", 1e-8)
        self.sandwich_norms = getattr(config, "sandwich_norms", True)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        if self.sandwich_norms:
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=post_eps)
            self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=post_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, self.input_layernorm(hidden_states))
        if self.sandwich_norms:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.sandwich_norms:
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


@support_torch_compile
class MuseGlimmerModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MuseGlimmerDecoderLayer(
                config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], 0
        )
        # DFlash/EAGLE aux capture: id N means the output of 0-based layer
        # N - 1 (id 0 is the embedding output), matching EagleModelMixin.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    _fused_ready: bool | None = None

    def _init_fused_step(self) -> bool:
        """Register weights and geometry for the single-command-buffer step.

        Returns False (and disables the path) when the layout is not the
        expected all-GGUF Metal configuration.
        """
        from vllm.quixicore.ops import _qc

        cfg = self.config
        try:
            qc = _qc()
            first_local = next(
                layer for layer in self.layers if layer.self_attn.is_local
            )
            first_full = next(
                layer for layer in self.layers if not layer.self_attn.is_local
            )
            self._fused_local_name = first_local.self_attn.attn.layer_name
            self._fused_full_name = first_full.self_attn.attn.layer_name

            def shards(module, count):
                cached = getattr(module, "_gguf_hetero_shards", None)
                if cached is not None:
                    return [w for w, _ in cached], [int(t) for _, t in cached]
                qw = module.qweight
                fallback = module.qweight_type.weight_type
                ws, ts = [], []
                for idx in qw.shard_id:
                    start, end, offset = qw.shard_offset_map[idx]
                    ws.append(qw[start:end, :offset].contiguous())
                    ts.append(
                        int(module.qweight_type.shard_weight_type.get(idx, fallback))
                    )
                assert len(ws) == count
                return ws, ts

            qc.muse_step_init(
                num_layers=len(self.layers),
                hidden=cfg.hidden_size,
                heads=cfg.num_attention_heads,
                kv_heads=cfg.num_key_value_heads,
                head_dim=cfg.head_dim,
                inter=cfg.intermediate_size,
                window=cfg.sliding_window,
                theta=cfg.rope_theta,
                eps=cfg.rms_norm_eps,
                post_eps=getattr(cfg, "post_norm_eps", 1e-8),
                max_rows=17,
                ref=self.norm.weight.data,
            )
            for i, layer in enumerate(self.layers):
                attn = layer.self_attn
                qkv_w, qkv_t = shards(attn.qkv_proj, 3)
                gu_w, gu_t = shards(layer.mlp.gate_up_proj, 2)
                kv_cache = attn.attn.kv_cache
                if isinstance(kv_cache, (list, tuple)):
                    kv_cache = kv_cache[0]
                qc.muse_step_layer(
                    i,
                    attn.is_local,
                    qkv_w,
                    qkv_t,
                    attn.gate_proj.qweight,
                    int(attn.gate_proj.qweight_type.weight_type),
                    attn.o_proj.qweight,
                    int(attn.o_proj.qweight_type.weight_type),
                    gu_w,
                    gu_t,
                    layer.mlp.down_proj.qweight,
                    int(layer.mlp.down_proj.qweight_type.weight_type),
                    layer.input_layernorm.weight.data,
                    attn.q_norm.weight.data,
                    attn.k_norm.weight.data,
                    layer.post_attention_layernorm.weight.data,
                    layer.pre_feedforward_layernorm.weight.data,
                    layer.post_feedforward_layernorm.weight.data,
                    kv_cache,
                )
            return True
        except Exception:
            logger.exception(
                "muse fused decode step unavailable; staying on the eager path"
            )
            return False

    def _maybe_fused_decode(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor | None:
        """Single-command-buffer decode for pure single-token steps."""
        if self.aux_hidden_state_layers:
            return None  # the spec target must return per-layer captures
        if hidden_states.device.type != "mps":
            return None
        from vllm.forward_context import get_forward_context
        from vllm.quixicore import quixicore_ops

        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        if self._fused_ready is None:
            self._fused_ready = quixicore_ops.is_available() and self._init_fused_step()
        if not self._fused_ready:
            return None
        local = metadata.get(self._fused_local_name)
        full = metadata.get(self._fused_full_name)
        if local is None or full is None:
            return None
        if (
            local.max_query_len != 1
            or local.num_actual_tokens != local.num_reqs
            or hidden_states.shape[0] != local.num_actual_tokens
        ):
            return None
        from vllm.quixicore.ops import _qc

        x = hidden_states.contiguous()
        _qc().muse_step_run(
            x,
            positions.to(torch.int32),
            local.block_table,
            local.seq_lens_gpu,
            local.slot_mapping.to(torch.long),
            full.block_table,
            full.seq_lens_gpu,
            full.slot_mapping.to(torch.long),
        )
        return x

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        # Weightless RMSNorm on the embeddings before the stack
        # (llama.cpp muse-glimmer.cpp: `embd_norm`). Raw embedding RMS is
        # ~0.06; the residual stream is trained against unit-RMS inputs.
        hidden_states = torch.nn.functional.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            None,
            self.config.rms_norm_eps,
        ).to(hidden_states.dtype)

        fused = self._maybe_fused_decode(hidden_states, positions)
        if fused is not None:
            return self.norm(fused)

        aux_hidden_states = [hidden_states] if 0 in self.aux_hidden_state_layers else []
        for idx, layer in enumerate(self.layers[self.start_layer : self.end_layer]):
            hidden_states = layer(positions, hidden_states)
            if (idx + 1) in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class MuseGlimmerForCausalLM(nn.Module, SupportsPP, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        quant_config = vllm_config.quant_config
        self.config = config

        self.model = MuseGlimmerModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        # Scale-then-softcap, per the reference: T * tanh(logits * scale / T).
        # LogitsProcessor's own soft_cap applies before its scale, which is
        # the wrong order for this model, so both are applied here instead.
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        assert get_pp_group().world_size == 1, (
            "Muse-Glimmer serves single-device; pipeline parallel is unsupported."
        )
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        scale = getattr(self.config, "logit_scale", 1.0)
        if scale != 1.0:
            logits = logits * scale
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            # Attention output gate is its own module; do not stack it.
            mapped = None
            if ".self_attn.gate_proj" not in name:
                for target, source, shard_id in stacked_params_mapping:
                    if source in name:
                        mapped = (name.replace(source, target), shard_id)
                        break
            if mapped is not None:
                target_name, shard_id = mapped
                param = params_dict[target_name]
                param.weight_loader(param, weight, shard_id)
                loaded.add(target_name)
            else:
                param = params_dict[name]
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, weight)
                loaded.add(name)
        return loaded


# ---------------------------------------------------------------------------
# Vision: Perception-Encoder ViT tower + 3-layer MLP projector, from mmproj.
# ---------------------------------------------------------------------------
#
# The published mmproj carries a 50-block pre-norm ViT (width 1536, 16 heads,
# biases everywhere, plain non-gated MLP) with a 14px patch embedding and
# 1024 learned positions (a 32x32 grid). Images are resized to 896x896 (a
# 64x64 patch grid); the position table is bilinearly interpolated to 64x64.
# After the tower, 2x2 neighboring patches are concatenated (6144) and pass
# through mm.0 (6144->4096), mm.1 (4096->4096), mm.2 (4096->6656).
#
# Assumptions the GGUF cannot express, to be validated against output
# quality: GELU in the tower MLP and between projector layers (the
# CLIP/Perception-Encoder convention), and single-tile 896x896 preprocessing.

from collections.abc import Mapping, Sequence  # noqa: E402
from typing import Any  # noqa: E402

import torch.nn.functional as F  # noqa: E402
from transformers import BatchFeature  # noqa: E402

from vllm.model_executor.models.interfaces import SupportsMultiModal  # noqa: E402
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402
from vllm.multimodal.inputs import (  # noqa: E402
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import (  # noqa: E402
    MultiModalDataItems,
)
from vllm.multimodal.processing import (  # noqa: E402
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

from .utils import init_vllm_registered_model  # noqa: E402

IMAGE_PLACEHOLDER = "<|patch|>"


def _rope_2d(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J interleaved rotation over the last dim, per-position tables."""
    # x: [batch, seq, heads, half]; cos/sin: [seq, half/2]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


class MuseGlimmerVisionBlock(nn.Module):
    def __init__(
        self, hidden: int, heads: int, intermediate: int, eps: float, is_global: bool
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden, eps=eps)
        self.layer_norm2 = nn.LayerNorm(hidden, eps=eps)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.self_attn.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.self_attn.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.self_attn.out_proj = nn.Linear(hidden, hidden, bias=True)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.mlp.fc2 = nn.Linear(intermediate, hidden, bias=True)
        self.num_heads = heads
        self.head_dim = hidden // heads
        self.is_global = is_global

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_w: torch.Tensor,
        rope_sin_w: torch.Tensor,
        rope_cos_h: torch.Tensor,
        rope_sin_h: torch.Tensor,
        num_windows: int,
    ) -> torch.Tensor:
        # x: [batch, seq, hidden], seq already in window-grouped order.
        residual = x
        h = self.layer_norm1(x)
        b, s, d = h.shape
        q = self.self_attn.q_proj(h).view(b, s, self.num_heads, self.head_dim)
        k = self.self_attn.k_proj(h).view(b, s, self.num_heads, self.head_dim)
        v = self.self_attn.v_proj(h).view(b, s, self.num_heads, self.head_dim)
        # 2D RoPE on every layer: first half of head_dim rotates by the width
        # position, second half by the height position (theta 10000).
        half = self.head_dim // 2
        q = torch.cat(
            [
                _rope_2d(q[..., :half], rope_cos_w, rope_sin_w),
                _rope_2d(q[..., half:], rope_cos_h, rope_sin_h),
            ],
            dim=-1,
        )
        k = torch.cat(
            [
                _rope_2d(k[..., :half], rope_cos_w, rope_sin_w),
                _rope_2d(k[..., half:], rope_cos_h, rope_sin_h),
            ],
            dim=-1,
        )
        if self.is_global or num_windows <= 1:
            attn = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
            attn = attn.transpose(1, 2).reshape(b, s, d)
        else:
            # Block-diagonal window attention: tokens are window-grouped, so
            # batched SDPA per window is exact.
            wl = s // num_windows
            qw = q.view(b * num_windows, wl, self.num_heads, self.head_dim)
            kw = k.view(b * num_windows, wl, self.num_heads, self.head_dim)
            vw = v.view(b * num_windows, wl, self.num_heads, self.head_dim)
            attn = F.scaled_dot_product_attention(
                qw.transpose(1, 2), kw.transpose(1, 2), vw.transpose(1, 2)
            )
            attn = attn.transpose(1, 2).reshape(b, s, d)
        x = residual + self.self_attn.out_proj(attn)
        residual = x
        h = self.mlp.fc2(F.gelu(self.mlp.fc1(self.layer_norm2(x))))
        return residual + h


class MuseGlimmerVisionTower(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        v = config
        self.patch_size = v.patch_size
        self.hidden_size = v.hidden_size
        self.patch_embed = nn.Conv2d(
            3, v.hidden_size, kernel_size=v.patch_size, stride=v.patch_size, bias=False
        )
        self.position_embed = nn.Parameter(torch.zeros(v.num_positions, v.hidden_size))
        self.pre_layernorm = nn.LayerNorm(v.hidden_size, eps=v.layer_norm_eps)
        # Sparse block-diagonal window attention on most layers; every 4th
        # layer (1-based) and the last one attend globally.
        n = v.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                MuseGlimmerVisionBlock(
                    v.hidden_size,
                    v.num_attention_heads,
                    v.intermediate_size,
                    v.layer_norm_eps,
                    is_global=((i + 1) % 4 == 0) or (i == n - 1),
                )
                for i in range(n)
            ]
        )
        self.post_layernorm = nn.LayerNorm(v.hidden_size, eps=v.layer_norm_eps)
        self.rope_theta = 10000.0

    def _interp_positions(self, grid: int) -> torch.Tensor:
        table = self.position_embed
        src = int(table.shape[0] ** 0.5)
        if src * src != table.shape[0]:
            raise ValueError(f"non-square position table: {table.shape}")
        if src == grid:
            return table
        t = table.view(1, src, src, -1).permute(0, 3, 1, 2)
        t = F.interpolate(
            t.float(), size=(grid, grid), mode="bilinear", align_corners=False
        ).to(table.dtype)
        return t.permute(0, 2, 3, 1).reshape(grid * grid, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [batch, 3, H, W]; H == W and divisible by patch_size.
        x = self.patch_embed(pixel_values.to(self.patch_embed.weight.dtype))
        b, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # [b, gh*gw, d], row-major grid
        x = x + self._interp_positions(gh).to(x.dtype)

        # Window grouping (32x32 patches per window, the position-table grid).
        device = x.device
        win = int(self.position_embed.shape[0] ** 0.5)
        nwin_h = (gh + win - 1) // win
        nwin_w = (gw + win - 1) // win
        rows = torch.arange(gh, device=device)
        cols = torch.arange(gw, device=device)
        gy, gx = torch.meshgrid(rows, cols, indexing="ij")
        wy, hh = gy // win, gy % win
        wx, ww = gx // win, gx % win
        # Permuted rank of each grid cell: windows row-major, row-major inside.
        rank = ((wy * nwin_w + wx) * (win * win) + hh * win + ww).flatten()
        sp_perm = torch.argsort(rank)
        inv_perm = torch.argsort(sp_perm)
        x = x[:, sp_perm]

        # 1-indexed per-axis rope tables in permuted order, theta 10000,
        # GPT-J interleaved pairs within each half of head_dim.
        head_dim = self.layers[0].head_dim
        half = head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, half, 2, device=device, dtype=torch.float32) / half)
        )
        pos_w = (sp_perm % gw + 1).to(torch.float32)
        pos_h = (sp_perm // gw + 1).to(torch.float32)
        fw = torch.outer(pos_w, inv_freq)
        fh = torch.outer(pos_h, inv_freq)
        cw, sw = fw.cos().to(x.dtype), fw.sin().to(x.dtype)
        ch, sh = fh.cos().to(x.dtype), fh.sin().to(x.dtype)

        num_windows = nwin_h * nwin_w
        x = self.pre_layernorm(x)
        for layer in self.layers:
            x = layer(x, cw, sw, ch, sh, num_windows)
        x = self.post_layernorm(x)
        return x[:, inv_perm]


class MuseGlimmerProjector(nn.Module):
    def __init__(self, vision_hidden: int, merge: int, mid: int, out: int) -> None:
        super().__init__()
        self.merge = merge
        self.linear_1 = nn.Linear(vision_hidden * merge * merge, mid, bias=False)
        self.linear_2 = nn.Linear(mid, mid, bias=False)
        self.linear_3 = nn.Linear(mid, out, bias=False)

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        # x: [batch, grid*grid, hidden] in row-major grid order. Pixel-shuffle
        # each 2x2 neighborhood, channel-outer: the 6144 projector input is
        # [c0s0, c0s1, c0s2, c0s3, c1s0, ...] (spatial index fastest).
        b, s, d = x.shape
        m = self.merge
        x = x.view(b, grid // m, m, grid // m, m, d)
        # -> [b, oy, ox, ry, rx, d] -> [b, out_tokens, d, m*m]
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(b, (grid // m) ** 2, d, m * m)
        x = x.reshape(b, (grid // m) ** 2, d * m * m)
        x = self.linear_1(x.to(self.linear_1.weight.dtype))
        x = F.gelu(x)
        x = self.linear_2(x)
        x = F.gelu(x)
        return self.linear_3(x)


from transformers.image_processing_utils import BaseImageProcessor  # noqa: E402
from transformers.processing_utils import ProcessorMixin  # noqa: E402


class MuseGlimmerImageProcessor(BaseImageProcessor):
    """Fixed single-tile preprocessing: resize to image_size^2, normalize."""

    model_input_names = ["pixel_values"]

    def __init__(self, vision_config=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.merge = vision_config.spatial_merge_size
        self.mean = vision_config.image_mean
        self.std = vision_config.image_std

    @property
    def tokens_per_image(self) -> int:
        grid = self.image_size // self.patch_size
        return (grid // self.merge) ** 2

    def preprocess(self, images, **kwargs) -> BatchFeature:
        import numpy as np
        from PIL import Image

        if not isinstance(images, (list, tuple)):
            images = [images]
        out = []
        for img in images:
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.asarray(img))
            img = img.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            arr = torch.from_numpy(np.asarray(img).copy()).float() / 255.0
            arr = arr.permute(2, 0, 1)
            mean = torch.tensor(self.mean).view(3, 1, 1)
            std = torch.tensor(self.std).view(3, 1, 1)
            out.append((arr - mean) / std)
        return BatchFeature({"pixel_values": torch.stack(out)}, tensor_type=None)


class MuseGlimmerProcessor(ProcessorMixin):
    """HF-processor facade over the tokenizer and image processor."""

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, tokenizer, image_processor, image_token_id: int) -> None:
        # Kimi-style init: set attributes directly, skip ProcessorMixin's
        # class-name validation (these are runtime-built objects).
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        data: dict[str, Any] = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            encodings = [self.tokenizer(t, add_special_tokens=False) for t in text]
            data["input_ids"] = [e["input_ids"] for e in encodings]
        if images:
            data.update(self.image_processor.preprocess(images))
        return BatchFeature(data, tensor_type=None)


class MuseGlimmerProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self):
        tokenizer = self.get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if not isinstance(image_token_id, int):
            raise ValueError(f"tokenizer cannot resolve {IMAGE_PLACEHOLDER!r} to an id")
        config = self.get_hf_config()
        config.media_placeholder_token_id = image_token_id
        return MuseGlimmerProcessor(
            tokenizer,
            MuseGlimmerImageProcessor(config.vision_config),
            image_token_id,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class MuseGlimmerDummyInputsBuilder(BaseDummyInputsBuilder[MuseGlimmerProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
    ):
        size = self.info.get_hf_config().vision_config.image_size
        return {
            "image": self._get_dummy_images(
                width=size, height=size, num_images=mm_counts.get("image", 0)
            )
        }


class MuseGlimmerMultiModalProcessor(
    BaseMultiModalProcessor[MuseGlimmerProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor()
        tokens_per_image = processor.image_processor.tokens_per_image
        image_token_id = processor.image_token_id

        def get_replacement(item_idx: int):
            return [image_token_id] * tokens_per_image

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    MuseGlimmerMultiModalProcessor,
    info=MuseGlimmerProcessingInfo,
    dummy_inputs=MuseGlimmerDummyInputsBuilder,
)
class MuseGlimmerForConditionalGeneration(
    nn.Module, SupportsMultiModal, SupportsPP, SupportsEagle3
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        vision_config = config.vision_config

        self.vision_tower = MuseGlimmerVisionTower(vision_config)
        self.projector = MuseGlimmerProjector(
            vision_config.hidden_size,
            vision_config.spatial_merge_size,
            vision_config.projector_hidden_size,
            config.hidden_size,
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            architectures=["MuseGlimmerForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return None
        if isinstance(pixel_values, list):
            pixel_values = torch.cat(
                [p if p.dim() == 4 else p.unsqueeze(0) for p in pixel_values]
            )
        elif pixel_values.dim() == 5:
            pixel_values = pixel_values.flatten(0, 1)
        device = self.vision_tower.patch_embed.weight.device
        features = self.vision_tower(pixel_values.to(device))
        grid = self.config.vision_config.image_size // (
            self.config.vision_config.patch_size
        )
        projected = self.projector(features, grid)
        return list(projected.unbind(0))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings and is_multimodal is not None:
            flat = torch.cat([e.to(inputs_embeds.dtype) for e in multimodal_embeddings])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[is_multimodal] = flat.to(inputs_embeds.device)
        return inputs_embeds

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
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.model.aux_hidden_state_layers

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        lang_weights = []
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith(("vision_tower.", "projector.")):
                param = params_dict[name]
                param.data.copy_(weight.to(param.dtype))
                loaded.add(name)
            elif name.startswith("language_model."):
                lang_weights.append((name[len("language_model.") :], weight))
            else:
                lang_weights.append((name, weight))
        loaded.update(
            "language_model." + n
            for n in self.language_model.load_weights(lang_weights)
        )
        return loaded
