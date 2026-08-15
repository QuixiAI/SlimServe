# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash drafter for Muse-Glimmer-30B, from the published GGUF.

Architecturally this drafter is exactly the DFlashQwen3 shape: five decoder
layers of QKV + per-head QK-RMSNorm + RoPE + sliding-window attention and a
SwiGLU MLP, plus the `fc` fusion of five concatenated target hidden states.
The checkpoint carries no token embedding and no output head; both are
shared with the Muse-Glimmer target through the generic dflash proposer
(`has_own_embed_tokens` / `has_own_lm_head`), the same contract as the
Laguna drafter.

GGUF tensor names arrive from `MuseGlimmerDFlashGGUFAdapter` already in
HF-style form (`model.layers.N...`, `model.fc.weight`,
`model.hidden_norm.weight` for `enc.output_norm`, `model.norm.weight`).
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class MuseGlimmerDFlashModel(DFlashQwen3Model):
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Torch-native context K/V precompute for MPS.

        The parent uses the CUDA custom ops (`ops.rms_norm`,
        `ops.rotary_embedding`) and the TurboQuant `do_kv_cache_update`;
        none exist on Metal. Context volumes here are small (accepted tokens
        per step across five layers), so plain torch is adequate.
        """
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()

        hidden = context_states.shape[-1]
        normed = F.rms_norm(
            context_states.float(),
            (hidden,),
            self._hidden_norm_weight.float(),
            self._rms_norm_eps,
        ).to(context_states.dtype)

        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            kv = attn.qkv_proj(normed)[0][..., attn.q_size :]
            k, v = kv.split([attn.kv_size, attn.kv_size], dim=-1)
            k = F.rms_norm(
                k.view(-1, attn.num_kv_heads, attn.head_dim).float(),
                (attn.head_dim,),
                attn.k_norm.weight.float(),
                self._rms_norm_eps,
            ).to(kv.dtype)
            k = k.view(-1, attn.kv_size)
            k, _ = attn.rotary_emb(context_positions, k, None)

            if context_slot_mapping is None:
                continue
            slot_mapping = (
                context_slot_mapping[i] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            kv_cache = attn.attn.kv_cache
            if isinstance(kv_cache, (list, tuple)):
                kv_cache = kv_cache[0]
            if kv_cache.numel() == 0:
                continue  # profiling run: no cache allocated yet
            block_size = kv_cache.shape[2]
            slot = slot_mapping.to(torch.long)
            block_idx = slot // block_size
            block_off = slot % block_size
            k_heads = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v_heads = v.reshape(-1, attn.num_kv_heads, attn.head_dim)
            kv_cache[0][block_idx, block_off] = k_heads.to(kv_cache.dtype)
            kv_cache[1][block_idx, block_off] = v_heads.to(kv_cache.dtype)


class MuseGlimmerDFlashDraftModel(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "Muse-Glimmer DFlash shares the target lm_head and requires "
                "matching vocabularies "
                f"({self.config.draft_vocab_size} != {target_vocab_size})."
            )
        # Shared with the target through the generic dflash proposer.
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = MuseGlimmerDFlashModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)
        self.draft_id_to_target_id = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Names are already HF-style from the GGUF adapter; no re-prefixing.
        model_weights = {}
        for name, loaded_weight in weights:
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            # Shared with the target; absent from the drafter GGUF.
            skip_substrs=["embed_tokens", "lm_head", "mask_embedding"],
        )
        loaded = loader.load_weights(
            model_weights.items(), mapper=self.model.hf_to_vllm_mapper
        )
        loaded.add("lm_head.weight")
        loaded.add("model.embed_tokens.weight")
        self.model._build_fused_kv_buffers()
        return loaded
