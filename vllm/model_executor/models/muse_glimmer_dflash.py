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


def _store_kv_at_slots(
    kv_cache: torch.Tensor,
    attn,
    slot: torch.Tensor,
    k_heads: torch.Tensor,
    v_heads: torch.Tensor,
) -> None:
    """Write per-token K/V into whichever cache layout the backend uses.

    Two layouts are in play and they are not interchangeable:

    * split  ``(2, num_blocks, block_size, num_kv_heads, head_size)`` --
      leading K/V planes, used by the Metal path this drafter was written
      against.
    * packed ``(num_blocks, num_kv_heads, block_size, 2 * head_size)`` --
      K and V interleaved in the content dim, used by TRITON_ATTN and
      ROCM_AITER_FA.

    Assuming the split layout on a packed cache silently indexes block 0
    as if it were the K plane; the shapes happen to stay plausible, so it
    surfaces as a broadcast error rather than as corruption (observed:
    value [N, 8, 128] into indexing result [N, 256], where 256 is
    2 * head_size). Anything unrecognized raises rather than guessing.
    """
    head_size = k_heads.shape[-1]
    if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
        block_size = kv_cache.shape[2]
        block_idx, block_off = slot // block_size, slot % block_size
        kv_cache[0][block_idx, block_off] = k_heads.to(kv_cache.dtype)
        kv_cache[1][block_idx, block_off] = v_heads.to(kv_cache.dtype)
        return
    if kv_cache.dim() == 4 and kv_cache.shape[-1] >= 2 * head_size:
        # (num_blocks, num_kv_heads, block_size, 2 * head_size [+ scales]).
        # Advanced indices at positions 0 and 2 are separated by a slice,
        # so the gathered dim leads: the view is [N, num_kv_heads, ...].
        block_size = kv_cache.shape[2]
        block_idx, block_off = slot // block_size, slot % block_size
        kv_cache[block_idx, :, block_off, :head_size] = k_heads.to(kv_cache.dtype)
        kv_cache[block_idx, :, block_off, head_size : 2 * head_size] = v_heads.to(
            kv_cache.dtype
        )
        return
    raise RuntimeError(
        "dflash drafter: unrecognized KV cache layout "
        f"{tuple(kv_cache.shape)} for head_size={head_size}; refusing to "
        "guess which axis holds K/V."
    )


class MuseGlimmerDFlashModel(DFlashQwen3Model):
    _fused_ready: bool | None = None

    def _init_fused_step(self) -> bool:
        """Register geometry + weights for the fused single-command-buffer
        drafter forward (dflash_step in the Metal extension)."""
        try:
            import os

            from vllm.quixicore.ops import _qc

            qc = _qc()
            cfg = self.config
            heads = cfg.num_attention_heads
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // heads)
            dflash_cfg = getattr(cfg, "dflash_config", {}) or {}
            window = dflash_cfg.get("swa_window_size") or getattr(
                cfg, "sliding_window", 0
            )
            if not window:
                return False

            def shards(module, count):
                # Mixed-type merged layers (the drafter QKV: q4/q4/q6)
                # materialize per-shard views on first forward and release
                # the merged buffer; reuse those views if present.
                hetero = getattr(module, "_gguf_hetero_shards", None)
                if hetero is not None:
                    ws = [w for w, _ in hetero]
                    ts = [int(t) for _, t in hetero]
                    assert len(ws) == count
                    return ws, ts
                qw = module.qweight
                fallback = int(module.qweight_type.weight_type)
                if not getattr(qw, "shard_offset_map", None):
                    return [qw], [fallback]
                ws, ts = [], []
                for idx in qw.shard_id:
                    start, end, offset = qw.shard_offset_map[idx]
                    ws.append(qw[start:end, :offset].contiguous())
                    ts.append(
                        int(module.qweight_type.shard_weight_type.get(idx, fallback))
                    )
                assert len(ws) == count
                return ws, ts

            trunc = int(os.environ.get("DFLASH_FUSED_TRUNC", "0")) or len(self.layers)
            qc.dflash_step_init(
                min(trunc, len(self.layers)),
                cfg.hidden_size,
                heads,
                cfg.num_key_value_heads,
                head_dim,
                cfg.intermediate_size,
                window,
                cfg.rope_theta,
                cfg.rms_norm_eps,
                17,
                self.norm.weight.data,
            )
            for i, layer in enumerate(self.layers):
                if i >= trunc:
                    break
                attn = layer.self_attn
                qkv_w, qkv_t = shards(attn.qkv_proj, 3)
                gu_w, gu_t = shards(layer.mlp.gate_up_proj, 2)
                kv_cache = attn.attn.kv_cache
                if isinstance(kv_cache, (list, tuple)):
                    kv_cache = kv_cache[0]
                qc.dflash_step_layer(
                    i,
                    qkv_w,
                    qkv_t,
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
                    kv_cache,
                )
            return True
        except Exception:
            logger.exception("fused dflash drafter step unavailable; staying on eager")
            return False

    def _maybe_fused_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Single-command-buffer drafter forward for the block-denoise shape
        (one request, whole block queried at once, BIDIRECTIONAL within the
        block: every expanded row attends to the full base + m sequence).
        Gated OFF by default until the rested A/B passes."""
        import os

        if os.environ.get("VLLM_MUSE_FUSED_DRAFTER", "0") != "1":
            return None
        if positions.device.type != "mps":
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
        meta = metadata.get(self.layers[0].self_attn.attn.layer_name)
        if meta is None:
            return None
        m = meta.num_actual_tokens
        if not (meta.num_reqs == 1 and meta.max_query_len == m and m <= 17):
            return None
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        if input_embeds.shape[0] != m or input_embeds.dtype != torch.bfloat16:
            return None
        from vllm.quixicore.ops import _qc

        x = input_embeds.contiguous()
        pos = positions.to(torch.int32)
        bt = meta.block_table[:1].expand(m, -1)
        # This fork's DFlash runs SWA layers CAUSAL (qwen3_dflash.py:59; the
        # published config sets no dflash_config.causal override), so the
        # expansion is the target verify's exact per-row causal construction:
        # row i sees base + i + 1. (`+ steps` also materializes the tensor;
        # the kernel reads seq_lens as a contiguous int32 array.)
        steps = torch.arange(m, dtype=torch.int32, device=x.device)
        sl = (meta.seq_lens_gpu[:1] - (m - 1)).expand(m) + steps
        _qc().dflash_step_run(x, pos, bt, sl, meta.slot_mapping.to(torch.long))
        return self.norm(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fused = self._maybe_fused_forward(input_ids, positions, input_embeds)
        if fused is not None:
            return fused
        return super().forward(input_ids, positions, input_embeds)

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
            slot = slot_mapping.to(torch.long)
            k_heads = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v_heads = v.reshape(-1, attn.num_kv_heads, attn.head_dim)
            _store_kv_at_slots(kv_cache, attn, slot, k_heads, v_heads)


class MuseGlimmerDFlashDraftModel(DFlashQwen3ForCausalLM):
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft sampling in one command buffer (shared lm_head GEMM
        + row argmax; softcap/scale are argmax-invariant). Enabled via
        speculative_config.use_local_argmax_reduction; falls back to the
        compute_logits path off-Metal or off-shape."""
        lm = self.lm_head
        qw = getattr(lm, "qweight", None)
        m = hidden_states.shape[0]
        if (
            hidden_states.device.type == "mps"
            and qw is not None
            and 9 <= m <= 17
            and qw.shape[0] % 64 == 0
        ):
            try:
                from vllm.quixicore.ops import _qc

                return _qc().dflash_sample_greedy(
                    hidden_states.contiguous(),
                    qw,
                    int(lm.qweight_type.weight_type),
                    qw.shape[0],
                )
            except Exception:
                logger.exception("fused greedy sampling failed; eager fallback")
        logits = self.compute_logits(hidden_states)
        return logits.argmax(dim=-1)

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

    @staticmethod
    def _remap_published(name: str) -> str:
        """Map the published safetensors names (bare, `encoder.*`) onto the
        module tree. The GGUF adapter already emits HF-style names, which
        pass through unchanged."""
        if name.startswith("model."):
            return name
        if name.startswith("encoder.fc."):
            return "model.fc." + name[len("encoder.fc.") :]
        if name.startswith("encoder.output_norm_enc."):
            return "model.hidden_norm." + name[len("encoder.output_norm_enc.") :]
        if name.startswith("layers."):
            return "model." + name
        if name == "norm.weight":
            return "model.norm.weight"
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Names are already HF-style from the GGUF adapter; the published
        # safetensors assistant needs the remap above.
        model_weights = {}
        for name, loaded_weight in weights:
            name = self._remap_published(name)
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
        # Filled from the target's embedding during sharing setup, like the
        # two above (absent from both drafter artifacts).
        loaded.add("model.mask_embedding")
        self.model._build_fused_kv_buffers()
        return loaded
