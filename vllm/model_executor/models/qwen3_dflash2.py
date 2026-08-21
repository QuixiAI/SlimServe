# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash 2 drafter for Qwen3.8-27B, from the published GGUF.

The backbone is exactly the DFlashQwen3 shape the Muse-Glimmer drafter uses:
five decoder layers of split-QKV + per-head QK-RMSNorm + RoPE + sliding-window
(here explicitly non-causal) attention and a SwiGLU MLP, plus the `fc` fusion
of five concatenated target hidden states. DFlash 2 adds two things
(inco.ai/blog/dflash2, llama.cpp PR #27342; layouts verified against the
GGUF -- perf/qwen38_metal_design.md):

- A two-tap dynamic depthwise convolution before AND after every attention
  and MLP sublayer. Each sublayer pair computes one dynamic coefficient
  projection from its normed input and applies `base + coeff` weighted taps
  block-locally (tap 1 reads the predecessor row; row 0 of each 1+N draft
  block is the last verified token, so position 1's tap naturally reads it).
- A path selector over the top-k candidates per position: adjacent pairs
  score `S(a, b) = U(b) + <A(a) * H(h), B(b)>`; a sequential walk from the
  anchor picks one coherent path. `select_draft_path` runs the walk directly
  (greedy at T=0), so only the actual predecessor's score row is ever
  computed.

The checkpoint carries no token embedding and no output head; both are
shared with the target through the generic dflash proposer, the same
contract as the Muse and Laguna drafters.
"""

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.muse_glimmer_dflash import MuseGlimmerDFlashModel

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)

# Env-gated diagnostic dump for the recall@k replay (set
# QWEN38_DFLASH_DUMP=<dir> to record per-propose drafter candidates).
_DUMP_DIR = os.environ.get("QWEN38_DFLASH_DUMP")
_DUMP_STEP = [0]


def _dump_draft_record(record: dict) -> None:
    if _DUMP_DIR is None:
        return
    os.makedirs(_DUMP_DIR, exist_ok=True)
    torch.save(record, os.path.join(_DUMP_DIR, f"draft_{_DUMP_STEP[0]:05d}.pt"))
    _DUMP_STEP[0] += 1


def _apply_two_tap_conv(
    x: torch.Tensor,
    coeffs: torch.Tensor,
    base: torch.Tensor,
    side: int,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """One side of the dynamic conv: out_t = w0*x_t + w1*x_{t-1} (block-local).

    x: (tokens, hidden). coeffs: (tokens, 2 sides, 2 taps, n_groups) from the
    sublayer pair's projection. base: (2 sides, 2 taps, hidden). The shift is
    local to each draft block and zero-padded at block row 0 (the anchor).
    """
    tokens, hidden = x.shape
    # Propose calls are always whole 1+N blocks. Anything else (profiling
    # dummy runs, parity harnesses) is treated as a single block: the shift
    # semantics stay well-defined and the shapes stay legal.
    bs = block_size if block_size and tokens % block_size == 0 else tokens
    dyn = coeffs[:, side].repeat_interleave(group_size, dim=-1)  # (T, 2, hidden)

    out = (base[side, 0] + dyn[:, 0]) * x
    shifted = x.view(-1, bs, hidden).roll(shifts=1, dims=1)
    shifted = shifted.clone()
    shifted[:, 0] = 0
    out = out + (base[side, 1] + dyn[:, 1]) * shifted.view(tokens, hidden)
    return out


class DFlash2QwenDecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        disable_tp: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            disable_tp=disable_tp,
        )
        dflash_config = getattr(config, "dflash_config", None) or {}
        self.conv_kernel_size = int(dflash_config["conv_kernel_size"])
        if self.conv_kernel_size != 2:
            raise ValueError(
                f"DFlash 2 conv is two-tap; got kernel {self.conv_kernel_size}"
            )
        group_channels = int(dflash_config["conv_group_size"])
        if config.hidden_size % group_channels != 0:
            raise ValueError("hidden_size must be divisible by conv_group_size")
        self.conv_group_channels = group_channels
        self.n_conv_groups = config.hidden_size // group_channels
        self.dflash_block_size = int(getattr(config, "block_size", 0))

        dtype = vllm_config.model_config.dtype
        conv_out = 2 * self.conv_kernel_size * self.n_conv_groups
        # Conv weights arrive dequantized from the GGUF adapter (they are
        # tiny next to the backbone); plain fp16 modules, no quant method.
        self.attn_conv_base = nn.Parameter(
            torch.zeros(2, self.conv_kernel_size, config.hidden_size, dtype=dtype),
            requires_grad=False,
        )
        self.attn_conv_proj = ReplicatedLinear(
            config.hidden_size,
            conv_out,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.attn_conv_proj",
            return_bias=False,
        )
        self.ffn_conv_base = nn.Parameter(
            torch.zeros(2, self.conv_kernel_size, config.hidden_size, dtype=dtype),
            requires_grad=False,
        )
        self.ffn_conv_proj = ReplicatedLinear(
            config.hidden_size,
            conv_out,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.ffn_conv_proj",
            return_bias=False,
        )

    def _conv_coeffs(self, proj: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
        # (tokens, 2*K*groups) -> (tokens, side, tap, group); the projection
        # output is laid out side-major, matching the GGUF tensor (verified
        # against llama.cpp's reshape in PR #27342).
        return proj(x).view(x.shape[0], 2, self.conv_kernel_size, self.n_conv_groups)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        attn_coeffs = self._conv_coeffs(self.attn_conv_proj, hidden_states)
        hidden_states = _apply_two_tap_conv(
            hidden_states,
            attn_coeffs,
            self.attn_conv_base,
            0,
            self.dflash_block_size,
            self.conv_group_channels,
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = _apply_two_tap_conv(
            hidden_states,
            attn_coeffs,
            self.attn_conv_base,
            1,
            self.dflash_block_size,
            self.conv_group_channels,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        ffn_coeffs = self._conv_coeffs(self.ffn_conv_proj, hidden_states)
        hidden_states = _apply_two_tap_conv(
            hidden_states,
            ffn_coeffs,
            self.ffn_conv_base,
            0,
            self.dflash_block_size,
            self.conv_group_channels,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = _apply_two_tap_conv(
            hidden_states,
            ffn_coeffs,
            self.ffn_conv_base,
            1,
            self.dflash_block_size,
            self.conv_group_channels,
        )
        return hidden_states, residual


class DFlash2QwenModel(DFlashQwen3Model):
    decoder_layer_cls = DFlash2QwenDecoderLayer

    # Torch-native context K/V precompute for MPS. The Muse implementation is
    # model-agnostic (it reads only generic DFlashQwen3 attention attributes),
    # so it is borrowed rather than copied; the CUDA parent keeps its custom
    # ops elsewhere.
    precompute_and_store_context_kv = (
        MuseGlimmerDFlashModel.precompute_and_store_context_kv
    )


class DFlash2QwenDraftModel(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = self.config.vocab_size
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "DFlash 2 shares the target lm_head and requires matching "
                f"vocabularies ({self.config.draft_vocab_size} != "
                f"{target_vocab_size})."
            )
        # Shared with the target through the generic dflash proposer.
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False
        # Vocabulary is shared with the target verbatim; no d2t mapping
        # (compute_logits reads this, and this __init__ bypasses the parent).
        self.draft_id_to_target_id = None

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlash2QwenModel(
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

        dflash_config = self.config.dflash_config
        dtype = vllm_config.model_config.dtype
        self.selector_rank = int(dflash_config["selector_rank"])
        self.selector_top_k = int(dflash_config["selector_top_k"])
        # A and B token tables plus the context gate H. The tables arrive
        # dequantized from the GGUF adapter (2 x ~127 MB fp16).
        self.selector_predecessor = nn.Embedding(
            self.config.vocab_size, self.selector_rank, dtype=dtype
        )
        self.selector_successor = nn.Embedding(
            self.config.vocab_size, self.selector_rank, dtype=dtype
        )
        self.selector_hidden = ReplicatedLinear(
            self.config.hidden_size,
            self.selector_rank,
            bias=False,
            params_dtype=dtype,
            prefix=maybe_prefix(prefix, "selector_hidden"),
            return_bias=False,
        )

    @torch.inference_mode()
    def select_draft_path(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy selector walk over the drafter's top-k candidates.

        hidden_states: (n_blocks * steps, hidden) drafter output for the
        MASK rows only (positions 1..block-1 of each block, block-major) --
        exactly the speculator's sample rows. anchor_token_ids: (n_blocks,)
        the last verified token per request; the anchor contributes only its
        A-table embedding, never its hidden state. Returns
        (n_blocks, steps) selected draft token ids.

        Greedy only (the serving path is greedy): scores for position t are
        S(a, b) = U_t(b) + <A(a) * H(h_t), B(b)> and only the realized
        predecessor's row is computed, so the walk is `steps` small batched
        steps. Temperature sampling with kept 16-way distributions is the
        follow-up needed for lossless non-greedy rejection sampling.
        """
        n_blocks = anchor_token_ids.shape[0]
        tokens, hidden = hidden_states.shape
        assert tokens % n_blocks == 0, (tokens, n_blocks)
        steps = tokens // n_blocks

        logits = self.compute_logits(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        top_vals, top_ids = logits.topk(self.selector_top_k, dim=-1)
        top_vals = top_vals.view(n_blocks, steps, -1)
        top_ids = top_ids.view(n_blocks, steps, -1)

        gate = self.selector_hidden(hidden_states).view(n_blocks, steps, -1)

        chosen = []
        prev = self.selector_predecessor(anchor_token_ids)  # (n_blocks, rank)
        for pos in range(steps):
            succ = self.selector_successor(top_ids[:, pos])  # (nb, k, rank)
            cond = (prev * gate[:, pos]).unsqueeze(-1)  # (nb, rank, 1)
            scores = torch.bmm(succ.to(cond.dtype), cond).squeeze(-1)  # (nb, k)
            scores = scores + top_vals[:, pos].to(scores.dtype)
            idx = scores.argmax(dim=-1)  # (nb,)
            tok = top_ids[:, pos].gather(1, idx.unsqueeze(-1)).squeeze(-1)
            chosen.append(tok)
            prev = self.selector_predecessor(tok)
        return torch.stack(chosen, dim=1)  # (n_blocks, steps)

    @torch.inference_mode()
    def select_draft_path_sampled(
        self,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        temperature: torch.Tensor,
        idx_mapping: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sampled selector walk (llama.cpp PR #27342 host walk, dflash2).

        Per position the walk scores the top-k candidates
        ``S(a, b) = U(b) + <A(a) * H(h), B(b)>`` against the realized
        predecessor, forms ``q = softmax(S / T)`` over the k candidates,
        samples the successor from ``q``, and KEEPS the k-way distribution:
        the candidate ids' rows of ``draft_logits`` receive ``S / T``
        (everything else -inf), so ``softmax(draft_logits)`` reproduces
        ``q`` exactly and rejection sampling stays lossless. This mirrors
        the gumbel draft path's contract of storing temperature-applied
        logits (spec_decode/speculator.py::sample_draft).

        Requests with temperature <= 0 take the greedy argmax walk (the
        rejection sampler verifies those by token equality and ignores
        ``draft_logits``).

        The categorical draw uses device gumbel-max noise. On MPS this is
        unkeyed (no stateless Philox primitive), matching the platform's
        target-sampling fallback; per-request seed parity remains a native
        CUDA/HIP property. Draft noise does not affect the output
        distribution after rejection sampling.

        Args:
            hidden_states: (n_blocks * steps, hidden) drafter output for the
                MASK rows only, block-major (the speculator's sample rows).
            anchor_token_ids: (n_blocks,) last verified token per request.
            temperature: (max_num_reqs,) per-request-state temperatures.
            idx_mapping: (n_blocks,) request-state row per block.
            draft_logits: (max_num_reqs, steps, vocab) fp32; the blocks'
                rows are rewritten in place.

        Returns:
            (n_blocks, steps) sampled draft token ids.
        """
        n_blocks = anchor_token_ids.shape[0]
        tokens, hidden = hidden_states.shape
        assert tokens % n_blocks == 0, (tokens, n_blocks)
        steps = tokens // n_blocks

        logits = self.compute_logits(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        top_vals, top_ids = logits.topk(self.selector_top_k, dim=-1)
        top_vals = top_vals.view(n_blocks, steps, -1)
        top_ids = top_ids.view(n_blocks, steps, -1)

        gate = self.selector_hidden(hidden_states).view(n_blocks, steps, -1)

        req_rows = idx_mapping.to(torch.long)
        temps = temperature[req_rows].to(torch.float32)  # (n_blocks,)
        greedy = temps <= 0
        safe_t = torch.where(greedy, torch.ones_like(temps), temps)

        # Sparse k-way distributions: candidate ids carry S/T, rest -inf.
        draft_logits[req_rows] = float("-inf")

        chosen = []
        prev = self.selector_predecessor(anchor_token_ids)  # (n_blocks, rank)
        for pos in range(steps):
            succ = self.selector_successor(top_ids[:, pos])  # (nb, k, rank)
            cond = (prev * gate[:, pos]).unsqueeze(-1)  # (nb, rank, 1)
            scores = torch.bmm(succ.to(cond.dtype), cond).squeeze(-1)  # (nb, k)
            scores = scores + top_vals[:, pos].to(scores.dtype)
            scaled = scores.to(torch.float32) / safe_t.unsqueeze(-1)
            draft_logits[req_rows.unsqueeze(1), pos, top_ids[:, pos]] = scaled

            gumbel = -torch.empty_like(scaled).exponential_().log()
            perturbed = torch.where(greedy.unsqueeze(-1), scaled, scaled + gumbel)
            idx = perturbed.argmax(dim=-1)  # (nb,)
            tok = top_ids[:, pos].gather(1, idx.unsqueeze(-1)).squeeze(-1)
            chosen.append(tok)
            prev = self.selector_predecessor(tok)
        chosen_t = torch.stack(chosen, dim=1)  # (n_blocks, steps)
        _dump_draft_record(
            {
                "anchor_ids": anchor_token_ids.cpu(),
                "top_ids": top_ids.cpu(),
                "top_vals": top_vals.float().cpu(),
                "chosen": chosen_t.cpu(),
                "temps": temps.cpu(),
            }
        )
        return chosen_t

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # Names are already HF-style from the GGUF adapter.
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
        loaded.add("model.mask_embedding")
        self.model._build_fused_kv_buffers()
        return loaded
