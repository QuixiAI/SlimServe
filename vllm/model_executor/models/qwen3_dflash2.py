# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 draft model (z-lab DFlash 2): non-causal block-attention drafter.

Five sliding-window qwen3 layers + rank-r candidate selector + two-tap
grouped dynamic conv over target hidden-state taps; block_size 8 drafts
7 tokens per verify. Serves as the `dflash` speculative method's draft
model for Qwen3.8-27B (upstream vLLM PR #52816, github.com/z-lab/dflash).
"""

import os
from collections.abc import Callable, Iterable
from functools import cache

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models.muse_glimmer_dflash import MuseGlimmerDFlashModel
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)

# QWEN38_DFLASH_DUMP=<dir> to record per-propose drafter candidates (GGUF
# drafter stack below).
_DUMP_DIR = os.environ.get("QWEN38_DFLASH_DUMP")
_DUMP_STEP = [0]
_FUSED_CONV = os.environ.get("VLLM_QWEN38_FUSED_DFLASH2_CONV", "1") != "0"
_FUSED_CONV_AVAILABLE: bool | None = None


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """FlashInfer's radix top-k, or None for torch.topk.

    This top-k spans the vocabulary and is the selector's largest single cost,
    where the radix kernel is about twice torch.topk.
    """
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
        logger.info_once(
            "flashinfer is unavailable; the DFlash2 selector uses torch.topk, "
            "at roughly half the speed."
        )
        return None
    from flashinfer import top_k

    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    return impl(scores, k, sorted=True, deterministic=True)


@cache
def _dflash_conv_kernel_available() -> bool:
    """Fused Metal grouped-conv route (VLLM_QC_DFLASH_CONV=0 restores the
    eager chain). The eager form is ~10 dispatches per call x 4 calls per
    drafter layer — the largest single encode block in the drafter step."""
    import os

    if os.environ.get("VLLM_QC_DFLASH_CONV", "1") == "0":
        return False
    if not current_platform.is_metal():
        return False
    try:
        from vllm.quixicore import quixicore_ops

        if not quixicore_ops.is_available():
            return False
        import vllm._quixicore_C as qc

        if not hasattr(qc, "qc_dflash_conv"):  # stale .so guard
            return False
    except (ImportError, AttributeError):
        return False
    return True


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    if (
        _dflash_conv_kernel_available()
        and hidden_states.is_contiguous()
        and delta.stride(2) == 1
        and delta.stride(1) == delta.size(2)
        and base.is_contiguous()
    ):
        from vllm.quixicore import quixicore_ops

        return quixicore_ops.qc_dflash_conv(hidden_states, delta, base, block_size)
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(DFlashQwen3DecoderLayer):
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
        draft_config = config.dflash_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            # Query tokens per request: the bonus token plus the mask tokens.
            block_size=1 + speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    decoder_layer_cls = DFlash2Qwen3DecoderLayer

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )
        # The selector carries its own @support_torch_compile, but it is built
        # while the active model tag is still the draft's, so without a tag of its
        # own the two share a compile-cache namespace and the selector loads the
        # draft's graph -- a different input signature, within the same startup.
        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    model_cls = DFlash2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
            CompressedTensorsLinearMethod,
        )

        # Upstream requires an unquantized lm_head here. We additionally accept
        # the compressed-tensors method: on the Metal serving stack the target's
        # fp8-channel lm_head applies through the same deterministic quixicore
        # GEMV that computes serving logits every step, and greedy losslessness
        # is enforced by verification regardless of drafter exactness -- the
        # candidate list only has to be good, not bitwise-unquantized.
        if not isinstance(
            self.lm_head.quant_method,
            (
                UnquantizedEmbeddingMethod,
                UnquantizedLinearMethod,
                CompressedTensorsLinearMethod,
            ),
        ):
            raise ValueError(
                "DFlash2 requires an unquantized or fp8-channel target LM head "
                f"for candidate TopK; got {type(self.lm_head.quant_method).__name__}."
            )

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, selector.top_k)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, selector.top_k)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values


EntryClass = DFlash2Qwen3ForCausalLM


# --------------------------------------------------------------------------- #
# GGUF-built DFlash2 drafter (arch "DFlash2QwenDraftModel"), vendored from the
# Muse-Glimmer-shaped dflash backbone. The safetensors drafter above (arch
# "DFlash2DraftModel") and this stack coexist: they serve the same published
# drafter through the HF and GGUF checkpoints respectively.
# --------------------------------------------------------------------------- #


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
    global _FUSED_CONV_AVAILABLE
    if (
        _FUSED_CONV
        and x.device.type == "mps"
        and x.is_contiguous()
        and coeffs.is_contiguous()
        and base.is_contiguous()
        and x.dtype == coeffs.dtype == base.dtype
    ):
        if _FUSED_CONV_AVAILABLE is None:
            from vllm.quixicore.ops import _qc

            _FUSED_CONV_AVAILABLE = hasattr(_qc(), "dflash2_two_tap_conv")
        if _FUSED_CONV_AVAILABLE:
            from vllm.quixicore.ops import quixicore_ops

            return quixicore_ops.dflash2_two_tap_conv(
                x, coeffs, base, side, bs, group_size
            )

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

        Greedy reference path: scores for position t are
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
        seeds: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sampled selector walk (llama.cpp PR #27342 host walk, dflash2).

        ``seeds`` ((n_blocks,) per-request seeds) and ``positions``
        ((n_blocks, steps) absolute positions of the drafted tokens) key the
        categorical draws through the stateless (seed, pos) uniforms, so
        same-seed repeats reproduce on Metal; without them the draw falls
        back to the device RNG.

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

        The categorical draw uses Gumbel-max noise. On Metal it is keyed by
        (request seed, absolute draft position, candidate column) through the
        stateless splitmix64 stream, so same-seed requests reproduce across
        repeats and boots. Draft noise does not affect the output distribution
        after rejection sampling.

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

            if seeds is not None and positions is not None:
                from vllm.v1.worker.gpu.sample.gumbel import stateless_uniform_2d

                u = stateless_uniform_2d(seeds, positions[:, pos], scaled.shape[-1])
                gumbel = -torch.log(-torch.log(u)).to(scaled.dtype)
            else:
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
