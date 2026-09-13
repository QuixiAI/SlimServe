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
- mHC: the fork's fused MHCPreOp / MHCFusedPostPreOp / MHCPostOp
  (quixicore dsv4_mhc_* kernels on Ampere; identical math, DSV4's
  ``hc_post_alpha=2.0`` == the reference's ``2*sigmoid``).

The DSA layers run SPARSE through the pooled indexer
(vllm/model_executor/layers/glm5_next_indexer.py) and the quixicore NoPE
sparse MLA kernel. The MTP head (layer 45) and video inputs are later
phases.
"""

from collections.abc import Iterable

from typing import ClassVar, Literal

import dataclasses
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.platforms import current_platform
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
import vllm.model_executor.layers.glm5_next_mhc_ops  # noqa: F401  (registers torch.ops.vllm.glm5_mhc_*)
from vllm.model_executor.layers.glm5_next_mhc_project import (
    plain_bf16_projection,
    prepare_router_projection,
    projection_enabled,
    register_projection_stream,
    router_projection_enabled,
)
from vllm.model_executor.layers.glm5_next_indexer import (
    Glm5NextPooledIndexer,
)
from vllm.model_executor.layers.glm5_next_indexer_workspace import (
    Glm5NextIndexerWorkspace,
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
    SupportsEagle3,
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

F32_OVERRIDES_ENV = "SLIMSERVE_F32_OVERRIDES"


def _load_f32_overrides(model_path: str | None) -> dict[str, torch.Tensor]:
    """Tensors to substitute for the checkpoint's copies, keyed by HF name.

    Some NVFP4 conversions of GLM-5.3-Flash (RedHatAI's, 2026-08) saved the
    router bias (``mlp.gate.e_score_correction_bias``), the KDA decay
    tensors (``A_log``, ``dt_bias``) and the mHC base/scale vectors in BF16
    although the native checkpoint keeps them in F32 and this model holds
    them as F32 parameters. ``slimserve.f32_overrides`` rebuilds them from
    the native checkpoint into ``<model>/f32-overrides.safetensors``; when
    that file is present it wins over the shard copies. Set
    ``SLIMSERVE_F32_OVERRIDES=0`` to serve the shard copies for an A/B.
    """
    import os

    from slimserve.f32_overrides import OVERRIDES_FILE

    if not model_path or os.environ.get(F32_OVERRIDES_ENV, "1") == "0":
        return {}
    path = os.path.join(model_path, OVERRIDES_FILE)
    if not os.path.isfile(path):
        return {}
    from safetensors.torch import load_file

    overrides = load_file(path)
    bad = [k for k, v in overrides.items() if v.dtype != torch.float32]
    if bad:
        raise ValueError(
            f"{path}: {len(bad)} override tensors are not float32, e.g. {bad[0]}"
        )
    logger.info("glm5_next: %d F32 override tensors from %s", len(overrides), path)
    return overrides


def iter_with_overrides(
    weights: Iterable[tuple[str, torch.Tensor]],
    overrides: dict[str, torch.Tensor],
    extras: dict[str, torch.Tensor] | None = None,
    strict: bool = False,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield ``weights`` with any name present in ``overrides`` replaced, then
    the ``extras`` (tensors the checkpoint stream does not carry, such as the
    block scales of a swapped FP8 weight). ``strict`` turns an override that
    never appeared into an error: for the FP8 swap-set a missing substitution
    would let the loader copy a BF16 shard into an FP8 parameter."""
    if not overrides and not extras:
        yield from weights
        return
    pending = set(overrides)
    for name, weight in weights:
        if name in overrides:
            pending.discard(name)
            yield name, overrides[name]
        else:
            yield name, weight
    if pending:
        msg = (
            f"glm5_next: {len(pending)} override tensors never appeared in the "
            f"checkpoint stream, e.g. {sorted(pending)[0]}"
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    if extras:
        yield from extras.items()


def _load_fp8_swapset(
    model_path: str | None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(weights to substitute, block scales to inject) from the FP8 swap-set
    ``slimserve.fp8_swapset`` wrote next to the checkpoint; empty when the
    files are absent or ``SLIMSERVE_FP8_SWAPSET=0``. The quantization config
    group that makes those modules block-FP8 is added by the model loader
    from the same manifest, so the two cannot disagree."""
    import json
    import os

    from slimserve.fp8_swapset import manifest_path

    path = manifest_path(model_path)
    if path is None:
        return {}, {}
    with open(path) as fh:
        manifest = json.load(fh)
    from safetensors.torch import load_file

    # The tensor file shares the manifest's stem, so a sidecar selected by
    # stem (SLIMSERVE_FP8_SWAPSET=<stem>) reads its own weights.
    file = os.path.splitext(path)[0] + ".safetensors"
    tensors = load_file(file)
    if sorted(tensors) != sorted(manifest["tensors"]):
        raise ValueError(f"{file}: tensor names do not match the manifest {path}")
    subs = {k: v for k, v in tensors.items() if k.endswith(".weight")}
    extras = {k: v for k, v in tensors.items() if k.endswith(".weight_scale")}
    bad = [k for k, v in subs.items() if v.dtype != torch.float8_e4m3fn]
    bad += [k for k, v in extras.items() if v.dtype != torch.float32]
    if bad or len(subs) + len(extras) != len(tensors):
        raise ValueError(
            f"{file}: expected float8_e4m3fn weights and float32 weight_scale "
            f"tensors only, e.g. {(bad or sorted(tensors))[0]}"
        )
    logger.info(
        "glm5_next: FP8 swap-set from %s: %d weights, %d block scales",
        file,
        len(subs),
        len(extras),
    )
    return subs, extras


class Glm5NextMLAAttention(nn.Module):
    """NoPE MLA for the DeepSeek-sparse-attention layers, sparse through
    the pooled indexer."""

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        indexer_workspace: Glm5NextIndexerWorkspace | None = None,
    ) -> None:
        super().__init__()
        cache_config = vllm_config.cache_config
        extra = vllm_config.additional_config or {}
        if extra.get("glm5_next_main_kv_fp8", False):
            # Per-layer fp8 (e4m3) main KV for the sparse NoPE MLA layers only:
            # the KDA state, the pooled indexer cache and a block drafter's
            # own attention keep the engine-wide dtype. The sparse backend
            # decodes fp8 in software on sm80 (mla_decode_fp8_sparse_nope).
            cache_config = dataclasses.replace(cache_config, cache_dtype="fp8")
            logger.info_once("glm5_next: sparse MLA layers use fp8 (e4m3) main KV")
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
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
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

        assert topk_indices_buffer is not None
        self.indexer = Glm5NextPooledIndexer(
            vllm_config,
            config,
            quant_config=quant_config,
            cache_config=cache_config,
            topk_indices_buffer=topk_indices_buffer,
            prefix=f"{prefix}.indexer",
            workspace=indexer_workspace,
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
            indexer=self.indexer,
            is_sparse=True,
            topk_indices_buffer=topk_indices_buffer,
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

    Uses the owned mHC ops (split SIMT Triton for small batches, QuixiCore
    CUDA for larger prefill): the first site mixes the expanded streams;
    later sites combine the previous sublayer's post/comb placement with
    the next site's pre-mix. RMSNorm follows the BF16 rounding boundary
    inside the transition. The layer returns its FFN output with the
    placement deferred to the next layer (or the model's final
    ``MHCPostOp``). hc parameters are flat on the layer, matching the
    checkpoint's ``hc_{attn,ffn}_{fn,base,scale}`` names.
    """

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        indexer_workspace: Glm5NextIndexerWorkspace | None = None,
        mhc_stream_key: str = "",
        enable_router_projection: bool = False,
    ) -> None:
        super().__init__()
        quant_config = vllm_config.quant_config
        self.layer_idx = int(prefix.rsplit(".", 1)[1])
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self._fuse_mhc_norm = (
            current_platform.is_cuda() and current_platform.is_device_capability((8, 0))
        )
        self.is_linear = config.layer_types[self.layer_idx] == "linear_attention"

        if self.is_linear:
            from slimserve.fp8_swapset import beta_block_rows

            self.self_attn = KimiGatedDeltaNetAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
                fuse_gate_a=True,
                beta_block_rows=beta_block_rows(
                    vllm_config.model_config.model,
                    f"{prefix}.self_attn.in_proj_qkvgfab",
                ),
            )
        else:
            self.self_attn = Glm5NextMLAAttention(
                config,
                vllm_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
                indexer_workspace=indexer_workspace,
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
        self.post_attention_layernorm = RMSNorm(self.hidden_size, config.rms_norm_eps)

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

        self._mhc_stream_key = mhc_stream_key
        self._overlap_kda = (
            bool(mhc_stream_key)
            and self._fuse_mhc_norm
            and self.is_linear
            and plain_bf16_projection(self.self_attn.in_proj_qkvgfab)
        )
        self._overlap_router = (
            bool(mhc_stream_key)
            and self._fuse_mhc_norm
            and enable_router_projection
            and isinstance(self.mlp, DeepseekV2MoE)
            and prepare_router_projection(self.mlp)
        )

    def _site_pre(self, residual, fn, scale, base, norm_weight):
        post_mix, res_mix, x = torch.ops.vllm.glm5_mhc_pre(
            residual,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight if self._fuse_mhc_norm else None,
            self.rms_norm_eps,
        )
        return residual, post_mix, res_mix, x

    def _site_fused(self, x, residual, post_mix, res_mix, fn, scale, base, norm_weight):
        return torch.ops.vllm.glm5_mhc_fused_post_pre(
            x,
            residual,
            post_mix,
            res_mix,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            norm_weight if self._fuse_mhc_norm else None,
            self.rms_norm_eps,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = None
        if residual is None:
            # First layer: x is the expanded [T, hc_mult, D] stream tensor.
            residual, post_mix, res_mix, x = self._site_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm.weight,
            )
        elif self._overlap_kda:
            residual, post_mix, res_mix, x, projected = (
                torch.ops.vllm.glm5_mhc_project_runtime(
                    x,
                    residual,
                    post_mix,
                    res_mix,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.input_layernorm.weight,
                    self.self_attn.in_proj_qkvgfab.weight,
                    self._mhc_stream_key,
                    False,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    self.rms_norm_eps,
                )
            )
        else:
            residual, post_mix, res_mix, x = self._site_fused(
                x,
                residual,
                post_mix,
                res_mix,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm.weight,
            )
        # Only the measured SM80 path fuses norm inside the mHC transition.
        if not self._fuse_mhc_norm:
            x = self.input_layernorm(x)
        if self.is_linear:
            attn_out = torch.empty_like(x)
            if projected is None:
                self.self_attn(x, positions, attn_out)
            else:
                self.self_attn(x, positions, attn_out, projected_qkvgfab=projected)
        else:
            attn_out = self.self_attn(positions, x)

        router_logits = None
        if self._overlap_router:
            residual, post_mix, res_mix, x, router_logits = (
                torch.ops.vllm.glm5_mhc_project_runtime(
                    attn_out,
                    residual,
                    post_mix,
                    res_mix,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.post_attention_layernorm.weight,
                    self.mlp.gate.weight,
                    self._mhc_stream_key,
                    True,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    self.rms_norm_eps,
                )
            )
        else:
            residual, post_mix, res_mix, x = self._site_fused(
                attn_out,
                residual,
                post_mix,
                res_mix,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm.weight,
            )
        if not self._fuse_mhc_norm:
            x = self.post_attention_layernorm(x)
        if router_logits is None:
            x = self.mlp(x)
        else:
            x = self.mlp(x, router_logits=router_logits)
        return x, residual, post_mix, res_mix


@support_torch_compile
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
        # Shared top-k index buffer for every DSA layer: index_topk expanded
        # from pools plus the (index_kpool - 1)-token tail, padded to 32.
        width = config.index_topk + config.index_kpool - 1
        width = (width + 31) // 32 * 32
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            width,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.indexer_workspace = Glm5NextIndexerWorkspace.from_config(
            vllm_config.additional_config
        )
        # One stream per model/PP-stage, never global. The opaque operation
        # joins before return, so sequential layers can safely share it.
        self.mhc_projection_stream = None
        mhc_stream_key = ""
        if projection_enabled(
            vllm_config.additional_config,
            sm80=current_platform.is_cuda()
            and current_platform.is_device_capability((8, 0)),
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            dtype=vllm_config.model_config.dtype,
            lora=vllm_config.lora_config is not None,
        ):
            self.mhc_projection_stream = torch.cuda.Stream()
            mhc_stream_key = maybe_prefix(prefix, "mhc_projection_stream")
            register_projection_stream(
                vllm_config.compilation_config,
                mhc_stream_key,
                self.mhc_projection_stream,
            )
        enable_router_projection = router_projection_enabled(
            vllm_config.additional_config
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm5NextDecoderLayer(
                config,
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                indexer_workspace=self.indexer_workspace,
                mhc_stream_key=mhc_stream_key,
                enable_router_projection=enable_router_projection,
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )
        if self.mhc_projection_stream is not None:
            logger.info(
                "GLM mHC projection overlap enabled: %d KDA layers, %d router sites "
                "(first-layer pre-only transition remains unchanged)",
                sum(getattr(layer, "_overlap_kda", False) for layer in self.layers),
                sum(getattr(layer, "_overlap_router", False) for layer in self.layers),
            )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )
        # EAGLE-3 / DFlash convention: a value v in this tuple captures the
        # completed output of layer v - 1 (set_eagle3_aux_hidden_state_layers
        # passes the drafter's target_layer_ids + 1).
        self.aux_hidden_state_layers: tuple[int, ...] = ()

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
        aux_hidden_states: list[torch.Tensor] = []
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            x, residual, post_mix, res_mix = layer(
                x, positions, residual, post_mix, res_mix
            )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                # DFlash / EAGLE-3 tap: the completed output of this layer.
                # The reference layer output is the four-stream mHC tensor
                # post_ffn * mlp_out + comb_ffn^T residual (what the next
                # transition would consume); the drafters were trained on
                # its hc_contract, the mean over the streams (SGLang PR
                # 36708, sglang.kernels.ops.layernorm.mhc.hc_contract).
                taps = torch.ops.vllm.glm5_mhc_post(x, residual, post_mix, res_mix)
                aux_hidden_states.append(taps.mean(dim=1))
        assert layer is not None
        streams = torch.ops.vllm.glm5_mhc_post(x, residual, post_mix, res_mix)
        # HyperHead: unweighted mean over the streams (reference; DSV4's
        # weighted head does not apply).
        hidden_states = streams.mean(dim=1)
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


# EAGLE-3 / DFlash auxiliary hidden-state taps (V2 speculators call
# set_eagle3_aux_hidden_state_layers on the top-level model).
_DFLASH_DEFAULT_TAPS = (
    6,
    15,
    25,
    34,
    43,
)  # incoai/GLM-5.3-Flash-DFlash2 target_layer_ids + 1


class _Glm5NextAuxTaps:
    """Listed BEFORE SupportsEagle3 in the bases: the protocol ships concrete
    defaults that expect an EagleModelMixin ``self.model``."""

    supports_eagle3: ClassVar[Literal[True]] = True

    def _text_model(self) -> "Glm5NextTextModel":
        lm = getattr(self, "language_model", None)
        return lm.model if lm is not None else self.model

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self._text_model().aux_hidden_state_layers = tuple(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return _DFLASH_DEFAULT_TAPS


class Glm5NextForCausalLM(
    nn.Module, _Glm5NextAuxTaps, HasInnerState, IsHybrid, SupportsPP, SupportsEagle3
):
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
        # KDA merged input projection: [q, k, v, b(beta), f_a, g_a]
        ("in_proj_qkvgfab", "q_proj", 0),
        ("in_proj_qkvgfab", "k_proj", 1),
        ("in_proj_qkvgfab", "v_proj", 2),
        ("in_proj_qkvgfab", "b_proj", 3),
        ("in_proj_qkvgfab", "f_a_proj", 4),
        ("in_proj_qkvgfab", "g_a_proj", 5),
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
        self._model_path = vllm_config.model_config.model
        self.model = Glm5NextTextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
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
            conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"],
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
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
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
        model_path = self._model_path
        weights = iter_with_overrides(weights, _load_f32_overrides(model_path))
        fp8_weights, fp8_scales = _load_fp8_swapset(model_path)
        weights = iter_with_overrides(weights, fp8_weights, fp8_scales, strict=True)
        for name, weight in weights:
            # Vision tower and MTP layer: later phases.
            if name.startswith("model.visual."):
                continue
            if f"layers.{num_text_layers}." in name:
                continue
            for pref, new in self.hf_to_vllm_prefix.items():
                if name.startswith(pref):
                    name = new + name[len(pref) :]
                    break
            else:
                continue

            is_expert = ".mlp.experts." in name
            mapped = False
            if not is_expert:
                for target, ckpt_name, shard_id in self.stacked_params_mapping:
                    token = f".{ckpt_name}."
                    if token not in name and not name.endswith(f".{ckpt_name}"):
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
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(param, weight)
            else:
                param.data.copy_(weight)
            loaded.add(name)
        return loaded


# ============================================================ multimodal


from collections.abc import Mapping, Sequence  # noqa: E402
from typing import Any  # noqa: E402

from transformers.models.glm5_next import (  # noqa: E402
    Glm5NextConfig,
    Glm5NextImageProcessor,
    Glm5NextProcessor,
)
from transformers.models.glm5_next.image_processing_glm5_next import (  # noqa: E402
    smart_resize as glm5_next_smart_resize,
)

from vllm.model_executor.models.glm5_next_vision import (  # noqa: E402
    Glm5NextVisionTransformer,
)
from vllm.model_executor.models.interfaces import (  # noqa: E402
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen2_5_vl import (  # noqa: E402
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
)
from vllm.model_executor.models.qwen2_vl import (  # noqa: E402
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from vllm.model_executor.models.utils import (  # noqa: E402
    _merge_multimodal_embeddings,
)
from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: E402
from vllm.config.multimodal import BaseDummyOptions  # noqa: E402
from vllm.inputs import MultiModalDataDict  # noqa: E402
from vllm.multimodal.inputs import (  # noqa: E402
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems  # noqa: E402
from vllm.multimodal.processing import (  # noqa: E402
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)

_IMAGE_MARKUP = "<|begin_of_image|><|image|><|end_of_image|>"


class Glm5NextProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Glm5NextConfig)

    def get_hf_processor(self, **kwargs: object) -> Glm5NextProcessor:
        return self.ctx.get_hf_processor(Glm5NextProcessor, **kwargs)

    def get_image_processor(self, **kwargs: object) -> Glm5NextImageProcessor:
        return self.get_hf_processor(**kwargs).image_processor

    def get_data_parser(self):
        return Qwen2VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        return {"image": self.get_max_image_tokens()}

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 1,
        do_resize: bool = True,
        image_processor,
        mm_kwargs: Mapping[str, object],
    ) -> tuple[ImageSize, int]:
        vc = self.get_hf_config().vision_config
        patch, merge, tp = vc.patch_size, vc.spatial_merge_size, vc.temporal_patch_size
        min_tok = getattr(image_processor, "min_image_tokens", 16)
        max_tok = getattr(image_processor, "max_image_tokens", 8000)
        merged = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        min_tok = merged.get("min_image_tokens", min_tok)
        max_tok = merged.get("max_image_tokens", max_tok)
        if do_resize:
            # The GLM processor resizes to fit, then PADS to the aligned
            # canvas; the grid is the padded canvas.
            h, w = glm5_next_smart_resize(
                num_frames=tp,
                height=image_height,
                width=image_width,
                temporal_factor=tp,
                factor=patch * merge,
                min_pixels=min_tok,
                max_pixels=max_tok,
            )
            size = ImageSize(width=w, height=h)
        else:
            size = ImageSize(width=image_width, height=image_height)
        grid_t = max((num_frames + (-num_frames % tp)) // tp, 1)
        n_tokens = grid_t * (size.height // patch) * (size.width // patch) // (merge**2)
        return size, n_tokens

    def get_image_size_with_most_features(self, max_pixels=None) -> ImageSize:
        vc = self.get_hf_config().vision_config
        image_processor = self.get_image_processor()
        max_tok = getattr(image_processor, "max_image_tokens", 8000)
        side_patches = int((max_tok * vc.spatial_merge_size**2) ** 0.5)
        side = side_patches * vc.patch_size
        return ImageSize(width=side, height=side)


class Glm5NextDummyInputsBuilder(BaseDummyInputsBuilder[Glm5NextProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return _IMAGE_MARKUP * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        w, h = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=w,
                height=h,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        }


class Glm5NextMultiModalProcessor(BaseMultiModalProcessor[Glm5NextProcessingInfo]):
    def _get_mm_fields_config(
        self, hf_inputs, hf_processor_mm_kwargs
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen2vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        merge_len = image_processor.merge_size**2
        image_token_id = hf_config.image_token_id

        def get_replacement(item_idx: int):
            grid = out_mm_kwargs["image"][item_idx]["image_grid_thw"].data
            assert isinstance(grid, torch.Tensor)
            return [image_token_id] * (int(grid.prod()) // merge_len)

        return [
            PromptReplacement(
                modality="image", target=[image_token_id], replacement=get_replacement
            )
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Glm5NextMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm5NextDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    nn.Module,
    _Glm5NextAuxTaps,
    SupportsMultiModal,
    SupportsPP,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
):
    """GLM-5.3-Flash: vision tower + hybrid text backbone."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return _IMAGE_MARKUP
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        with self._mark_tower_model(vllm_config, "image"):
            self.visual = Glm5NextVisionTransformer(
                config.vision_config,
                quant_config=None,  # tower is unquantized in the checkpoint
                prefix=maybe_prefix(prefix, "visual"),
            )
        self.language_model = Glm5NextForCausalLM(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model")
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # --- hybrid state (delegated) ---
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    # --- multimodal ---
    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_image_input(self, **kwargs) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None:
            return None
        return Qwen2_5_VLImagePixelInputs(
            type="pixel_values",
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_image_input(**kwargs)
        if image_input is None:
            return []
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        embeds = self.visual(pixel_values, grid_thw=grid_thw)
        merge = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge // merge).tolist()
        return tuple(embeds.split(sizes))

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        is_multimodal = _require_is_multimodal(is_multimodal)
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        text_weights: list[tuple[str, torch.Tensor]] = []
        vis_params = dict(self.visual.named_parameters())
        loaded: set[str] = set()
        vis_stacked = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        for name, weight in weights:
            if not name.startswith("model.visual."):
                text_weights.append((name, weight))
                continue
            vname = name[len("model.visual.") :]
            # merger: gate/up/down live on the merger's mlp submodule
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                vname = vname.replace(f"merger.{leaf}", f"merger.mlp.{leaf}")
            mapped = False
            for target, ckpt, shard in vis_stacked:
                if f".{ckpt}." in vname:
                    tgt = vname.replace(ckpt, target)
                    if tgt in vis_params:
                        p = vis_params[tgt]
                        p.weight_loader(p, weight, shard)
                        loaded.add("visual." + tgt)
                        mapped = True
                    break
            if mapped:
                continue
            if vname not in vis_params:
                logger.warning_once("glm5_next: unmatched vision weight %s", name)
                continue
            p = vis_params[vname]
            loader = getattr(p, "weight_loader", None)
            if loader is not None:
                loader(p, weight)
            else:
                p.data.copy_(weight)
            loaded.add("visual." + vname)
        for n in self.language_model.load_weights(text_weights):
            loaded.add("language_model." + n)
        return loaded
