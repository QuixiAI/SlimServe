# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weights adapter for the ``glm5-next`` architecture (GLM-5.3-Flash).

antirez's ds4-layout conversion: llama.cpp ``blk.N.*`` names with the DeepSeek
MLA / indexer spellings from the ``glm-dsa`` converter plus ``kda_*`` for the
Kimi-Delta-Attention layers and ``hc_*`` for the hyper-connections. Same
reason as `glm_dsa` for carrying the map explicitly: transformers' GGUF reader
and ``gguf.get_tensor_name_map`` know nothing about this layout.

Names are emitted in the HF checkpoint layout (``model.language_model.layers.N``,
``lm_head``) because that is what ``Glm5NextForCausalLM.load_weights`` maps
through its ``hf_to_vllm_prefix`` and ``stacked_params_mapping``; fused modules
are fed through the pre-fusion shard names listed there (``q_proj``/``k_proj``/
``v_proj``/``b_proj``/``f_a_proj``/``g_a_proj`` -> ``in_proj_qkvgfab``,
``{q,k,v}_conv1d`` -> ``conv1d``, ``q_a_proj``/``kv_a_proj_with_mqa`` ->
``fused_qkv_a_proj``, ``gate_proj``/``up_proj`` -> ``gate_up_proj``).

Dtype policy, per tensor group:

* **Routed experts** (IQ2_XXS gate/up, Q2_K down): 3D packed stacks yielded
  whole under the expert-0 name; ``GGUFMoEMethod`` keeps them packed.
* **Q8_0 / Q4_K linears** (KDA q/k/v/beta/f_a/g_a/f_b/g_b/output, MLA q_a/q_b/
  kv_a/output, dense FFN, shared experts, lm_head): packed through the GGUF
  linear method. ``in_proj_qkvgfab`` mixes Q4_K (q, k) with Q8_0 (v, beta,
  f_a, g_a); the merged GGUF parameter records a type per shard.
* **Absorbed MLA** ``attn_k_b`` (H, kv_lora, qk_nope) / ``attn_v_b`` (H, v_head,
  kv_lora), Q8_0: dequantized and re-emitted as one BF16 ``kv_b_proj`` of
  ``[H*(qk_nope+v_head), kv_lora]`` (needs a transpose of ``k_b``). 11 layers x
  32 MiB; ``kv_b_proj`` is marked unquantized.
* **token_embd** Q8_0: dequantized to BF16, because the text model builds
  ``embed_tokens`` without a quant config (glm5_next.py:408).
* **F32 / BF16 tensors** (norms, hc_*, KDA conv/dt_bias/A_log/o_norm, router
  weight and ``exp_probs_b`` bias, every indexer tensor, nextn eh_proj/norms):
  passed through as-is; the loader casts into the parameter dtype. Their
  modules are unquantized by construction.
* **nextn** (block 45): mapped to the MTP names and dropped unless the model
  being loaded is the ``glm5_next_mtp`` draft, in which case only that block
  plus the shared embedding / head is emitted.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import gguf
import gguf.quants
import regex as re
import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.gguf_weight_utils import (
    gguf_quant_weights_iterator_multi,
)
from vllm.transformers_utils.gguf_utils import gguf_architecture, gguf_reader

from .default import GGUFWeightsAdapter

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import ModelConfig

logger = init_logger(__name__)

ARCH = "glm5-next"
_BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

# Checkpoint prefix of the text backbone in the HF layout the model loads.
CKPT_PREFIX = "model.language_model."

# Tensors every block (trunk and nextn) carries.
_COMMON_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}

# mHC: bare fp32 Parameters flat on the layer; trunk blocks only.
_HC_RENAMES = {
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
}

# Dense FFN (leading_dense_block_count layers).
_DENSE_RENAMES = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# MoE layers. The 3D expert stacks map onto the expert-0 slot and are yielded
# whole (RoutedExperts' full_load path slices them in one strided copy).
_MOE_RENAMES = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
    "ffn_gate_exps.weight": "mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.0.down_proj.weight",
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
}

# KDA (linear_attention) blocks. The six in_proj shards and the three conv
# shards are named for their pre-fusion checkpoint names; the loader fuses.
_KDA_RENAMES = {
    "kda_q.weight": "self_attn.q_proj.weight",
    "kda_k.weight": "self_attn.k_proj.weight",
    "kda_v.weight": "self_attn.v_proj.weight",
    "kda_beta.weight": "self_attn.b_proj.weight",
    "kda_f_a.weight": "self_attn.f_a_proj.weight",
    "kda_g_a.weight": "self_attn.g_a_proj.weight",
    "kda_q_conv.weight": "self_attn.q_conv1d.weight",
    "kda_k_conv.weight": "self_attn.k_conv1d.weight",
    "kda_v_conv.weight": "self_attn.v_conv1d.weight",
    "kda_f_b.weight": "self_attn.f_b_proj.weight",
    "kda_g_b.weight": "self_attn.g_b_proj.weight",
    "kda_dt_bias.weight": "self_attn.dt_bias",
    "kda_a_log.weight": "self_attn.A_log",
    "kda_o_norm.weight": "self_attn.o_norm.weight",
    "kda_output.weight": "self_attn.o_proj.weight",
}

# Sparse MLA (deepseek_sparse_attention) blocks, trunk and nextn alike.
_MLA_RENAMES = {
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    # Both absorbed halves map to one destination; prepare_weights fuses them.
    "attn_k_b.weight": "self_attn.kv_b_proj.weight",
    "attn_v_b.weight": "self_attn.kv_b_proj.weight",
    # Pooled indexer (Glm5NextPooledIndexer). wk and weights_proj stay
    # separate modules here (no wk_weights_proj fusion, unlike DeepSeek).
    "indexer.attn_q_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.attn_k.weight": "self_attn.indexer.wk.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.proj.weight": "self_attn.indexer.weights_proj.weight",
    "indexer.pool_ape.weight": "self_attn.indexer.index_kpool_compress_ape",
    "indexer.pool_gate.weight": "self_attn.indexer.index_kpool_compress_gate",
}

# nextn extras. Glm5NextMTP's loader (DeepSeekMTP lineage) takes names in the
# original layout and inserts `.mtp_block.` itself for everything that is not
# enorm / hnorm / eh_proj / shared_head / embed_tokens.
_MTP_RENAMES = {
    "nextn.enorm.weight": "enorm.weight",
    "nextn.hnorm.weight": "hnorm.weight",
    "nextn.eh_proj.weight": "eh_proj.weight",
    "nextn.shared_head_norm.weight": "shared_head.norm.weight",
}

_GLOBAL_RENAMES = {
    "token_embd.weight": CKPT_PREFIX + "embed_tokens.weight",
    "output_norm.weight": CKPT_PREFIX + "norm.weight",
    "output.weight": "lm_head.weight",
}

_KV_B_SUFFIXES = ("attn_k_b.weight", "attn_v_b.weight")
_MTP_TOP_LEVEL = ("embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head")

LINEAR_ATTENTION = "linear_attention"


# --------------------------------------------------------------- pure layout


def kda_in_proj_layout(num_heads: int, head_dim: int) -> list[tuple[str, int, int]]:
    """``(gguf suffix, shard_id, rows)`` of ``in_proj_qkvgfab`` in shard order.

    Mirrors KimiGatedDeltaNetAttention with ``fuse_gate_a=True`` (the way
    glm5_next builds it): ``[q, k, v | beta | f_a | g_a]`` with output sizes
    ``[P, P, P, H, D, D]`` where ``P = H * D``; beta / f_a / g_a are the
    replicated shards.
    """
    proj = num_heads * head_dim
    return [
        ("kda_q.weight", 0, proj),
        ("kda_k.weight", 1, proj),
        ("kda_v.weight", 2, proj),
        ("kda_beta.weight", 3, num_heads),
        ("kda_f_a.weight", 4, head_dim),
        ("kda_g_a.weight", 5, head_dim),
    ]


def kda_conv_layout(
    num_heads: int, head_dim: int, kernel: int
) -> list[tuple[str, int, tuple[int, int, int]]]:
    """``(gguf suffix, shard_id, torch shape)`` of the fused ``conv1d``.

    The fused parameter is ``[3P, 1, K]``; each GGUF conv is ``(K, 1, P)`` in
    ggml order, i.e. ``(P, 1, K)`` torch-ordered, which the fused loader copies
    into rows ``[shard*P, (shard+1)*P)``.
    """
    proj = num_heads * head_dim
    return [
        ("kda_q_conv.weight", 0, (proj, 1, kernel)),
        ("kda_k_conv.weight", 1, (proj, 1, kernel)),
        ("kda_v_conv.weight", 2, (proj, 1, kernel)),
    ]


def assemble_kv_b_proj(
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    qk_nope: int,
    v_head: int,
    kv_lora: int,
) -> torch.Tensor:
    """Rebuild ``kv_b_proj`` from the absorbed per-head MLA weights.

    ``k_b`` is ``(H, kv_lora, qk_nope)``, ``v_b`` is ``(H, v_head, kv_lora)``
    (torch order of ggml ``attn_k_b (qk_nope, kv_lora, H)`` and ``attn_v_b
    (kv_lora, v_head, H)``). ``kv_b_proj`` is ``[H*(qk_nope + v_head), kv_lora]``
    with, per head, the ``qk_nope`` rows of ``k_b^T`` followed by the ``v_head``
    rows of ``v_b``. Returned in BF16.
    """
    k_b = k_b.to(torch.float32)
    v_b = v_b.to(torch.float32)
    heads = k_b.shape[0]
    if tuple(k_b.shape) != (heads, kv_lora, qk_nope):
        raise ValueError(
            f"attn_k_b has shape {tuple(k_b.shape)}, expected "
            f"({heads}, {kv_lora}, {qk_nope})"
        )
    if tuple(v_b.shape) != (heads, v_head, kv_lora):
        raise ValueError(
            f"attn_v_b has shape {tuple(v_b.shape)}, expected "
            f"({heads}, {v_head}, {kv_lora})"
        )
    k_b = k_b.transpose(1, 2).contiguous()  # (H, qk_nope, kv_lora)
    fused = torch.cat([k_b, v_b], dim=1)  # (H, qk_nope + v_head, kv_lora)
    return fused.reshape(heads * (qk_nope + v_head), kv_lora).to(torch.bfloat16)


def _cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def expected_block_shapes(cfg: Any, layer: int) -> dict[str, tuple[int, ...]]:
    """Torch-ordered shape of every tensor block ``layer`` should carry.

    ``cfg`` is the text config (object or dict). Blocks past
    ``num_hidden_layers`` are the nextn head: sparse MLA + MoE, no hc.
    """
    hidden = int(_cfg(cfg, "hidden_size"))
    num_layers = int(_cfg(cfg, "num_hidden_layers"))
    is_mtp = layer >= num_layers
    layer_types = _cfg(cfg, "layer_types")
    mlp_types = _cfg(cfg, "mlp_layer_types")
    if is_mtp:
        is_linear, is_dense = False, False
    else:
        is_linear = layer_types[layer] == LINEAR_ATTENTION
        is_dense = mlp_types[layer] == "dense"

    shapes: dict[str, tuple[int, ...]] = {
        "attn_norm.weight": (hidden,),
        "ffn_norm.weight": (hidden,),
    }
    if not is_mtp:
        hc = int(_cfg(cfg, "hc_mult"))
        mix = (2 + hc) * hc
        for site in ("attn", "ffn"):
            shapes[f"hc_{site}_fn.weight"] = (mix, hc * hidden)
            shapes[f"hc_{site}_base.weight"] = (mix,)
            shapes[f"hc_{site}_scale.weight"] = (3,)

    if is_linear:
        la = _cfg(cfg, "linear_attn_config")
        heads, dim = int(la["num_heads"]), int(la["head_dim"])
        kernel = int(la["short_conv_kernel_size"])
        proj = heads * dim
        for suffix, _shard, rows in kda_in_proj_layout(heads, dim):
            shapes[suffix] = (rows, hidden)
        for suffix, _shard, shape in kda_conv_layout(heads, dim, kernel):
            shapes[suffix] = shape
        shapes["kda_f_b.weight"] = (proj, dim)
        shapes["kda_g_b.weight"] = (proj, dim)
        shapes["kda_dt_bias.weight"] = (proj,)
        shapes["kda_a_log.weight"] = (heads,)
        shapes["kda_o_norm.weight"] = (dim,)
        shapes["kda_output.weight"] = (hidden, proj)
    else:
        heads = int(_cfg(cfg, "num_attention_heads"))
        q_lora = int(_cfg(cfg, "q_lora_rank"))
        kv_lora = int(_cfg(cfg, "kv_lora_rank"))
        qk_nope = int(_cfg(cfg, "qk_nope_head_dim"))
        qk_rope = int(_cfg(cfg, "qk_rope_head_dim"))
        v_head = int(_cfg(cfg, "v_head_dim"))
        idx_heads = int(_cfg(cfg, "index_n_heads"))
        idx_dim = int(_cfg(cfg, "index_head_dim"))
        kpool = int(_cfg(cfg, "index_kpool"))
        shapes.update(
            {
                "attn_q_a.weight": (q_lora, hidden),
                "attn_q_a_norm.weight": (q_lora,),
                "attn_q_b.weight": (heads * (qk_nope + qk_rope), q_lora),
                "attn_kv_a_mqa.weight": (kv_lora + qk_rope, hidden),
                "attn_kv_a_norm.weight": (kv_lora,),
                "attn_output.weight": (hidden, heads * v_head),
                "attn_k_b.weight": (heads, kv_lora, qk_nope),
                "attn_v_b.weight": (heads, v_head, kv_lora),
                "indexer.attn_q_b.weight": (idx_heads * idx_dim, q_lora),
                "indexer.attn_k.weight": (idx_dim, hidden),
                "indexer.k_norm.weight": (idx_dim,),
                "indexer.k_norm.bias": (idx_dim,),
                "indexer.proj.weight": (idx_heads, hidden),
                "indexer.pool_ape.weight": (kpool, idx_dim),
                "indexer.pool_gate.weight": (idx_dim, hidden),
            }
        )

    if is_dense:
        inter = int(_cfg(cfg, "intermediate_size"))
        shapes["ffn_gate.weight"] = (inter, hidden)
        shapes["ffn_up.weight"] = (inter, hidden)
        shapes["ffn_down.weight"] = (hidden, inter)
    else:
        experts = int(_cfg(cfg, "n_routed_experts"))
        moe_inter = int(_cfg(cfg, "moe_intermediate_size"))
        shared = int(_cfg(cfg, "n_shared_experts")) * moe_inter
        shapes.update(
            {
                "ffn_gate_inp.weight": (experts, hidden),
                "exp_probs_b.bias": (experts,),
                "ffn_gate_exps.weight": (experts, moe_inter, hidden),
                "ffn_up_exps.weight": (experts, moe_inter, hidden),
                "ffn_down_exps.weight": (experts, hidden, moe_inter),
                "ffn_gate_shexp.weight": (shared, hidden),
                "ffn_up_shexp.weight": (shared, hidden),
                "ffn_down_shexp.weight": (hidden, shared),
            }
        )

    if is_mtp:
        shapes.update(
            {
                "nextn.eh_proj.weight": (hidden, 2 * hidden),
                "nextn.enorm.weight": (hidden,),
                "nextn.hnorm.weight": (hidden,),
                "nextn.shared_head_norm.weight": (hidden,),
            }
        )
    return shapes


def expected_global_shapes(cfg: Any) -> dict[str, tuple[int, ...]]:
    hidden = int(_cfg(cfg, "hidden_size"))
    vocab = int(_cfg(cfg, "vocab_size"))
    return {
        "token_embd.weight": (vocab, hidden),
        "output_norm.weight": (hidden,),
        "output.weight": (vocab, hidden),
    }


def expected_tensor_shapes(cfg: Any) -> dict[str, tuple[int, ...]]:
    """Every tensor name the file should carry, with its torch-order shape."""
    num_layers = int(_cfg(cfg, "num_hidden_layers"))
    nextn = int(_cfg(cfg, "num_nextn_predict_layers", 0) or 0)
    out = dict(expected_global_shapes(cfg))
    for layer in range(num_layers + nextn):
        for suffix, shape in expected_block_shapes(cfg, layer).items():
            out[f"blk.{layer}.{suffix}"] = shape
    return out


def verify_tensor_shapes(present: dict[str, tuple[int, ...]], cfg: Any) -> None:
    """Fail loudly if the file's tensors disagree with the derived config.

    ``present`` maps tensor name -> torch-ordered shape. Every expected tensor
    must exist with the expected shape and nothing unexpected may be present:
    a silent drop or a shape drift would otherwise surface as a confusing
    loader error deep inside a fused-shard copy.
    """
    expected = expected_tensor_shapes(cfg)
    missing = sorted(set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    wrong = sorted(
        f"{name}: file {tuple(present[name])} != expected {expected[name]}"
        for name in set(expected) & set(present)
        if tuple(present[name]) != tuple(expected[name])
    )
    problems = []
    if missing:
        problems.append(f"{len(missing)} missing, e.g. {missing[:4]}")
    if extra:
        problems.append(f"{len(extra)} unmapped, e.g. {extra[:4]}")
    if wrong:
        problems.append(f"{len(wrong)} wrong shape, e.g. {wrong[:4]}")
    if problems:
        raise ValueError(
            f"{ARCH} GGUF does not match its config: " + "; ".join(problems)
        )


def block_renames(cfg: Any, layer: int) -> dict[str, str]:
    """GGUF suffix -> HF suffix (relative to ``layers.N.``) for block ``layer``."""
    num_layers = int(_cfg(cfg, "num_hidden_layers"))
    renames = dict(_COMMON_RENAMES)
    if layer >= num_layers:
        renames.update(_MLA_RENAMES)
        renames.update(_MOE_RENAMES)
        renames.update(_MTP_RENAMES)
        return renames
    renames.update(_HC_RENAMES)
    if _cfg(cfg, "layer_types")[layer] == LINEAR_ATTENTION:
        renames.update(_KDA_RENAMES)
    else:
        renames.update(_MLA_RENAMES)
    if _cfg(cfg, "mlp_layer_types")[layer] == "dense":
        renames.update(_DENSE_RENAMES)
    else:
        renames.update(_MOE_RENAMES)
    return renames


def build_name_map_for(
    cfg: Any, present: Iterable[str], *, mtp: bool
) -> dict[str, str]:
    """GGUF name -> HF checkpoint name for the tensors in ``present``.

    ``mtp=False``: the trunk (blocks ``< num_hidden_layers``) plus embedding,
    final norm and lm_head. ``mtp=True``: the nextn block only, with the
    embedding and output head re-homed under it the way Glm5NextMTP's loader
    expects (``layers.L.embed_tokens`` -> ``model.embed_tokens``,
    ``layers.L.shared_head.head``).
    """
    num_layers = int(_cfg(cfg, "num_hidden_layers"))
    nextn = int(_cfg(cfg, "num_nextn_predict_layers", 0) or 0)
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in sorted(present):
        match = _BLK_RE.match(name)
        if match is None:
            if name not in _GLOBAL_RENAMES:
                unmapped.append(name)
                continue
            if mtp:
                if name == "token_embd.weight":
                    name_map[name] = (
                        f"{CKPT_PREFIX}layers.{num_layers}.embed_tokens.weight"
                    )
                elif name == "output.weight":
                    name_map[name] = (
                        f"{CKPT_PREFIX}layers.{num_layers}.shared_head.head.weight"
                    )
            else:
                name_map[name] = _GLOBAL_RENAMES[name]
            continue
        layer, suffix = int(match.group(1)), match.group(2)
        if layer >= num_layers + nextn:
            unmapped.append(name)
            continue
        if (layer >= num_layers) != mtp:
            continue
        hf_suffix = block_renames(cfg, layer).get(suffix)
        if hf_suffix is None:
            unmapped.append(name)
            continue
        name_map[name] = f"{CKPT_PREFIX}layers.{layer}.{hf_suffix}"
    if unmapped:
        raise ValueError(
            f"{ARCH} GGUF: {len(unmapped)} tensors have no mapping, e.g. {unmapped[:5]}"
        )
    return name_map


def vllm_module_name(hf_name: str, mtp_layer: int | None) -> str:
    """The vLLM module a checkpoint tensor lands in, as a dotted suffix.

    ``is_layer_skipped_gguf`` matches ``unquantized_modules`` entries as
    substrings of the module prefix, and the model registers its layers as
    ``model.layers.N`` (text-only) or ``language_model.model.layers.N``
    (multimodal wrapper) -- neither contains the checkpoint's
    ``model.language_model.`` prefix, so that prefix is stripped and a
    leading dot is kept to anchor the layer index (``.layers.3.`` cannot match
    ``.layers.13.``). For the MTP draft, block-internal modules live under
    ``layers.L.mtp_block.``.
    """
    module = hf_name
    for suffix in (".weight", ".bias"):
        if module.endswith(suffix):
            module = module[: -len(suffix)]
            break
    if module.startswith(CKPT_PREFIX):
        module = "." + module[len(CKPT_PREFIX) :]
    if mtp_layer is not None:
        head = f".layers.{mtp_layer}."
        if module.startswith(head):
            rest = module[len(head) :]
            if not rest.startswith(_MTP_TOP_LEVEL):
                module = head + "mtp_block." + rest
    return module


# ------------------------------------------------------------------ adapter


class Glm5NextGGUFAdapter(GGUFWeightsAdapter):
    """Adapter for GLM-5.3-Flash's ``glm5-next`` GGUF layout."""

    @classmethod
    def matches(cls, config: PretrainedConfig) -> bool:
        model_type = getattr(config.get_text_config(), "model_type", None)
        return model_type in ("glm5_next", "glm5_next_text", "glm5_next_mtp")

    @staticmethod
    def matches_gguf(gguf_path: str) -> bool:
        return gguf_architecture(gguf_path) == ARCH

    @staticmethod
    def _text_config(model_config: ModelConfig):
        return model_config.hf_config.get_text_config()

    @classmethod
    def _is_mtp_model(cls, model_config: ModelConfig) -> bool:
        return getattr(cls._text_config(model_config), "model_type", None) == (
            "glm5_next_mtp"
        )

    def patch_hf_config(self, model_path: str, hf_config: PretrainedConfig):
        # The config was built from this very file by GGUFConfigParser; the
        # default patcher assumes a gguf arch table that does not exist for
        # glm5-next.
        del model_path
        return hf_config

    def update_tie_word_embeddings(self, model_path, hf_config, gguf_to_hf_name_map):
        # The file always ships a separate output.weight.
        hf_config.update({"tie_word_embeddings": False})

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        text_config = self._text_config(model_config)
        is_mtp = self._is_mtp_model(model_config)
        self._mtp_layer = int(text_config.num_hidden_layers) if is_mtp else None

        present: dict[str, tuple[int, ...]] = {}
        for gguf_file in self._get_all_gguf_files(model_config.model):
            for tensor in gguf_reader(gguf_file).tensors:
                present[tensor.name] = tuple(int(d) for d in reversed(tensor.shape))
        verify_tensor_shapes(present, text_config)

        name_map = build_name_map_for(text_config, present, mtp=is_mtp)
        logger.info(
            "%s GGUF: mapped %d of %d tensors for the %s",
            ARCH,
            len(name_map),
            len(present),
            "MTP draft" if is_mtp else "trunk",
        )
        return name_map

    def get_unquantized_modules(self, weight_type_map: dict[str, str]) -> list[str]:
        """Modules the GGUF linear method must build unquantized.

        Derived from the weight types (F32/BF16 -> unquantized) like the
        default, but spelled as vLLM module suffixes (see
        ``vllm_module_name``), plus the two modules this adapter feeds
        dequantized regardless of the file's type: ``kv_b_proj`` and the
        embedding.
        """
        mtp_layer = getattr(self, "_mtp_layer", None)
        modules: set[str] = set()
        for hf_name, weight_type in weight_type_map.items():
            if not hf_name.endswith(".weight"):
                continue
            if weight_type in ("F32", "F16", "BF16") or hf_name.endswith(
                ("kv_b_proj.weight", "embed_tokens.weight")
            ):
                modules.add(vllm_module_name(hf_name, mtp_layer))
        return sorted(modules)

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        text_config = self._text_config(model_config)
        name_map = self.load_spec.gguf_to_hf_name_map
        qk_nope = int(text_config.qk_nope_head_dim)
        v_head = int(text_config.v_head_dim)
        kv_lora = int(text_config.kv_lora_rank)

        # The absorbed MLA halves and the embedding feed modules vLLM builds
        # unquantized, so they are dequantized here rather than handed over
        # as packed blocks; everything else goes through the shared packed
        # iterator.
        kv_parts: dict[int, dict[str, torch.Tensor]] = {}
        deferred: set[str] = set()
        for gguf_file in self.load_spec.weights_source:
            for tensor in gguf_reader(gguf_file).tensors:
                hf_name = name_map.get(tensor.name)
                if hf_name is None:
                    continue
                match = _BLK_RE.match(tensor.name)
                suffix = match.group(2) if match else tensor.name
                if suffix in _KV_B_SUFFIXES:
                    deferred.add(tensor.name)
                    layer = int(match.group(1))
                    slot = kv_parts.setdefault(layer, {})
                    slot[suffix] = torch.from_numpy(
                        gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                    )
                    if len(slot) == 2:
                        kv_parts.pop(layer)
                        yield (
                            hf_name,
                            assemble_kv_b_proj(
                                slot["attn_k_b.weight"],
                                slot["attn_v_b.weight"],
                                qk_nope,
                                v_head,
                                kv_lora,
                            ),
                        )
                elif tensor.name == "token_embd.weight":
                    deferred.add(tensor.name)
                    value = torch.from_numpy(
                        gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                    )
                    yield hf_name, value.to(torch.bfloat16)

        ordinary = {g: hf for g, hf in name_map.items() if g not in deferred}
        yield from self.map_weights(
            gguf_quant_weights_iterator_multi(self.load_spec.weights_source, ordinary)
        )

        for layer, slot in kv_parts.items():
            raise RuntimeError(
                f"layer {layer}: incomplete MLA kv_b_proj, got {sorted(slot)}"
            )
