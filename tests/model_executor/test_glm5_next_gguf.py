# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline plumbing proofs for the GLM-5.3-Flash ``glm5-next`` GGUF.

Reads only the GGUF header (metadata and tensor info) of antirez's Q2 file,
so nothing here touches weights: ``SLIMSERVE_GLM53F_GGUF`` names the file,
else the registry's cache location of the ``glm53f-gguf`` source (what
``slimserve glm53f-q2-1 -y`` downloads); a header-only sparse copy works too.
Pins:

* the config derived from ``glm5-next.*`` metadata,
* the adapter's name map: every tensor in the file is mapped exactly once
  (the two absorbed MLA halves both landing on ``kv_b_proj`` is the one
  intended double), and every emitted name resolves -- through the model's
  own ``hf_to_vllm_prefix`` / ``stacked_params_mapping`` / expert mapping --
  to a parameter the text model registers under a GGUF quant config,
* shape / shard sanity of the fused KDA projection, the fused conv and the
  ``kv_b_proj`` rebuild.

The model class itself cannot be instantiated here (it needs CUDA and the
native transformers ``glm5_next`` package), so the expected parameter set is
written out from ``glm5_next.py`` / ``kimi_gdn_linear_attn.py`` /
``glm5_next_indexer.py`` and the stacked mapping is parsed from the model
source so the two cannot drift apart silently.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _gguf_path() -> Path:
    env = os.environ.get("SLIMSERVE_GLM53F_GGUF")
    if env:
        return Path(env).expanduser()
    from slimserve import registry

    src = registry._registry()["sources"]["glm53f-gguf"]
    q2 = src["quants"]["Q2"]["files"][0]["path"]
    return registry.cache_root() / src["local_dir"] / q2


HEADER = _gguf_path()
REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "perf/results/2026-09-10/glm53f-q2-discovery/gguf_header_census.txt"
MODEL_SRC = REPO / "vllm/model_executor/models/glm5_next.py"

needs_header = pytest.mark.skipif(
    not HEADER.is_file(),
    reason="GLM-5.3-Flash Q2 GGUF not present "
    "(SLIMSERVE_GLM53F_GGUF or the registry cache)",
)

F32 = {"F32", "F16", "BF16"}


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def cfg():
    from vllm.transformers_utils.gguf_glm5_next import (
        build_glm5_next_config_from_gguf,
    )

    return build_glm5_next_config_from_gguf(str(HEADER))


@pytest.fixture(scope="module")
def text(cfg):
    return cfg.get_text_config()


@pytest.fixture(scope="module")
def tensors():
    """name -> (ggml type name, torch-order shape) from the header file."""
    from vllm.transformers_utils.gguf_utils import gguf_reader

    return {
        t.name: (t.tensor_type.name, tuple(int(d) for d in reversed(t.shape)))
        for t in gguf_reader(str(HEADER)).tensors
    }


def _census_names() -> set[str]:
    names = set()
    for line in CENSUS.read_text().splitlines():
        if line.startswith(("blk.", "output", "token_embd")):
            names.add(line.split()[0])
    return names


# ------------------------------------------------------------------- config


@needs_header
def test_config_from_metadata(cfg, text):
    assert cfg.architectures == ["Glm5NextForConditionalGeneration"]
    assert cfg.model_type == "glm5_next"
    assert text.model_type == "glm5_next_text"
    assert text.num_hidden_layers == 45
    assert text.num_nextn_predict_layers == 1
    assert text.hidden_size == 4096 and text.vocab_size == 154880
    assert text.intermediate_size == 12288 and text.max_position_embeddings == 1048576
    assert text.rms_norm_eps == 1e-5

    dsa = [
        i for i, t in enumerate(text.layer_types) if t == "deepseek_sparse_attention"
    ]
    assert dsa == list(range(3, 45, 4)) and len(dsa) == 11
    assert text.layer_types.count("linear_attention") == 34
    assert text.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 42
    assert text.first_k_dense_replace == 3

    assert (text.n_routed_experts, text.num_experts_per_tok) == (288, 8)
    assert text.n_shared_experts == 1 and text.moe_intermediate_size == 2048
    assert text.routed_scaling_factor == 2.5 and text.norm_topk_prob is True
    assert (text.scoring_func, text.topk_method) == ("sigmoid", "noaux_tc")
    assert (text.n_group, text.topk_group) == (1, 1)
    assert text.swiglu_limit == 10.0 and text.hidden_act == "silu"

    # NoPE MLA
    assert (text.num_attention_heads, text.q_lora_rank, text.kv_lora_rank) == (
        64,
        1536,
        512,
    )
    assert (text.qk_nope_head_dim, text.qk_rope_head_dim, text.v_head_dim) == (
        256,
        0,
        256,
    )
    assert text.qk_head_dim == 256 and text.head_dim == 0

    # KDA
    la = text.linear_attn_config
    assert (la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]) == (
        64,
        128,
        4,
    )
    assert la["gate_lower_bound"] == -5.0 and la["use_full_rank_gate"] is False
    assert la["kda_layers"][:4] == [1, 2, 3, 5]  # 1-based, Kimi convention
    assert la["full_attn_layers"] == [i + 1 for i in dsa]
    assert (text.linear_num_heads, text.linear_head_dim) == (64, 128)
    assert text.linear_conv_kernel_dim == 4 and text.linear_lower_bound == -5.0

    # pooled indexer
    assert (text.index_n_heads, text.index_head_dim, text.index_topk) == (
        32,
        128,
        2048,
    )
    assert text.index_kpool == 4
    assert text.index_kpool_compress is True
    assert text.index_kpool_always_select_tail is True

    # mHC
    assert (text.hc_mult, text.hc_sinkhorn_iters, text.hc_eps) == (4, 20, 1e-6)

    # tokens: the full end-of-generation set, not just <|endoftext|>
    assert text.eos_token_id == [154820, 154827, 154829]
    assert text.bos_token_id == 154822 and text.pad_token_id == 154821
    assert text.tie_word_embeddings is False


@needs_header
def test_config_parser_dispatch(cfg):
    from vllm.transformers_utils.gguf_config_parser import GGUFConfigParser
    from vllm.transformers_utils.gguf_glm5_next import is_glm5_next_gguf

    assert is_glm5_next_gguf(str(HEADER))
    parsed_dict, parsed = GGUFConfigParser().parse(HEADER, trust_remote_code=False)
    assert parsed is cfg
    assert parsed_dict["architectures"] == ["Glm5NextForConditionalGeneration"]


def test_openai_parser_defaults():
    from vllm.entrypoints.openai.model_parsers import _PARSERS_BY_ARCHITECTURE

    assert _PARSERS_BY_ARCHITECTURE["glm5-next"] == ("glm47", "glm47")
    assert _PARSERS_BY_ARCHITECTURE["glm-dsa"] == ("glm45", "glm47")


def test_registry_has_text_only_entry():
    from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS

    assert _TEXT_GENERATION_MODELS["Glm5NextForCausalLM"] == (
        "glm5_next",
        "Glm5NextForCausalLM",
    )


# ------------------------------------------------------------ name coverage


def _adapter(cfg, mtp: bool):
    from vllm.model_executor.model_loader.gguf_adapters import (
        Glm5NextGGUFAdapter,
    )

    if mtp:
        # SpeculativeConfig.hf_config_override flattens the text config and
        # rewrites model_type; the adapter keys only on that.
        from vllm.transformers_utils.gguf_glm5_next import text_config_fields_from_gguf

        text = SimpleNamespace(**text_config_fields_from_gguf(str(HEADER)))
        text.model_type = "glm5_next_mtp"
        text.get_text_config = lambda: text
        hf_config = text
    else:
        hf_config = cfg
    adapter = Glm5NextGGUFAdapter(hf_config)
    model_config = SimpleNamespace(hf_config=hf_config, model=str(HEADER))
    return adapter, adapter.build_name_map(model_config)


@needs_header
def test_name_maps_cover_every_tensor_exactly_once(cfg, tensors):
    _, trunk = _adapter(cfg, mtp=False)
    _, mtp = _adapter(cfg, mtp=True)

    names = set(tensors)
    assert names == _census_names(), "header copy and census disagree"
    assert len(names) == 1412

    assert set(trunk) & set(mtp) == {"token_embd.weight", "output.weight"}
    assert (set(trunk) | set(mtp)) == names  # nothing unmapped, no phantoms
    assert all(n.startswith("blk.45.") for n in set(mtp) - set(trunk))
    assert not any(n.startswith("blk.45.") for n in trunk)  # nextn dropped

    # Two GGUF tensors onto one HF name only for the absorbed MLA pair.
    for name_map in (trunk, mtp):
        doubled = {hf for hf, n in Counter(name_map.values()).items() if n > 1}
        assert all(hf.endswith("self_attn.kv_b_proj.weight") for hf in doubled)
        assert all(Counter(name_map.values())[hf] == 2 for hf in doubled)

    assert trunk["token_embd.weight"] == "model.language_model.embed_tokens.weight"
    assert trunk["output.weight"] == "lm_head.weight"
    assert trunk["output_norm.weight"] == "model.language_model.norm.weight"
    assert mtp["token_embd.weight"] == (
        "model.language_model.layers.45.embed_tokens.weight"
    )
    assert mtp["output.weight"] == (
        "model.language_model.layers.45.shared_head.head.weight"
    )
    assert mtp["blk.45.nextn.eh_proj.weight"] == (
        "model.language_model.layers.45.eh_proj.weight"
    )


# --------------------------------------------- resolution against the model


def _stacked_params_mapping_from_source() -> list[tuple[str, str, int]]:
    src = MODEL_SRC.read_text()
    # The list closes on its own line; a `]` inside a comment must not end it.
    block = re.search(r"stacked_params_mapping = \[(.*?)\n    \]", src, re.S).group(1)
    found = re.findall(r'\("(\w+)", "(\w+)", (\d+)\)', block)
    assert len(found) >= 13, found
    assert found, "could not parse stacked_params_mapping from glm5_next.py"
    return [(a, b, int(c)) for a, b, c in found]


def _hf_to_vllm_prefix_from_source() -> dict[str, str]:
    src = MODEL_SRC.read_text()
    block = src.split("hf_to_vllm_prefix = {", 1)[1].split("}", 1)[0]
    return dict(re.findall(r'"([\w.]+)": "([\w.]+)"', block))


def _expected_text_params(text) -> set[str]:
    """Parameters ``Glm5NextForCausalLM`` registers under a GGUF quant config.

    Sources: glm5_next.py (layer/attn/mlp/lm_head module names),
    kimi_gdn_linear_attn.py (in_proj_qkvgfab, f_b_proj, g_b_proj, conv1d,
    dt_bias, A_log, o_norm, o_proj), glm5_next_indexer.py (wq_b, wk,
    weights_proj, k_norm, index_kpool_compress_{ape,gate}), deepseek_v2.py
    (gate, e_score_correction_bias, shared_experts, experts w13/w2), and the
    GGUF linear/MoE/embedding methods (``qweight`` + ``qweight_type``).
    """

    def q(name: str) -> set[str]:
        return {f"{name}.qweight", f"{name}.qweight_type"}

    params = {
        "model.embed_tokens.weight",  # built without quant_config
        "model.norm.weight",
        *q("lm_head"),
    }
    for i in range(text.num_hidden_layers):
        b = f"model.layers.{i}."
        params |= {b + "input_layernorm.weight", b + "post_attention_layernorm.weight"}
        params |= {
            b + f"hc_{site}_{part}"
            for site in ("attn", "ffn")
            for part in ("fn", "base", "scale")
        }
        a = b + "self_attn."
        if text.layer_types[i] == "linear_attention":
            params |= q(a + "in_proj_qkvgfab") | q(a + "f_b_proj") | q(a + "g_b_proj")
            params |= q(a + "o_proj")
            params |= {
                a + "conv1d.weight",
                a + "dt_bias",
                a + "A_log",
                a + "o_norm.weight",
            }
        else:
            params |= q(a + "fused_qkv_a_proj") | q(a + "q_b_proj") | q(a + "o_proj")
            params |= {
                a + "q_a_layernorm.weight",
                a + "kv_a_layernorm.weight",
                a + "kv_b_proj.weight",  # dequantized, unquantized module
                a + "indexer.wq_b.weight",
                a + "indexer.wk.weight",
                a + "indexer.weights_proj.weight",
                a + "indexer.k_norm.weight",
                a + "indexer.k_norm.bias",
                a + "indexer.index_kpool_compress_ape",
                a + "indexer.index_kpool_compress_gate",
            }
        m = b + "mlp."
        if text.mlp_layer_types[i] == "dense":
            params |= q(m + "gate_up_proj") | q(m + "down_proj")
        else:
            params |= {m + "gate.weight", m + "gate.e_score_correction_bias"}
            params |= {
                m + "experts." + n
                for n in (
                    "w13_qweight",
                    "w13_qweight_type",
                    "w2_qweight",
                    "w2_qweight_type",
                )
            }
            params |= q(m + "shared_experts.gate_up_proj") | q(
                m + "shared_experts.down_proj"
            )
    return params


def _emitted_names(gguf_name: str, hf_name: str, gguf_type: str) -> list[str]:
    """What prepare_weights yields for one GGUF tensor (adapter dtype policy)."""
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        _KV_B_SUFFIXES,
    )

    suffix = gguf_name.split(".", 2)[-1] if gguf_name.startswith("blk.") else gguf_name
    if suffix in _KV_B_SUFFIXES or gguf_name == "token_embd.weight":
        return [hf_name]  # dequantized -> plain `.weight`
    if gguf_type in F32:
        return [hf_name]  # passed through as-is
    # gguf_quant_weights_iterator_multi: `weight` -> `qweight_type`, `qweight`
    return [
        hf_name.replace("weight", "qweight_type"),
        hf_name.replace("weight", "qweight"),
    ]


def _resolve_text(name: str, stacked, prefixes) -> str | None:
    """Mirror Glm5NextForCausalLM.load_weights' name resolution."""
    for pref, new in prefixes.items():
        if name.startswith(pref):
            name = new + name[len(pref) :]
            break
    else:
        return None
    if ".mlp.experts." in name:
        # fused_moe_make_expert_params_mapping: experts.{E}.{proj}. -> w13_/w2_
        for proj, fused in (
            ("gate_proj", "w13_"),
            ("up_proj", "w13_"),
            ("down_proj", "w2_"),
        ):
            token = f"experts.0.{proj}."
            if token in name:
                return name.replace(token, f"experts.{fused}")
        return name
    for target, ckpt, _shard in stacked:
        if f".{ckpt}." in name or name.endswith(f".{ckpt}"):
            return name.replace(ckpt, target)
    return name


@needs_header
def test_every_mapped_target_exists_in_text_model(cfg, text, tensors):
    _, trunk = _adapter(cfg, mtp=False)
    stacked = _stacked_params_mapping_from_source()
    prefixes = _hf_to_vllm_prefix_from_source()
    expected = _expected_text_params(text)

    sources: Counter[str] = Counter()
    for gguf_name, hf_name in trunk.items():
        for emitted in _emitted_names(gguf_name, hf_name, tensors[gguf_name][0]):
            target = _resolve_text(emitted, stacked, prefixes)
            assert target is not None, f"{hf_name} is dropped by hf_to_vllm_prefix"
            assert target in expected, (
                f"{gguf_name} -> {emitted} -> {target} is not a model parameter"
            )
            sources[target] += 1

    assert set(sources) == expected, sorted(expected - set(sources))

    # Merged parameters take exactly their shard count; everything else one.
    merged_shards = {
        "in_proj_qkvgfab": 6,
        "conv1d": 3,
        "fused_qkv_a_proj": 2,
        "gate_up_proj": 2,
        "w13": 2,
    }
    for target, count in sources.items():
        stem = target.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        if target.rsplit(".", 1)[-1].startswith("w13_"):
            stem = "w13"
        want = merged_shards.get(stem, 1)
        if target.endswith("kv_b_proj.weight"):
            want = 2  # attn_k_b + attn_v_b, fused before the yield
        assert count == want, f"{target}: {count} sources, expected {want}"


def _resolve_mtp(name: str, spec_layer: int) -> str | None:
    """Mirror Glm5NextMTP.load_weights + DeepSeekMTP._rewrite_spec_layer_name."""
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model.") :]
    if not name.startswith(f"model.layers.{spec_layer}."):
        return None
    top = ("embed_tokens", "enorm", "hnorm", "eh_proj", "shared_head")
    if ".self_attn.indexer." in name or not any(t in name for t in top):
        name = name.replace(
            f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
        )
    elif "embed_tokens" in name:
        name = name.replace(f"model.layers.{spec_layer}.", "model.")
    if ".mlp.experts." in name:
        for proj, fused in (
            ("gate_proj", "w13_"),
            ("up_proj", "w13_"),
            ("down_proj", "w2_"),
        ):
            token = f"experts.0.{proj}."
            if token in name:
                return name.replace(token, f"experts.{fused}")
    for target, ckpt in (
        ("gate_up_proj", "gate_proj"),
        ("gate_up_proj", "up_proj"),
        ("fused_qkv_a_proj", "q_a_proj"),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa"),
    ):
        if ckpt in name:
            return name.replace(ckpt, target)
    return name


@needs_header
def test_mtp_map_resolves_to_draft_params(cfg, text, tensors):
    _, mtp = _adapter(cfg, mtp=True)
    L = text.num_hidden_layers
    b = f"model.layers.{L}."
    a = b + "mtp_block.self_attn."
    m = b + "mtp_block.mlp."

    def q(name):
        return {f"{name}.qweight", f"{name}.qweight_type"}

    expected = {
        "model.embed_tokens.weight",
        b + "enorm.weight",
        b + "hnorm.weight",
        b + "eh_proj.weight",
        b + "shared_head.norm.weight",
        *q(b + "shared_head.head"),
        b + "mtp_block.input_layernorm.weight",
        b + "mtp_block.post_attention_layernorm.weight",
        *q(a + "fused_qkv_a_proj"),
        *q(a + "q_b_proj"),
        *q(a + "o_proj"),
        a + "q_a_layernorm.weight",
        a + "kv_a_layernorm.weight",
        a + "kv_b_proj.weight",
        a + "indexer.wq_b.weight",
        a + "indexer.wk.weight",
        a + "indexer.weights_proj.weight",
        a + "indexer.k_norm.weight",
        a + "indexer.k_norm.bias",
        a + "indexer.index_kpool_compress_ape",
        a + "indexer.index_kpool_compress_gate",
        m + "gate.weight",
        m + "gate.e_score_correction_bias",
        *(
            m + "experts." + n
            for n in (
                "w13_qweight",
                "w13_qweight_type",
                "w2_qweight",
                "w2_qweight_type",
            )
        ),
        *q(m + "shared_experts.gate_up_proj"),
        *q(m + "shared_experts.down_proj"),
    }
    resolved = set()
    for gguf_name, hf_name in mtp.items():
        for emitted in _emitted_names(gguf_name, hf_name, tensors[gguf_name][0]):
            target = _resolve_mtp(emitted, L)
            assert target is not None, hf_name
            resolved.add(target)
    assert resolved == expected


# ------------------------------------------------------- dtype / quant policy


@needs_header
def test_dtype_policy(cfg, tensors):
    """Routed experts stay packed; small tensors are F32/BF16 in the file."""
    adapter, trunk = _adapter(cfg, mtp=False)
    types = {n: tensors[n][0] for n in trunk}
    assert {types[f"blk.{i}.ffn_gate_exps.weight"] for i in range(3, 45)} == {"IQ2_XXS"}
    assert {types[f"blk.{i}.ffn_down_exps.weight"] for i in range(3, 45)} == {"Q2_K"}
    for i in range(3, 45):
        assert types[f"blk.{i}.ffn_gate_inp.weight"] == "F32"
        assert types[f"blk.{i}.exp_probs_b.bias"] == "F32"
    for i in (0, 1, 2, 4):
        for s in ("kda_dt_bias", "kda_a_log", "kda_o_norm", "kda_q_conv"):
            assert types[f"blk.{i}.{s}.weight"] == "F32"
        assert types[f"blk.{i}.kda_q.weight"] == "Q4_K"
        assert types[f"blk.{i}.kda_v.weight"] == "Q8_0"
    assert types["blk.3.indexer.attn_k.weight"] == "BF16"
    assert types["blk.3.hc_attn_fn.weight"] == "BF16"

    # Unquantized modules, in vLLM-module spelling (no checkpoint prefix,
    # leading dot anchors the layer index).
    weight_type_map = {hf: tensors[g][0] for g, hf in trunk.items()}
    unq = set(adapter.get_unquantized_modules(weight_type_map))
    assert ".layers.3.self_attn.kv_b_proj" in unq  # dequantized regardless of Q8_0
    assert ".embed_tokens" in unq
    assert ".layers.3.self_attn.indexer.wk" in unq
    assert ".layers.3.self_attn.indexer.wq_b" in unq
    assert ".layers.3.self_attn.indexer.weights_proj" in unq
    assert ".layers.3.mlp.gate" in unq
    assert "lm_head" not in unq  # Q8_0 through the GGUF embedding method
    assert ".layers.3.self_attn.q_b_proj" not in unq
    assert ".layers.0.self_attn.f_b_proj" not in unq
    assert not any(m.startswith("model.") for m in unq)
    # A `.layers.3.` entry can never be a substring of a layer-13 prefix.
    assert not any(
        ".layers.3.mlp.gate" in f"language_model.model.layers.13.mlp.{x}"
        for x in ("gate", "gate_up_proj")
    )


def test_mtp_unquantized_modules_use_mtp_block():
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        Glm5NextGGUFAdapter,
        vllm_module_name,
    )

    assert vllm_module_name(
        "model.language_model.layers.3.self_attn.kv_b_proj.weight", None
    ) == (".layers.3.self_attn.kv_b_proj")
    assert (
        vllm_module_name("model.language_model.embed_tokens.weight", None)
        == ".embed_tokens"
    )
    assert vllm_module_name("lm_head.weight", None) == "lm_head"
    assert (
        vllm_module_name(
            "model.language_model.layers.45.self_attn.kv_b_proj.weight", 45
        )
        == ".layers.45.mtp_block.self_attn.kv_b_proj"
    )
    assert vllm_module_name("model.language_model.layers.45.eh_proj.weight", 45) == (
        ".layers.45.eh_proj"
    )
    assert (
        vllm_module_name("model.language_model.layers.45.embed_tokens.weight", 45)
        == ".layers.45.embed_tokens"
    )

    adapter = Glm5NextGGUFAdapter.__new__(Glm5NextGGUFAdapter)
    adapter._mtp_layer = 45
    unq = adapter.get_unquantized_modules(
        {
            "model.language_model.layers.45.self_attn.kv_b_proj.weight": "Q8_0",
            "model.language_model.layers.45.self_attn.q_b_proj.weight": "Q8_0",
            "model.language_model.layers.45.self_attn.indexer.wk.weight": "BF16",
            "model.language_model.layers.45.embed_tokens.weight": "Q8_0",
        }
    )
    assert unq == [
        ".layers.45.embed_tokens",
        ".layers.45.mtp_block.self_attn.indexer.wk",
        ".layers.45.mtp_block.self_attn.kv_b_proj",
    ]


# ------------------------------------------------------- shapes and shards


@needs_header
def test_file_shapes_match_derived_config(cfg, text, tensors):
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        expected_tensor_shapes,
        verify_tensor_shapes,
    )

    present = {n: shape for n, (_, shape) in tensors.items()}
    verify_tensor_shapes(present, text)  # every tensor, every shape
    assert set(expected_tensor_shapes(text)) == set(present)

    bad = dict(present)
    bad["blk.0.kda_q.weight"] = (8192, 4095)
    with pytest.raises(ValueError, match="wrong shape"):
        verify_tensor_shapes(bad, text)
    bad = dict(present)
    del bad["blk.7.attn_k_b.weight"]
    with pytest.raises(ValueError, match="missing"):
        verify_tensor_shapes(bad, text)
    bad = dict(present)
    bad["blk.0.kda_g_conv.weight"] = (8192, 1, 4)
    with pytest.raises(ValueError, match="unmapped"):
        verify_tensor_shapes(bad, text)


@needs_header
def test_kda_in_proj_shards_match_file_and_layer(text, tensors):
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        kda_conv_layout,
        kda_in_proj_layout,
    )

    la = text.linear_attn_config
    heads, dim, kernel = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    proj = heads * dim

    # KimiGatedDeltaNetAttention(fuse_gate_a=True): [P, P, P, H, D, D].
    layout = kda_in_proj_layout(heads, dim)
    assert [rows for _, _, rows in layout] == [proj] * 3 + [heads, dim, dim]
    assert [shard for _, shard, _ in layout] == [0, 1, 2, 3, 4, 5]
    assert sum(rows for _, _, rows in layout) == 3 * 8192 + 64 + 128 + 128
    for suffix, _shard, rows in layout:
        assert tensors[f"blk.0.{suffix}"][1] == (rows, text.hidden_size)

    conv = kda_conv_layout(heads, dim, kernel)
    assert [shape for _, _, shape in conv] == [(proj, 1, kernel)] * 3
    assert sum(shape[0] for _, _, shape in conv) == 3 * proj  # fused [3P, 1, K]
    for suffix, _shard, shape in conv:
        assert tensors[f"blk.0.{suffix}"][1] == shape

    # f_b / g_b are [P, D] column-parallel weights, o_proj is [hidden, P].
    assert tensors["blk.0.kda_f_b.weight"][1] == (proj, dim)
    assert tensors["blk.0.kda_g_b.weight"][1] == (proj, dim)
    assert tensors["blk.0.kda_output.weight"][1] == (text.hidden_size, proj)
    assert tensors["blk.0.kda_dt_bias.weight"][1] == (proj,)
    assert tensors["blk.0.kda_a_log.weight"][1] == (heads,)


def test_fused_conv1d_loader_accepts_per_shard_conv():
    """The GGUF conv (P, 1, K) lands in rows [shard*P, (shard+1)*P) of [3P, 1, K]."""
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        _make_fused_conv1d_weight_loader,
    )
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        kda_conv_layout,
    )

    heads, dim, kernel = 2, 3, 4
    proj = heads * dim
    param = torch.zeros(3 * proj, 1, kernel)
    loader = _make_fused_conv1d_weight_loader([proj] * 3, tp_size=1, tp_rank=0)
    for _suffix, shard, shape in kda_conv_layout(heads, dim, kernel):
        loader(param, torch.full(shape, float(shard + 1)), shard)
    for shard in range(3):
        assert torch.all(param[shard * proj : (shard + 1) * proj] == shard + 1)


def test_kv_b_proj_rebuild():
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        assemble_kv_b_proj,
    )

    heads, kv_lora, qk_nope, v_head = 3, 5, 2, 4
    k_b = torch.randn(heads, kv_lora, qk_nope)
    v_b = torch.randn(heads, v_head, kv_lora)
    fused = assemble_kv_b_proj(k_b, v_b, qk_nope, v_head, kv_lora)
    assert fused.shape == (heads * (qk_nope + v_head), kv_lora)
    assert fused.dtype == torch.bfloat16
    per_head = fused.float().view(heads, qk_nope + v_head, kv_lora)
    torch.testing.assert_close(
        per_head[:, :qk_nope], k_b.transpose(1, 2), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(per_head[:, qk_nope:], v_b, atol=2e-2, rtol=2e-2)

    # The MLA wrapper splits kv_b_proj per head into [qk_nope | v_head]; check
    # the interleave is per head, not [all k | all v].
    head1 = fused.float()[(qk_nope + v_head) : 2 * (qk_nope + v_head)]
    torch.testing.assert_close(head1[:qk_nope], k_b[1].T, atol=2e-2, rtol=2e-2)

    with pytest.raises(ValueError, match="attn_k_b"):
        assemble_kv_b_proj(k_b.transpose(1, 2), v_b, qk_nope, v_head, kv_lora)
    with pytest.raises(ValueError, match="attn_v_b"):
        assemble_kv_b_proj(k_b, v_b.transpose(1, 2), qk_nope, v_head, kv_lora)


@needs_header
def test_kv_b_file_shapes_feed_the_rebuild(text, tensors):
    heads = text.num_attention_heads
    assert tensors["blk.3.attn_k_b.weight"][1] == (
        heads,
        text.kv_lora_rank,
        text.qk_nope_head_dim,
    )
    assert tensors["blk.3.attn_v_b.weight"][1] == (
        heads,
        text.v_head_dim,
        text.kv_lora_rank,
    )
    fused_rows = heads * (text.qk_nope_head_dim + text.v_head_dim)
    assert fused_rows == 32768  # kv_b_proj [32768, 512] per DSA layer


def test_expected_block_shapes_pure():
    """Tiny synthetic config: layout math without the file."""
    from vllm.model_executor.model_loader.gguf_adapters.glm5_next import (
        block_renames,
        expected_block_shapes,
    )

    cfg = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "hc_mult": 2,
        "linear_attn_config": {
            "num_heads": 2,
            "head_dim": 3,
            "short_conv_kernel_size": 4,
        },
        "num_attention_heads": 2,
        "q_lora_rank": 5,
        "kv_lora_rank": 6,
        "qk_nope_head_dim": 7,
        "qk_rope_head_dim": 0,
        "v_head_dim": 9,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_kpool": 3,
        "intermediate_size": 10,
        "n_routed_experts": 11,
        "moe_intermediate_size": 12,
        "n_shared_experts": 1,
        "vocab_size": 20,
    }
    kda = expected_block_shapes(cfg, 0)
    assert kda["kda_q.weight"] == (6, 8) and kda["kda_beta.weight"] == (2, 8)
    assert kda["kda_f_a.weight"] == (3, 8) and kda["kda_f_b.weight"] == (6, 3)
    assert kda["kda_q_conv.weight"] == (6, 1, 4) and kda["hc_attn_fn.weight"] == (8, 16)
    assert kda["ffn_gate.weight"] == (10, 8) and "ffn_gate_exps.weight" not in kda
    mla = expected_block_shapes(cfg, 1)
    assert mla["attn_q_b.weight"] == (14, 5) and mla["attn_kv_a_mqa.weight"] == (6, 8)
    assert mla["attn_k_b.weight"] == (2, 6, 7) and mla["attn_v_b.weight"] == (2, 9, 6)
    assert mla["indexer.pool_ape.weight"] == (3, 4) and mla["ffn_down_exps.weight"] == (
        11,
        8,
        12,
    )
    mtp = expected_block_shapes(cfg, 2)
    assert "hc_attn_fn.weight" not in mtp and mtp["nextn.eh_proj.weight"] == (8, 16)
    assert set(mtp) == set(block_renames(cfg, 2))
    assert set(kda) == set(block_renames(cfg, 0)) and set(mla) == set(
        block_renames(cfg, 1)
    )


# ---------------------------------------------------------------- tokenizer


@needs_header
def test_tokenizer_keeps_the_files_own_template():
    from vllm.transformers_utils.gguf_glm5_next import (
        build_glm5_next_tokenizer_from_gguf,
    )
    from vllm.transformers_utils.gguf_utils import gguf_reader

    tok = build_glm5_next_tokenizer_from_gguf(str(HEADER))
    template = str(
        gguf_reader(str(HEADER)).fields["tokenizer.chat_template"].contents()
    )
    assert tok.chat_template == template
    assert "[gMASK]<sop>" in template
    assert tok.bos_token_id == 154822 and tok.eos_token_id == 154820
    assert tok.convert_tokens_to_ids("<|user|>") == 154827
    assert tok.encode("[gMASK]", add_special_tokens=False) == [154822]
    sop = tok.convert_tokens_to_ids("<sop>")
    assert sop > 154820  # a CONTROL token, registered special, not BPE-split
    assert tok.encode("[gMASK]<sop>", add_special_tokens=False) == [154822, sop]
    assert (
        tok.decode(tok.encode("Paris is 77 42", add_special_tokens=False))
        == "Paris is 77 42"
    )
