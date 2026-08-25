# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF plumbing proofs for the Qwen3.8-27B campaign (real artifacts).

Skipped when the artifacts are not present; on the campaign box they pin
the config builders, the adapter name maps (bidirectional completeness
caught real bugs during bring-up), and the qwen35 pre-tokenizer split
against ids verified byte-identical to llama.cpp's /tokenize.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

TARGET = Path.home() / "models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf"
DRAFTER = (
    Path.home() / "models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
)

pytestmark = pytest.mark.skipif(
    not (TARGET.is_file() and DRAFTER.is_file()),
    reason="Qwen3.8-27B campaign artifacts not downloaded",
)


def test_target_config_from_gguf():
    from vllm.transformers_utils.gguf_qwen35 import build_qwen35_config_from_gguf

    cfg = build_qwen35_config_from_gguf(str(TARGET))
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    assert text.num_hidden_layers == 64  # MTP block excluded
    assert text.layer_types.count("linear_attention") == 48
    assert text.layer_types.count("full_attention") == 16
    assert [i for i, t in enumerate(text.layer_types) if t == "full_attention"][:3] == [
        3,
        7,
        11,
    ]
    assert (text.num_attention_heads, text.num_key_value_heads, text.head_dim) == (
        24,
        4,
        256,
    )
    assert (text.linear_num_key_heads, text.linear_num_value_heads) == (16, 48)
    assert text.rope_parameters["mrope_section"] == [11, 11, 10]
    assert text.rope_parameters["mrope_interleaved"] is True
    assert abs(text.rope_parameters["partial_rotary_factor"] - 0.25) < 1e-9
    assert text.vocab_size == 248320
    assert getattr(text, "gdn_tiled_v_head_layout", None) is True


def test_drafter_config_from_gguf():
    from vllm.transformers_utils.gguf_qwen35 import (
        build_qwen38_dflash2_config_from_gguf,
        is_dflash2_gguf,
    )

    assert is_dflash2_gguf(str(DRAFTER))
    cfg = build_qwen38_dflash2_config_from_gguf(str(DRAFTER))
    assert cfg.architectures == ["DFlash2QwenDraftModel"]
    assert cfg.num_hidden_layers == 5
    assert cfg.block_size == 8 and cfg.n_predict == 7
    d = cfg.dflash_config
    assert d["target_layer_ids"] == [5, 19, 33, 47, 61]  # GGUF is 1-based
    assert d["causal"] is False
    assert d["mask_token_id"] == 248070
    assert (d["selector_rank"], d["selector_top_k"]) == (256, 16)
    assert (d["conv_kernel_size"], d["conv_group_size"]) == (2, 16)
    assert cfg.sliding_window == 2048


def test_adapter_name_maps_are_complete():
    from gguf import GGUFReader

    from vllm.model_executor.model_loader.gguf_adapters import (
        DFlash2QwenGGUFAdapter,
        Qwen35GGUFAdapter,
    )
    from vllm.transformers_utils.gguf_qwen35 import (
        build_qwen35_config_from_gguf,
        build_qwen38_dflash2_config_from_gguf,
    )

    tcfg = build_qwen35_config_from_gguf(str(TARGET))
    tmap = Qwen35GGUFAdapter(tcfg).build_name_map(SimpleNamespace(hf_config=tcfg))
    tnames = {t.name for t in GGUFReader(str(TARGET)).tensors}
    mtp = {n for n in tnames if n.startswith("blk.64.")}
    assert len(mtp) == 15
    assert (tnames - mtp) - set(tmap) == set()  # every non-MTP tensor mapped
    assert set(tmap) - tnames == set()  # no phantom entries

    dcfg = build_qwen38_dflash2_config_from_gguf(str(DRAFTER))
    dmap = DFlash2QwenGGUFAdapter(dcfg).build_name_map(SimpleNamespace(hf_config=dcfg))
    dnames = {t.name for t in GGUFReader(str(DRAFTER)).tensors}
    assert len(dnames) == 81
    assert dnames - set(dmap) == set() and set(dmap) - dnames == set()


def test_qwen35_tokenizer_matches_llama_cpp_ids():
    from vllm.transformers_utils.gguf_qwen35 import (
        build_qwen35_tokenizer_from_gguf,
    )

    tok = build_qwen35_tokenizer_from_gguf(str(TARGET))
    prompt = "The three most important properties of a good inference engine are"
    # Verified byte-identical to llama.cpp /tokenize on 2026-08-20.
    assert tok.encode(prompt) == [
        760,
        2250,
        1379,
        2894,
        5706,
        314,
        264,
        1603,
        42903,
        4560,
        513,
    ]
    assert tok.decode(tok.encode("Paris is")) == "Paris is"
