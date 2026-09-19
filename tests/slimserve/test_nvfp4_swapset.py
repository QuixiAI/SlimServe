# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 sidecar's serving-side helpers: manifest family filter, config
group injection, the FP8 group trimmed of the claimed modules, the ignore
list, and the quantizer against vLLM's reference (CPU)."""

import json

import pytest
import torch

from slimserve import nvfp4_swapset as ns


def _manifest(tmp_path):
    modules = [(0, "self_attn.in_proj_qkvgfab"), (0, "self_attn.o_proj"), (3, "self_attn.o_proj"), (3, "self_attn.q_b_proj")]
    shards = {
        "model.language_model.layers.0.self_attn.q_proj": "self_attn.in_proj_qkvgfab",
        "model.language_model.layers.0.self_attn.o_proj": "self_attn.o_proj",
        "model.language_model.layers.3.self_attn.o_proj": "self_attn.o_proj",
        "model.language_model.layers.3.self_attn.q_b_proj": "self_attn.q_b_proj",
    }
    tensors = sorted(f"{s}.{t}" for s in shards for t in ("weight_packed", "weight_scale", "weight_global_scale"))
    m = {
        "file": "nvfp4-swapset.safetensors",
        "families": sorted({mod for _, mod in modules}),
        "tensors": tensors,
        "modules": [f"layers.{layer}.{mod}" for layer, mod in modules],
        "shards": shards,
        "config_group": ns.config_group(modules),
    }
    with open(tmp_path / "nvfp4-swapset.json", "w") as fh:
        json.dump(m, fh)
    return m


def test_off_by_default(tmp_path, monkeypatch):
    _manifest(tmp_path)
    monkeypatch.delenv(ns.SWAPSET_ENV, raising=False)
    assert ns.load_manifest(str(tmp_path)) is None
    assert ns.claimed_modules(str(tmp_path)) == set()
    assert ns.hash_factor(str(tmp_path)) == "nvfp4_swapset=off"


def test_family_filter_and_group(tmp_path, monkeypatch):
    _manifest(tmp_path)
    monkeypatch.setenv(ns.SWAPSET_ENV, "nvfp4-swapset")
    monkeypatch.setenv(ns.FAMILIES_ENV, "self_attn.o_proj")
    m = ns.load_manifest(str(tmp_path))
    assert m["modules"] == ["layers.0.self_attn.o_proj", "layers.3.self_attn.o_proj"]
    assert set(m["shards"]) == {"model.language_model.layers.0.self_attn.o_proj", "model.language_model.layers.3.self_attn.o_proj"}
    assert len(m["tensors"]) == 6
    assert m["config_group"]["targets"] == [r"re:.*\.layers\.(0|3)\.self_attn\.o_proj$"]
    assert m["config_group"]["input_activations"] is None
    assert m["config_group"]["weights"]["group_size"] == 16
    monkeypatch.setenv(ns.FAMILIES_ENV, "mlp.bogus")
    with pytest.raises(ValueError):
        ns.load_manifest(str(tmp_path))


def test_apply_config_group_trims_fp8_and_ignore(tmp_path, monkeypatch):
    _manifest(tmp_path)
    monkeypatch.setenv(ns.SWAPSET_ENV, "nvfp4-swapset")
    monkeypatch.delenv(ns.FAMILIES_ENV, raising=False)
    cfg = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "fp8": {"targets": [r"re:.*\.layers\.(0|1|3)\.self_attn\.o_proj$", r"re:.*\.layers\.(3)\.self_attn\.q_b_proj$", r"re:.*\.layers\.(0|1)\.self_attn\.in_proj_qkvgfab$"]},
        },
        "ignore": ["model.language_model.layers.0.self_attn.o_proj", "model.language_model.layers.1.self_attn.o_proj", "lm_head"],
    }
    assert ns.apply_config_group(str(tmp_path), cfg, "fp8")
    assert ns.GROUP_NAME in cfg["config_groups"]
    assert cfg["config_groups"]["fp8"]["targets"] == [r"re:.*\.layers\.(1)\.self_attn\.o_proj$", r"re:.*\.layers\.(1)\.self_attn\.in_proj_qkvgfab$"]
    assert cfg["ignore"] == ["model.language_model.layers.1.self_attn.o_proj", "lm_head"]
    assert ns.hash_factor(str(tmp_path)).startswith("nvfp4_swapset=")


def test_quantizer_matches_reference():
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        break_fp4_bytes,
        ref_nvfp4_quant,
    )

    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    gs = ns.global_scale_for([w])
    packed, scale = ns.quantize_nvfp4(w, gs)
    ref_q, ref_scale = ref_nvfp4_quant(w, gs, 16)
    assert torch.equal(break_fp4_bytes(packed, torch.float32), ref_q)
    assert torch.equal(scale.float(), ref_scale.float())
    err = (ns.dequant_nvfp4(packed, scale, gs) - w).norm() / w.norm()
    assert err < 0.15
