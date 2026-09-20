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


def _fake_checkpoint(tmp_path, experts=4, layers=(3, 4), hidden=64, inter=32):
    """A tiny compressed-tensors-shaped checkpoint dir: config.json and one
    safetensors shard with BF16 shared experts for `layers` (and layer 45)."""
    from safetensors.torch import save_file

    with open(tmp_path / "config.json", "w") as fh:
        json.dump({"text_config": {"n_routed_experts": experts}}, fh)
    torch.manual_seed(0)
    tensors = {}
    for layer in (*layers, 45):
        base = f"model.language_model.layers.{layer}.mlp.shared_experts"
        tensors[f"{base}.gate_proj.weight"] = torch.randn(inter, hidden).to(torch.bfloat16)
        tensors[f"{base}.up_proj.weight"] = torch.randn(inter, hidden).to(torch.bfloat16)
        tensors[f"{base}.down_proj.weight"] = torch.randn(hidden, inter).to(torch.bfloat16)
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    return tensors


def test_shared_expert_sidecar_build_and_manifest(tmp_path, monkeypatch):
    """The shared-expert sidecar: each served layer's BF16 shared expert
    quantized as expert E (the MTP layer left alone), the manifest naming the
    consumed shards and the claimed modules, and the serving-side helpers."""
    src = _fake_checkpoint(tmp_path)
    out = ns.build_shared_experts(str(tmp_path), None, 45)
    assert out.name == f"{ns.SHARED_STEM}.safetensors"
    with open(tmp_path / f"{ns.SHARED_STEM}.json") as fh:
        manifest = json.load(fh)
    assert manifest["kind"] == "shared_experts" and manifest["expert"] == 4
    assert manifest["layers"] == [3, 4]
    for layer in (3, 4):
        base = f"model.language_model.layers.{layer}.mlp.experts.4"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for t in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
                assert f"{base}.{proj}.{t}" in manifest["tensors"]
    assert all(".layers.45." not in t for t in manifest["tensors"])
    assert set(manifest["consume"]) == {n for n in src if ".layers.45." not in n}
    assert manifest["modules"] == [
        "layers.3.mlp.shared_experts.gate_up_proj", "layers.3.mlp.shared_experts.down_proj",
        "layers.4.mlp.shared_experts.gate_up_proj", "layers.4.mlp.shared_experts.down_proj",
    ]
    from safetensors.torch import load_file

    tensors = load_file(str(out))
    packed = tensors["model.language_model.layers.3.mlp.experts.4.gate_proj.weight_packed"]
    scale = tensors["model.language_model.layers.3.mlp.experts.4.gate_proj.weight_scale"]
    gs = tensors["model.language_model.layers.3.mlp.experts.4.gate_proj.weight_global_scale"]
    assert packed.dtype == torch.uint8 and packed.shape == (32, 32)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (32, 4)
    w = src["model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"].float()
    err = (ns.dequant_nvfp4(packed, scale, gs) - w).norm() / w.norm()
    assert err < 0.2
    # Serving side: off by default, on through the env; the claims join the dense sidecar's.
    monkeypatch.delenv(ns.SHARED_ENV, raising=False)
    monkeypatch.delenv(ns.SWAPSET_ENV, raising=False)
    assert ns.load_shared_manifest(str(tmp_path)) is None
    assert ns.shared_expert_layers(str(tmp_path)) == set()
    assert ns.shared_hash_factor(str(tmp_path)) == "nvfp4_shared_swapset=off"
    monkeypatch.setenv(ns.SHARED_ENV, "1")
    assert ns.shared_expert_layers(str(tmp_path)) == {3, 4}
    assert ns.shared_hash_factor(str(tmp_path)).startswith("nvfp4_shared_swapset=")
    assert "layers.3.mlp.shared_experts.down_proj" in ns.claimed_modules(str(tmp_path))
    assert "layers.45.mlp.shared_experts.down_proj" not in ns.claimed_modules(str(tmp_path))


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
