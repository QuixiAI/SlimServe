# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""slimserve.fp8_swapset: block quantization round trip, the compressed-tensors
targets it writes, and the config-group injection (group added, swapped
modules leave the ignore list, off by environment)."""

import json

import pytest
import torch

from slimserve import fp8_swapset


def test_block_quant_round_trip_and_partial_row_block():
    torch.manual_seed(0)
    w = (torch.randn(200, 256) * 0.02).to(torch.bfloat16)
    q, s = fp8_swapset.quantize_block(w)
    assert q.shape == (200, 256) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 2) and s.dtype == torch.float32
    d = fp8_swapset.dequant_bf16(q, s)
    rel = (d.float() - w.float()).norm() / w.float().norm()
    assert rel < 0.05
    with pytest.raises(ValueError):
        fp8_swapset.quantize_block(torch.zeros(8, 200, dtype=torch.bfloat16))


def test_targets_are_per_module_layer_sets():
    names = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.2.mlp.up_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
    ]
    targets = fp8_swapset.targets_for(names)
    assert targets == [
        r"re:.*\.layers\.(0|2)\.mlp\.gate_up_proj$",
        r"re:.*\.layers\.(3)\.self_attn\.o_proj$",
    ]


def _write_manifest(tmp_path):
    manifest = {
        "file": fp8_swapset.SWAPSET_FILE,
        "tensors": [
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight_scale",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight_scale",
        ],
        "modules": ["language_model.model.layers.0.mlp.gate_up_proj"],
        "config_group": {"targets": [r"re:.*\.layers\.(0)\.mlp\.gate_up_proj$"]},
    }
    (tmp_path / fp8_swapset.MANIFEST_FILE).write_text(json.dumps(manifest))


def test_apply_config_group_adds_group_and_prunes_ignore(tmp_path, monkeypatch):
    _write_manifest(tmp_path)
    cfg = {
        "quant_method": "compressed-tensors",
        "config_groups": {"group_0": {}},
        "ignore": [
            "model.language_model.layers.0.mlp.gate_proj",
            "model.language_model.layers.0.mlp.up_proj",
            "model.language_model.layers.1.mlp.gate_proj",
            "lm_head",
        ],
    }
    assert fp8_swapset.apply_config_group(str(tmp_path), cfg)
    assert fp8_swapset.GROUP_NAME in cfg["config_groups"]
    assert cfg["ignore"] == ["model.language_model.layers.1.mlp.gate_proj", "lm_head"]
    assert fp8_swapset.hash_factor(str(tmp_path)).startswith("fp8_swapset=")
    assert fp8_swapset.hash_factor(str(tmp_path)) != "fp8_swapset=off"
    monkeypatch.setenv(fp8_swapset.SWAPSET_ENV, "0")
    assert fp8_swapset.manifest_path(str(tmp_path)) is None
    assert not fp8_swapset.apply_config_group(str(tmp_path), {"quant_method": "x"})
    assert fp8_swapset.hash_factor(str(tmp_path)) == "fp8_swapset=off"


def test_apply_config_group_refuses_non_compressed_tensors(tmp_path):
    _write_manifest(tmp_path)
    with pytest.raises(ValueError):
        fp8_swapset.apply_config_group(str(tmp_path), {"quant_method": "fp8"})
    assert not fp8_swapset.apply_config_group(str(tmp_path / "missing"), {})
