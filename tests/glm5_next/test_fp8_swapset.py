"""The FP8 swap-set for GLM-5.3-Flash conversions: builder, manifest, config
group, loader substitution + scale injection, kill switch, hash factor."""

import json

import pytest
import torch
from safetensors.torch import save_file

from slimserve import fp8_swapset
from vllm.model_executor.models.glm5_next import (
    _load_fp8_swapset,
    iter_with_overrides,
)

L = "model.language_model.layers"
CT_GROUP = {
    "format": "float-quantized",
    "input_activations": {
        "dynamic": True,
        "group_size": 128,
        "num_bits": 8,
        "strategy": "group",
        "symmetric": True,
        "type": "float",
    },
    "output_activations": None,
    "targets": [
        "re:.*\\.layers\\.45\\.mlp\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)$"
    ],
    "weights": {
        "block_structure": [128, 128],
        "dynamic": False,
        "num_bits": 8,
        "strategy": "block",
        "symmetric": True,
        "type": "float",
    },
}


def _fake_checkpoints(tmp_path):
    """A native shard with FP8 swap tensors (and BF16 ones that must be left
    alone) and a conversion shard holding their BF16 dequant."""
    native, conv = tmp_path / "native", tmp_path / "conv"
    native.mkdir()
    conv.mkdir()
    torch.manual_seed(0)
    nat, con = {}, {}

    def fp8(name, n, k):
        w = torch.randn(n, k) * 0.02
        s = w.abs().amax().clamp(min=1e-6).view(1, 1) / 448.0
        q = (w / s).clamp(-448, 448).to(torch.float8_e4m3fn)
        nat[name + ".weight"] = q
        nat[name + ".weight_scale_inv"] = s.expand(
            (n + 127) // 128, k // 128
        ).contiguous()
        con[name + ".weight"] = (q.float() * s).to(torch.bfloat16)

    fp8(f"{L}.0.mlp.gate_proj", 256, 128)
    fp8(f"{L}.0.mlp.up_proj", 256, 128)
    fp8(f"{L}.3.self_attn.o_proj", 128, 256)
    fp8(f"{L}.3.mlp.shared_experts.down_proj", 128, 128)
    fp8(f"{L}.45.mlp.shared_experts.down_proj", 128, 128)  # MTP layer: skipped
    # KDA o_proj is BF16 in native: not a swap candidate.
    nat[f"{L}.4.self_attn.o_proj.weight"] = torch.randn(128, 128).to(torch.bfloat16)
    con[f"{L}.4.self_attn.o_proj.weight"] = nat[f"{L}.4.self_attn.o_proj.weight"]
    save_file(nat, str(native / "model-00001-of-00001.safetensors"))
    save_file(con, str(conv / "model-00001-of-00001.safetensors"))
    (conv / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "mixed-precision",
                    "config_groups": {"group_1": CT_GROUP},
                }
            }
        )
    )
    return native, conv


def test_builder_selects_fp8_twins_writes_manifest_and_group(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    out = fp8_swapset.build(str(native), str(conv))
    assert out == conv / fp8_swapset.SWAPSET_FILE
    from safetensors.torch import load_file

    tensors = load_file(str(out))
    names = sorted(tensors)
    assert names == sorted(
        [
            f"{L}.0.mlp.gate_proj.weight",
            f"{L}.0.mlp.gate_proj.weight_scale",
            f"{L}.0.mlp.up_proj.weight",
            f"{L}.0.mlp.up_proj.weight_scale",
            f"{L}.3.mlp.shared_experts.down_proj.weight",
            f"{L}.3.mlp.shared_experts.down_proj.weight_scale",
            f"{L}.3.self_attn.o_proj.weight",
            f"{L}.3.self_attn.o_proj.weight_scale",
        ]
    )
    assert tensors[f"{L}.3.self_attn.o_proj.weight"].dtype == torch.float8_e4m3fn
    assert tensors[f"{L}.3.self_attn.o_proj.weight_scale"].dtype == torch.float32
    manifest = json.loads((conv / fp8_swapset.MANIFEST_FILE).read_text())
    assert manifest["tensors"] == names
    assert manifest["modules"] == [
        "language_model.model.layers.0.mlp.gate_up_proj",
        "language_model.model.layers.3.mlp.shared_experts.down_proj",
        "language_model.model.layers.3.self_attn.o_proj",
    ]
    group = manifest["config_group"]
    assert group["weights"] == CT_GROUP["weights"]
    assert group["targets"] == [
        "re:.*\\.layers\\.(0)\\.mlp\\.gate_up_proj$",
        "re:.*\\.layers\\.(3)\\.mlp\\.shared_experts\\.down_proj$",
        "re:.*\\.layers\\.(3)\\.self_attn\\.o_proj$",
    ]
    # The DSA target must not reach the KDA o_proj of layer 4.
    import re

    pat = group["targets"][2][len("re:") :]
    assert re.match(pat, "language_model.model.layers.3.self_attn.o_proj")
    assert not re.match(pat, "language_model.model.layers.4.self_attn.o_proj")
    assert not re.match(pat, "language_model.model.layers.43.self_attn.o_proj")


def test_config_group_merge_hash_factor_and_kill_switch(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    assert fp8_swapset.hash_factor(str(conv)) == "fp8_swapset=off"
    assert not fp8_swapset.apply_config_group(str(conv), {"quant_method": "x"})
    fp8_swapset.build(str(native), str(conv))
    qc = {"quant_method": "compressed-tensors", "config_groups": {"group_1": CT_GROUP}}
    assert fp8_swapset.apply_config_group(str(conv), qc)
    assert set(qc["config_groups"]) == {"group_1", fp8_swapset.GROUP_NAME}
    assert qc["config_groups"][fp8_swapset.GROUP_NAME]["weights"]["strategy"] == "block"
    with pytest.raises(ValueError, match="compressed-tensors"):
        fp8_swapset.apply_config_group(str(conv), {"quant_method": "fp8"})
    on = fp8_swapset.hash_factor(str(conv))
    assert on.startswith("fp8_swapset=") and on != "fp8_swapset=off"
    monkeypatch.setenv(fp8_swapset.SWAPSET_ENV, "0")
    assert fp8_swapset.hash_factor(str(conv)) == "fp8_swapset=off"
    assert not fp8_swapset.apply_config_group(str(conv), qc)
    assert _load_fp8_swapset(str(conv)) == ({}, {})


def test_loader_substitutes_weights_and_injects_scales(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    fp8_swapset.build(str(native), str(conv))
    subs, extras = _load_fp8_swapset(str(conv))
    assert len(subs) == 4 and len(extras) == 4
    assert all(v.dtype == torch.float8_e4m3fn for v in subs.values())
    assert all(v.dtype == torch.float32 for v in extras.values())

    def stream():
        yield f"{L}.0.mlp.gate_proj.weight", torch.zeros(256, 128, dtype=torch.bfloat16)
        yield f"{L}.0.mlp.up_proj.weight", torch.zeros(256, 128, dtype=torch.bfloat16)
        yield (
            f"{L}.3.self_attn.o_proj.weight",
            torch.zeros(128, 256, dtype=torch.bfloat16),
        )
        yield (
            f"{L}.3.mlp.shared_experts.down_proj.weight",
            torch.zeros(128, 128, dtype=torch.bfloat16),
        )
        yield (
            f"{L}.4.self_attn.o_proj.weight",
            torch.zeros(128, 128, dtype=torch.bfloat16),
        )

    got = list(iter_with_overrides(stream(), subs, extras, strict=True))
    names = [n for n, _ in got]
    assert names[:5] == [n for n, _ in stream()]  # order of the stream kept
    assert sorted(names[5:]) == sorted(extras)  # scales injected after it
    d = dict(got)
    assert d[f"{L}.3.self_attn.o_proj.weight"].dtype == torch.float8_e4m3fn
    assert d[f"{L}.4.self_attn.o_proj.weight"].dtype == torch.bfloat16  # untouched
    assert d[f"{L}.3.self_attn.o_proj.weight_scale"].shape == (1, 2)

    def short_stream():
        yield f"{L}.0.mlp.gate_proj.weight", torch.zeros(256, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="never appeared"):
        list(iter_with_overrides(short_stream(), subs, extras, strict=True))


def test_manifest_and_file_must_agree(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    fp8_swapset.build(str(native), str(conv))
    manifest = json.loads((conv / fp8_swapset.MANIFEST_FILE).read_text())
    manifest["tensors"].append("extra")
    (conv / fp8_swapset.MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="do not match the manifest"):
        _load_fp8_swapset(str(conv))
