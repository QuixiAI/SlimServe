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
    # KDA layer 4: BF16 in native, not a swap candidate; a self-quant one.
    for suffix, n, k in [
        ("o_proj", 128, 256),
        ("q_proj", 256, 128),
        ("k_proj", 256, 128),
        ("v_proj", 256, 128),
        ("b_proj", 8, 128),
        ("f_a_proj", 128, 128),
        ("g_a_proj", 128, 128),
    ]:
        w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
        nat[f"{L}.4.self_attn.{suffix}.weight"] = w
        con[f"{L}.4.self_attn.{suffix}.weight"] = w
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
            torch.zeros(128, 256, dtype=torch.bfloat16),
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


def _dequant(q, scale):
    return fp8_swapset.dequant_bf16(q, scale).float()


def test_quantize_block_is_absmax_over_448_with_a_ragged_last_block():
    torch.manual_seed(1)
    w = (torch.randn(200, 256) * 0.05).to(torch.bfloat16)
    q, s = fp8_swapset.quantize_block(w)
    assert q.shape == (200, 256) and q.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 2) and s.dtype == torch.float32
    blk = w[128:200, 128:256].float().abs().amax()
    assert torch.isclose(s[1, 1], blk / 448.0)
    err = (_dequant(q, s) - w.float()).abs()
    # e4m3 keeps 3 mantissa bits: half an ulp of the block's absmax at worst.
    assert err.max() <= w.float().abs().max() / 16 * 1.01
    assert err.norm() / w.float().norm() < 0.05
    with pytest.raises(ValueError, match="multiple of 128"):
        fp8_swapset.quantize_block(torch.zeros(128, 100, dtype=torch.bfloat16))


def test_quantize_beta_lays_each_rank_in_its_own_block():
    torch.manual_seed(2)
    w = (torch.randn(8, 256) * 0.05).to(torch.bfloat16)
    q, s = fp8_swapset.quantize_beta(w, tp_size=2)
    assert q.shape == (2 * fp8_swapset.BETA_ROWS, 256) and s.shape == (2, 2)
    d = _dequant(q, s)
    for r in range(2):
        rows = d[r * fp8_swapset.BETA_ROWS : (r + 1) * fp8_swapset.BETA_ROWS]
        ref = w[4 * r : 4 * r + 4].float()
        assert (rows[:4] - ref).abs().max() <= ref.abs().max() / 16 * 1.01
        assert (rows[4:] == 0).all()
        # The scale is the rank's own absmax, not the whole tensor's.
        assert torch.isclose(
            s[r, 0], w[4 * r : 4 * r + 4, :128].float().abs().amax() / 448
        )
    with pytest.raises(ValueError, match="do not shard"):
        fp8_swapset.quantize_beta(w, tp_size=3)


@pytest.mark.parametrize("tp_size", [0, -4])
def test_beta_quantization_rejects_nonpositive_shards(tp_size):
    with pytest.raises(ValueError, match="must be positive"):
        fp8_swapset.quantize_beta(torch.ones(8, 128), tp_size=tp_size)


@pytest.mark.parametrize("tp_size", [None, 0, -4])
def test_kda_builder_rejects_nonpositive_shards_before_reading_weights(tp_size):
    with pytest.raises(SystemExit, match="positive --tp-size"):
        fp8_swapset.build(
            "unused-native", "unused-model", self_quant_kda=True, tp_size=tp_size
        )


def test_self_quant_kda_extends_sidecar_manifest_and_targets(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    with pytest.raises(SystemExit, match="tp-size"):
        fp8_swapset.build(str(native), str(conv), self_quant_kda=True)
    out = fp8_swapset.build(str(native), str(conv), self_quant_kda=True, tp_size=2)
    from safetensors.torch import load_file

    tensors = load_file(str(out))
    kda = [n for n in tensors if f"{L}.4." in n]
    assert sorted(kda) == sorted(
        f"{L}.4.self_attn.{p}.{t}"
        for p in [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
            "o_proj",
        ]
        for t in ["weight", "weight_scale"]
    )
    assert tensors[f"{L}.4.self_attn.b_proj.weight"].shape == (256, 128)
    assert tensors[f"{L}.4.self_attn.b_proj.weight_scale"].shape == (2, 1)
    assert tensors[f"{L}.4.self_attn.q_proj.weight_scale"].shape == (2, 1)
    assert tensors[f"{L}.4.self_attn.o_proj.weight_scale"].shape == (1, 2)
    # The native-FP8 twins are still there, byte for byte.
    assert f"{L}.3.self_attn.o_proj.weight" in tensors
    manifest = json.loads((conv / fp8_swapset.MANIFEST_FILE).read_text())
    assert manifest["tp_size"] == 2 and manifest["beta_rows"] == 128
    assert manifest["self_quantized"] == sorted(n for n in kda if n.endswith(".weight"))
    assert (
        "language_model.model.layers.4.self_attn.in_proj_qkvgfab" in manifest["modules"]
    )
    assert "language_model.model.layers.4.self_attn.o_proj" in manifest["modules"]
    targets = manifest["config_group"]["targets"]
    assert "re:.*\\.layers\\.(4)\\.self_attn\\.in_proj_qkvgfab$" in targets
    assert "re:.*\\.layers\\.(3|4)\\.self_attn\\.o_proj$" in targets
    # The model asks for the padded beta shard only where the manifest says so.
    assert (
        fp8_swapset.beta_shard_rows(
            str(conv), "model.layers.4.self_attn.in_proj_qkvgfab"
        )
        == 128
    )
    assert (
        fp8_swapset.beta_shard_rows(
            str(conv), "model.layers.3.self_attn.in_proj_qkvgfab"
        )
        is None
    )
    monkeypatch.setenv(fp8_swapset.SWAPSET_ENV, "0")
    assert (
        fp8_swapset.beta_shard_rows(
            str(conv), "model.layers.4.self_attn.in_proj_qkvgfab"
        )
        is None
    )


def test_plain_sidecar_asks_for_no_beta_padding(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    fp8_swapset.build(str(native), str(conv))
    manifest = json.loads((conv / fp8_swapset.MANIFEST_FILE).read_text())
    assert "self_quantized" not in manifest and "tp_size" not in manifest
    assert (
        fp8_swapset.beta_shard_rows(
            str(conv), "model.layers.4.self_attn.in_proj_qkvgfab"
        )
        is None
    )


def test_loader_refuses_a_self_quant_sidecar_of_another_tp(tmp_path, monkeypatch):
    monkeypatch.delenv(fp8_swapset.SWAPSET_ENV, raising=False)
    native, conv = _fake_checkpoints(tmp_path)
    fp8_swapset.build(str(native), str(conv), self_quant_kda=True, tp_size=2)
    import vllm.model_executor.models.glm5_next as m

    monkeypatch.setattr(m, "get_tensor_model_parallel_world_size", lambda: 2)
    subs, extras = _load_fp8_swapset(str(conv))
    assert subs[f"{L}.4.self_attn.b_proj.weight"].shape == (256, 128)
    monkeypatch.setattr(m, "get_tensor_model_parallel_world_size", lambda: 4)
    with pytest.raises(ValueError, match="tp_size=2, serving with 4"):
        _load_fp8_swapset(str(conv))
