"""The F32 sidecar override for GLM-5.3-Flash conversions (loader + builder)."""

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from slimserve import f32_overrides
from vllm.model_executor.models.glm5_next import (
    F32_OVERRIDES_ENV,
    F32_OVERRIDES_FILE,
    _load_f32_overrides,
    iter_with_overrides,
)

NAME = "model.language_model.layers.3.mlp.gate.e_score_correction_bias"


def _stream():
    yield NAME, torch.full((288,), 7.0, dtype=torch.bfloat16)
    yield "model.language_model.layers.3.mlp.gate.weight", torch.zeros(2, 2)


def test_overrides_replace_matching_names_only(tmp_path, monkeypatch):
    monkeypatch.delenv(F32_OVERRIDES_ENV, raising=False)
    save_file(
        {NAME: torch.full((288,), 7.01, dtype=torch.float32)},
        str(tmp_path / F32_OVERRIDES_FILE),
    )
    overrides = _load_f32_overrides(str(tmp_path))
    assert set(overrides) == {NAME}
    got = dict(iter_with_overrides(_stream(), overrides))
    assert got[NAME].dtype == torch.float32 and float(got[NAME][0]) == pytest.approx(
        7.01
    )
    assert (
        got["model.language_model.layers.3.mlp.gate.weight"].dtype == torch.float32
    )  # untouched


def test_missing_file_and_kill_switch_are_no_ops(tmp_path, monkeypatch):
    monkeypatch.delenv(F32_OVERRIDES_ENV, raising=False)
    assert _load_f32_overrides(str(tmp_path)) == {}
    assert _load_f32_overrides(None) == {}
    save_file({NAME: torch.ones(288)}, str(tmp_path / F32_OVERRIDES_FILE))
    monkeypatch.setenv(F32_OVERRIDES_ENV, "0")
    assert _load_f32_overrides(str(tmp_path)) == {}
    assert dict(iter_with_overrides(_stream(), {}))[NAME].dtype == torch.bfloat16


def test_non_f32_sidecar_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv(F32_OVERRIDES_ENV, raising=False)
    save_file(
        {NAME: torch.ones(288, dtype=torch.bfloat16)},
        str(tmp_path / F32_OVERRIDES_FILE),
    )
    with pytest.raises(ValueError, match="not float32"):
        _load_f32_overrides(str(tmp_path))


def _write_shard(path, tensors):
    """Minimal safetensors writer so the builder's header parser sees real files."""
    save_file(tensors, str(path))


def test_builder_selects_downcast_tensors_and_skips_mtp_layer(tmp_path):
    native = tmp_path / "native"
    conv = tmp_path / "conv"
    native.mkdir()
    conv.mkdir()
    t = torch.arange(288, dtype=torch.float32) / 7
    a_log = torch.arange(64, dtype=torch.float32) / 3
    _write_shard(
        native / "model-00001-of-00001.safetensors",
        {
            NAME: t,
            "model.language_model.layers.0.self_attn.A_log": a_log,
            "model.language_model.layers.45.mlp.gate.e_score_correction_bias": (
                t.clone()
            ),
            "model.language_model.layers.3.mlp.gate.weight": torch.zeros(
                4, 4, dtype=torch.bfloat16
            ),
        },
    )
    _write_shard(
        conv / "model-00001-of-00001.safetensors",
        {
            NAME: t.to(torch.bfloat16),
            # already F32 in the conversion: must not be selected
            "model.language_model.layers.0.self_attn.A_log": a_log.clone(),
            "model.language_model.layers.45.mlp.gate.e_score_correction_bias": t.to(
                torch.bfloat16
            ),
            "model.language_model.layers.3.mlp.gate.weight": torch.zeros(
                4, 4, dtype=torch.bfloat16
            ),
        },
    )
    out = f32_overrides.build(str(native), str(conv))
    assert out == conv / f32_overrides.OVERRIDES_FILE
    with open(out, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    names = {k for k in header if k != "__metadata__"}
    assert names == {NAME}
    assert header[NAME]["dtype"] == "F32"
    restored = _load_f32_overrides(str(conv))
    assert torch.equal(restored[NAME], t)
