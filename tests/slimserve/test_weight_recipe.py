# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from slimserve import fetch, registry, weight_recipe
from slimserve.f32_overrides import _headers


def _plan(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    monkeypatch.delenv("SLIMSERVE_F32_OVERRIDES", raising=False)
    monkeypatch.delenv("SLIMSERVE_FP8_SWAPSET", raising=False)
    plan = registry.resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    plan.model_dir.mkdir()
    (plan.model_dir / "config.json").write_text("{}")
    for name in weight_recipe.SIDECARS[:2]:
        save_file(
            {"w": torch.ones(2)}, str(plan.model_dir / name), metadata={"source": "old"}
        )
    (plan.model_dir / "fp8-swapset.json").write_text(
        json.dumps({"source": "old", "tp_size": 4})
    )
    recipe = {
        **plan.weight_recipe,
        "artifact_digests": {
            name: weight_recipe.artifact_digest(plan.model_dir / name)
            for name in weight_recipe.SIDECARS
        },
    }
    return replace(
        plan,
        weight_recipe=recipe,
        quant=replace(
            plan.quant,
            files=[{"path": "config.json", "bytes": 2}],
        ),
    )


def test_recipe_belongs_to_the_rtx6000_record():
    plan = registry.resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    assert plan.weight_recipe["target_revision"] == plan.source["revision"]
    assert plan.weight_recipe["tp_size"] == plan.engine["tensor_parallel_size"] == 4
    assert plan.entry_file != plan.model_dir
    assert set(plan.weight_recipe["artifact_digests"]) == set(weight_recipe.SIDECARS)
    assert all(len(d) == 64 for d in plan.weight_recipe["artifact_digests"].values())
    a100 = registry.resolve("glm53-nvfp4-4", "a100", 4, None)
    assert a100.weight_recipe is None
    assert a100.entry_file == a100.model_dir


def test_fetch_prepares_verified_recipe_and_preserves_originals(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    build = Mock(side_effect=AssertionError("qualified local artifacts need no build"))
    monkeypatch.setattr(weight_recipe, "_build", build)
    fetch.ensure(plan, assume_yes=True)
    assert plan.entry_file.is_dir()
    assert (plan.entry_file / "config.json").resolve() == plan.model_dir / "config.json"
    assert (plan.entry_file / "fp8-swapset.safetensors").stat().st_ino != (
        plan.model_dir / "fp8-swapset.safetensors"
    ).stat().st_ino
    fetch.ensure(plan, assume_yes=True)
    build.assert_not_called()


def test_missing_sidecars_are_built_before_publication(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    original = plan.model_dir / "fp8-swapset.safetensors"
    data = original.read_bytes()
    original.unlink()

    def build(_plan, staging):
        assert not plan.entry_file.exists()
        for name in weight_recipe.SIDECARS:
            content = (
                data if name == original.name else (plan.model_dir / name).read_bytes()
            )
            (staging / name).write_bytes(content)

    monkeypatch.setattr(weight_recipe, "_build", build)
    fetch.ensure(plan, assume_yes=True)
    weight_recipe.validate(plan.entry_file, plan.weight_recipe)
    assert not original.exists()


def test_changed_tensor_bytes_are_rejected_on_next_start(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    fetch.ensure(plan, assume_yes=True)
    path = plan.entry_file / "fp8-swapset.safetensors"
    save_file({"w": torch.zeros(2)}, str(path), metadata={"source": "old"})
    with pytest.raises(ValueError, match="artifact digest mismatch"):
        fetch.ensure(plan, assume_yes=True)


@pytest.mark.parametrize("key", ["SLIMSERVE_F32_OVERRIDES", "SLIMSERVE_FP8_SWAPSET"])
def test_recipe_cannot_silently_disable_its_weights(tmp_path, monkeypatch, key):
    plan = _plan(tmp_path, monkeypatch)
    monkeypatch.setenv(key, "0")
    with pytest.raises(ValueError, match="requires"):
        fetch.ensure(plan, assume_yes=True)
    assert not plan.entry_file.exists()


def test_artifact_identity_ignores_only_local_provenance(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    for path, source in ((a, "host-a"), (b, "host-b")):
        save_file({"w": torch.ones(2)}, str(path), metadata={"source": source})
    assert weight_recipe.artifact_digest(a) == weight_recipe.artifact_digest(b)
    save_file({"other": torch.ones(2)}, str(b))
    assert weight_recipe.artifact_digest(a) != weight_recipe.artifact_digest(b)


@pytest.mark.parametrize("indexed", [False, True])
def test_builders_read_original_shards_not_their_previous_output(tmp_path, indexed):
    save_file(
        {"w": torch.ones(2, dtype=torch.bfloat16)}, str(tmp_path / "model.safetensors")
    )
    save_file({"w": torch.zeros(2)}, str(tmp_path / "f32-overrides.safetensors"))
    save_file({"w": torch.zeros(2)}, str(tmp_path / "fp8-swapset.safetensors"))
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {"w": "model.safetensors"},
                }
            )
        )
    header, shard, _ = _headers(str(tmp_path))["w"]
    assert header["dtype"] == "BF16"
    assert shard == str(tmp_path / "model.safetensors")
