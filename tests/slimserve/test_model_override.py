# SPDX-License-Identifier: Apache-2.0
"""--model: serve an operator's checkpoint with a profile's qualified config.

An override swaps weights only. The profile's engine arguments, drafter and
kernels were qualified against its registered model, so a checkpoint that is
not interchangeable with it must be refused rather than served.
"""
import json

import pytest

from slimserve import fetch, registry
from slimserve.registry import ProfileError, parse_model_override

BASE = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "vocab_size": 151552,
    "hidden_size": 4096,
    "num_hidden_layers": 45,
    "num_attention_heads": 64,
    "max_position_embeddings": 1048576,
    "n_routed_experts": 288,
    "num_experts_per_tok": 8,
    "quantization_config": {"quant_method": "compressed-tensors"},
}


def _model(tmp_path, name, **changes):
    directory = tmp_path / name
    directory.mkdir()
    config = {**BASE, **changes}
    (directory / "config.json").write_text(json.dumps(config))
    return directory


def test_repo_id_resolves_under_the_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    override = parse_model_override("orcarouter/GLM-5.3-Flash-Uncensored-NVFP4")
    assert override.repo == "orcarouter/GLM-5.3-Flash-Uncensored-NVFP4"
    assert override.directory == tmp_path / "orcarouter--GLM-5.3-Flash-Uncensored-NVFP4"
    assert override.base_url.endswith("/GLM-5.3-Flash-Uncensored-NVFP4/resolve/main")


def test_local_directory_is_served_in_place(tmp_path):
    directory = _model(tmp_path, "local-tune")
    override = parse_model_override(str(directory))
    assert override.repo is None and override.directory == directory.resolve()
    # Nothing to download for a directory that is already on disk.
    assert fetch.override_entries(override) == []


def test_a_missing_directory_and_a_bad_repo_id_are_rejected(tmp_path):
    with pytest.raises(ProfileError):
        parse_model_override(str(tmp_path / "nope"))
    with pytest.raises(ProfileError):
        parse_model_override("not-a-repo-id")
    with pytest.raises(ProfileError):
        parse_model_override("   ")


def test_plan_serves_the_override_and_remembers_the_registered_model(tmp_path):
    plan = registry.resolve("glm53f-nvfp4-8", "a100", 8, None)
    directory = _model(tmp_path, "tune")
    overridden = registry.replace_override(plan, parse_model_override(str(directory)))
    assert overridden.model_dir == directory.resolve()
    assert overridden.entry_file == directory.resolve()
    assert overridden.registered_model_dir == plan.model_dir
    # Only the checkpoint moves: the qualified configuration is untouched.
    assert overridden.engine == plan.engine
    assert overridden.speculative == plan.speculative


def test_an_interchangeable_checkpoint_has_no_conflicts(tmp_path):
    registered = _model(tmp_path, "registered")
    tune = _model(tmp_path, "tune")
    assert registry.override_conflicts(registered, tune) == []


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"architectures": ["Qwen3MoeForCausalLM"]}, "architectures"),
        ({"vocab_size": 152000}, "vocab_size"),
        ({"num_hidden_layers": 40}, "num_hidden_layers"),
        ({"n_routed_experts": 128}, "n_routed_experts"),
        ({"quantization_config": {"quant_method": "gptq"}}, "quantization"),
        ({"quantization_config": None}, "quantization"),
    ],
)
def test_a_different_model_is_reported_key_by_key(tmp_path, change, expected):
    registered = _model(tmp_path, "registered")
    other = _model(tmp_path, "other", **change)
    problems = registry.override_conflicts(registered, other)
    assert problems and any(p.startswith(expected) for p in problems)


def test_nested_text_config_participates(tmp_path):
    registered = _model(tmp_path, "registered")
    nested = tmp_path / "nested"
    nested.mkdir()
    config = {k: v for k, v in BASE.items() if k != "num_hidden_layers"}
    config["text_config"] = {"num_hidden_layers": 45}
    (nested / "config.json").write_text(json.dumps(config))
    assert registry.override_conflicts(registered, nested) == []


def test_a_checkpoint_without_a_config_is_an_error(tmp_path):
    registered = _model(tmp_path, "registered")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ProfileError):
        registry.override_conflicts(registered, empty)


def test_same_repo_name_under_two_owners_never_shares_a_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    theirs = parse_model_override("nvidia/GLM-5.3-Flash-NVFP4")
    ours = parse_model_override("RedHatAI/GLM-5.3-Flash-NVFP4")
    assert theirs.directory != ours.directory
    # And neither collides with a registered source's local_dir, which is the
    # bare repo name.
    assert theirs.directory.name != "GLM-5.3-Flash-NVFP4"
