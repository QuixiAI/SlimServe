# SPDX-License-Identifier: Apache-2.0
from slimserve.engine import engine_kwargs
from slimserve.registry import profile_blocked, resolve


def test_b70_fp8_recipe_is_gated_and_preserves_vision_and_native_mtp(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    profile = "qwen38-uncensored-fp8-4"
    plan = resolve(profile, "b70", 4, None)
    assert profile_blocked(profile, "b70")
    assert plan.entry_file == plan.model_dir
    assert plan.source["modalities"] == ["text", "image"]
    assert plan.chat_template_file.name == "qwen38_tool_calling.jinja"
    plan.model_dir.mkdir(parents=True)
    config = engine_kwargs(plan)["speculative_config"]
    assert config["model"] == str(plan.entry_file)
    assert config["method"] == "mtp"
    assert config["num_speculative_tokens"] == 3
    assert config["kv_cache_dtype"] == "fp8"
    assert plan.engine["compilation_config"]["max_cudagraph_capture_size"] == 128
    assert plan.env["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] == "120"
