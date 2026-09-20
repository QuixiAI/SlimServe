# SPDX-License-Identifier: Apache-2.0
from slimserve.engine import engine_kwargs, serve_argv
from slimserve.registry import files_for, profile_blocked, resolve


def test_b70_bf16_recipe_preserves_vision_and_qualification_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    profile = "qwen38-bf16-4"
    plan = resolve(profile, "b70", 4, None)
    assert profile_blocked(profile, "b70")
    assert plan.entry_file == plan.model_dir
    assert plan.source["modalities"] == ["text", "image"]
    assert "quantization" not in plan.engine
    assert plan.engine["kv_cache_dtype"] == "bfloat16"
    assert plan.engine["attention_backend"] == "FLASH_ATTN"
    argv = serve_argv(plan, "127.0.0.1", 8000)
    assert argv[argv.index("--kv-cache-dtype") + 1] == "bfloat16"
    assert argv[argv.index("--attention-backend") + 1] == "FLASH_ATTN"
    plan.model_dir.mkdir(parents=True)
    draft = engine_kwargs(plan)["speculative_config"]
    assert draft["model"] == str(plan.entry_file)
    assert draft["method"] == "mtp"
    assert draft["num_speculative_tokens"] == 1
    assert draft["kv_cache_dtype"] == "fp8"
    assert plan.engine["shutdown_timeout"] == 120
    assert plan.engine["compilation_config"]["max_cudagraph_capture_size"] == 64
    artifacts = files_for(plan)
    assert all(f["bytes"] > 0 for f in artifacts)
    assert any("preprocessor" in f["path"] for f in artifacts)
