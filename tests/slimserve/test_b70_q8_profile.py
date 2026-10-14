# SPDX-License-Identifier: Apache-2.0
from slimserve.engine import engine_kwargs
from slimserve.registry import files_for, profile_blocked, resolve


def test_b70_q8_recipe_uses_embedded_mtp_and_stays_gated():
    profile = "qwen38-hauhau-q8-4"
    plan = resolve(profile, "b70", 4, None)
    assert profile_blocked(profile, "b70")
    assert plan.entry_file.suffix == ".gguf"
    assert plan.engine["quantization"] == "gguf"
    draft = engine_kwargs(plan)["speculative_config"]
    assert draft["model"] == str(plan.entry_file)
    assert draft["method"] == "mtp"
    assert draft["num_speculative_tokens"] == 1
    assert draft["kv_cache_dtype"] == "fp8"
    assert plan.engine["compilation_config"]["max_cudagraph_capture_size"] == 64
    artifacts = files_for(plan)
    target = next(f for f in artifacts if f["role"] == "model")
    spec = next(f for f in artifacts if f["role"] == "speculator")
    assert (target["path"], target["sha256"]) == (spec["path"], spec["sha256"])


def test_b70_q8_recipe_serves_vision_through_the_published_mmproj():
    plan = resolve("qwen38-hauhau-q8-4", "b70", 4, None)
    assert plan.source["modalities"] == ["text", "image"]
    assert plan.engine["tool_call_parser"] == "qwen3_xml"
    shared = [f for f in files_for(plan) if f["role"] == "shared"]
    assert [f["path"] for f in shared] == [
        "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
    ]
    assert plan.source["revision"] in shared[0]["url"]
