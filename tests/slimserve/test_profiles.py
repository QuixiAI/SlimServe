# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The profile gate is the whole point of slimserve: it must refuse anything
that is not a tested configuration, before a multi-hundred-GiB load finds out.
"""

import signal
from unittest.mock import Mock

import pytest

from slimserve import registry
from slimserve.engine import engine_kwargs, serve_argv
from slimserve.registry import ProfileError, files_for, resolve
from slimserve.server import Server
from slimserve.stream import FrameFilter, visible_text


def test_every_profile_resolves_on_a_platform_it_claims():
    for profile_id in registry.profile_ids():
        entry = registry.describe(profile_id)
        for platform in entry["platforms"]:
            plan = resolve(profile_id, platform, entry["gpus"], None)
            assert plan.quant.allowed_on(platform, entry["gpus"])
            assert plan.engine, "a profile with no engine settings would serve nothing"


def test_quant_too_large_for_the_profile_names_a_bigger_one():
    with pytest.raises(ProfileError, match="try glm52-4"):
        resolve("glm52-2", "mi300x", 8, "Q4_K")


def test_profile_rejected_on_an_unsupported_platform():
    with pytest.raises(ProfileError, match="not supported"):
        resolve("k3-6", "a100", 8, None)


def test_profile_rejected_when_the_machine_is_too_small():
    with pytest.raises(ProfileError, match="needs 8 GPUs"):
        resolve("glm52-8", "mi300x", 6, None)


def test_unknown_names_are_rejected_with_the_legal_set():
    with pytest.raises(ProfileError, match="glm52-2"):
        registry.describe("nope")
    with pytest.raises(ProfileError, match="IQ2_XXS"):
        resolve("glm52-2", "mi300x", 2, "Q9_Z")


def test_platform_override_replaces_the_mi300x_kv_budget():
    """A100 has 80 GB cards; the MI300X byte budget would not fit."""
    amd = resolve("glm52-4", "mi300x", 4, "Q2_K")
    nvidia = resolve("glm52-4", "a100", 4, "Q2_K")
    assert "kv_cache_memory_bytes" in amd.engine
    assert "kv_cache_memory_bytes" not in nvidia.engine
    assert nvidia.engine["gpu_memory_utilization"] == 0.92
    assert nvidia.env == {}, "AITER is a ROCm switch"


def test_kimi_needs_the_native_kv_dtype():
    """This fork defaults the cache to fp8, which K3 cannot use."""
    for profile_id in ("k3-6", "k3-8"):
        assert resolve(profile_id, "mi300x", 8, None).engine["kv_cache_dtype"] == "auto"


def test_registry_contains_only_the_supported_model_artifacts():
    data = registry._registry()
    assert set(data["sources"]) == {"glm52-vision", "kimi-k3", "dsv4-flash"}
    glm = data["sources"]["glm52-vision"]
    kimi = data["sources"]["kimi-k3"]
    deepseek = data["sources"]["dsv4-flash"]
    assert set(glm["quants"]) == {
        "IQ2_XXS",
        "Q2_K",
        "Q4_K",
    }
    assert [entry["path"] for entry in glm["shared"]] == [
        "mmproj-GLM-5.2-Vision-f16.gguf",
        "chat_template.jinja",
    ]
    assert glm["speculator"]["repo"] == "RedHatAI/GLM-5.2-speculator.dspark"
    assert {
        entry["path"] for quant in glm["quants"].values() for entry in quant["files"]
    } == {
        f"antirez-routed/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K-"
        f"{shard:05d}-of-00005.gguf"
        for shard in range(1, 6)
    } | {
        f"antirez-routed/GLM-5.2-UD-Q2_K_RoutedQ2K-{shard:05d}-of-00006.gguf"
        for shard in range(1, 7)
    } | {
        f"antirez-routed/GLM-5.2-UD-Q4_K_RoutedQ4K-{shard:05d}-of-00010.gguf"
        for shard in range(1, 11)
    }

    assert set(kimi["quants"]) == {"IQ2_XXS-Q2_K"}
    assert kimi["quants"]["IQ2_XXS-Q2_K"]["assembly"]["output"] == (
        "Kimi-K3-IQ2_XXS-Q2_K.gguf"
    )
    assert [entry["path"] for entry in kimi["shared"]] == ["mmproj-BF16.gguf"]
    assert kimi["speculator"]["file"]["path"] == "Kimi-K3-DSpark-Q8_0.gguf"

    assert set(deepseek["quants"]) == {
        "MXFP4",
        "Q4_K",
        "Q4K-tail",
        "IQ2_XXS",
    }
    assert {
        entry["path"]
        for quant in deepseek["quants"].values()
        for entry in quant["files"]
    } == {
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf",
        "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-"
        "Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf",
        "DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-"
        "Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf",
        "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-"
        "Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf",
    }


def test_kimi_uses_the_registered_q8_dspark_gguf():
    assert "speculative_config" not in engine_kwargs(resolve("k3-6", "mi300x", 8, None))
    plan = resolve("k3-8", "mi300x", 8, None)
    speculative = engine_kwargs(plan)["speculative_config"]

    assert speculative["model"].endswith(
        "/Kimi-K3-DSpark-Q8_0-GGUF/Kimi-K3-DSpark-Q8_0.gguf"
    )
    assert speculative["method"] == "dspark"
    assert speculative["num_speculative_tokens"] == 7
    assert speculative["quantization"] == "gguf"
    assert speculative["attention_backend"] == "TRITON_ATTN"
    assert speculative["disable_draft_cudagraphs"] is True
    assert "draft_tensor_parallel_size" not in speculative

    [draft] = [entry for entry in files_for(plan) if entry["role"] == "speculator"]
    assert draft["bytes"] == 2390153888
    assert draft["url"] == (
        "https://huggingface.co/Lucebox/Kimi-K3-DSpark-Q8_0-GGUF/"
        "resolve/main/Kimi-K3-DSpark-Q8_0.gguf"
    )


def test_serve_argv_renders_flags_the_api_server_accepts():
    argv = serve_argv(resolve("glm52-2", "mi300x", 2, None), "127.0.0.1", 8000)
    assert "--enable-prefix-caching" in argv
    assert "--tensor-parallel-size" in argv
    # Booleans are flags, never "--flag True".
    assert "True" not in argv
    assert (
        argv[argv.index("--attention-config") + 1] == '{"sparse_mla_force_mqa": true}'
    )


def test_engine_kwargs_drop_server_only_settings():
    kwargs = engine_kwargs(resolve("k3-6", "mi300x", 6, None))
    assert "served_model_name" not in kwargs
    assert kwargs["tensor_parallel_size"] == 6
    assert kwargs["model"].endswith(".gguf")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Paris. <|close|> response <|sep|>", "Paris."),
        ("no frame here", "no frame here"),
        ("  4 <|end_of_msg|>", "4"),
    ],
)
def test_xtml_frame_is_stripped_from_the_answer(raw, expected):
    assert visible_text(raw) == expected


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        # A control token split across deltas must never reach the terminal.
        (["Par", "is.", " <", "|clo", "se|> response"], "Paris. "),
        (["Hello ", "world", "!"], "Hello world!"),
        # A bare '<' in ordinary prose is not the start of a frame.
        (["a < b and c ", "< d"], "a < b and c < d"),
        (["x", "<|", "end_of_msg|>", "dropped"], "x"),
    ],
)
def test_streaming_filter_never_emits_half_a_control_token(deltas, expected):
    frame = FrameFilter()
    assert "".join(frame.feed(d) for d in deltas) + frame.flush() == expected


def test_profiles_use_their_validated_graph_mode():
    """Only Kimi TP8 needs eager target execution for its stateful decode path."""
    for profile_id in registry.profile_ids():
        for platform in registry.describe(profile_id)["platforms"]:
            engine = resolve(
                profile_id, platform, registry.describe(profile_id)["gpus"], None
            ).engine
            cudagraph_mode = engine.get("compilation_config", {}).get("cudagraph_mode")
            if profile_id == "k3-8":
                assert cudagraph_mode == "NONE"
            else:
                assert cudagraph_mode not in (None, "NONE"), profile_id


def test_obsolete_enforce_eager_switch_stays_removed():
    """Graph mode is selected only through the compilation configuration."""
    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import EngineArgs

    assert not hasattr(ModelConfig, "enforce_eager")
    assert not hasattr(EngineArgs, "enforce_eager")


def test_server_stops_its_entire_worker_process_group(monkeypatch):
    process = Mock(pid=4321)
    process.poll.side_effect = [None, 0]
    signals = []
    monkeypatch.setattr(
        "slimserve.server.os.killpg", lambda pid, sig: signals.append((pid, sig))
    )
    server = Server.__new__(Server)
    server.process = process
    server._log = None

    server.stop()

    assert signals == [(4321, signal.SIGTERM)]
