# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The profile gate is the whole point of slimserve: it must refuse anything
that is not a tested configuration, before a multi-hundred-GiB load finds out.
"""

import pytest

from slimserve import registry
from slimserve.engine import engine_kwargs, serve_argv
from slimserve.registry import ProfileError, resolve
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


def test_no_profile_ever_runs_eager():
    """enforce_eager says performance does not matter, which is never true here.

    K3 carried it from bring-up, where it was a debugging crutch, and it stayed
    long enough to be measured as a throughput problem: a 93-layer decode step is
    thousands of tiny launches, so eager makes the loop launch-bound and no
    kernel work underneath it can help. Every profile gets CUDA graphs.
    """
    for profile_id in registry.profile_ids():
        for platform in registry.describe(profile_id)["platforms"]:
            engine = resolve(
                profile_id, platform, registry.describe(profile_id)["gpus"], None
            ).engine
            assert not engine.get("enforce_eager"), profile_id
            assert engine.get("compilation_config", {}).get("cudagraph_mode"), (
                profile_id
            )
