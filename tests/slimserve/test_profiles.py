# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The profile gate is the whole point of slimserve: it must refuse anything
that is not a tested configuration, before a multi-hundred-GiB load finds out.
"""

import hashlib
import signal
from dataclasses import replace
from unittest.mock import Mock

import pytest

from slimserve import cli, fetch, registry
from slimserve.engine import engine_kwargs, serve_argv
from slimserve.hardware import Machine
from slimserve.registry import ProfileError, files_for, resolve
from slimserve.server import Server
from slimserve.smoke import compatible_profile_ids, validate_acceleration
from slimserve.stream import FrameFilter, visible_text


def _big_enough(profile_id: str, platform: str) -> int:
    """The smallest machine that can hold this profile's default quant."""
    if registry.platform_gate(platform) != "memory":
        return 0
    entry = registry.describe(profile_id)
    source = registry._registry()["sources"][entry["source"]]
    return source["quants"][entry["default_quant"]]["min_memory_bytes"][platform]


def test_every_profile_resolves_on_a_platform_it_claims():
    for profile_id in registry.profile_ids():
        entry = registry.describe(profile_id)
        for platform in entry["platforms"]:
            memory = _big_enough(profile_id, platform)
            plan = resolve(profile_id, platform, entry["gpus"], None, memory)
            assert plan.quant.allowed_on(platform, entry["gpus"], memory)
            assert plan.engine, "a profile with no engine settings would serve nothing"


def test_every_profile_uses_dspark_with_turboquant():
    for profile_id in registry.profile_ids():
        entry = registry.describe(profile_id)
        for platform in entry["platforms"]:
            memory = _big_enough(profile_id, platform)
            plan = resolve(profile_id, platform, entry["gpus"], None, memory)
            if not plan.speculative:
                assert "speculative_config" not in engine_kwargs(plan)
                continue
            config = engine_kwargs(plan)["speculative_config"]
            source = registry._registry()["sources"][entry["source"]]
            registered = source["speculator"]["engine"]["method"]
            assert config["method"] == registered
            if registered == "dspark":
                assert config["attention_backend"] == "TURBOQUANT"
                assert config["kv_cache_dtype"] == "turboquant_k8v4"


def test_no_spec_cli_flag_disables_the_resolved_speculator(monkeypatch):
    plan = resolve("dsv4-q4ktail-2", "a100", 2, "IQ2_XXS")
    monkeypatch.setattr(cli.hardware, "detect", Mock(return_value=Mock(
        known=True,
        platform="a100",
        count=2,
        memory_bytes=0,
        device_name="A100",
    )))
    monkeypatch.setattr(cli.registry, "resolve", Mock(return_value=plan))
    monkeypatch.setattr(cli.fetch, "ensure", Mock())
    seen = []
    monkeypatch.setattr(cli, "_chat", lambda resolved, *_args: seen.append(resolved) or 0)

    assert cli.main(["dsv4-q4ktail-2", "--quant", "IQ2_XXS", "--no-spec"]) == 0
    assert len(seen) == 1
    assert seen[0].speculative is False
    assert "speculative_config" not in engine_kwargs(seen[0])


def test_every_profile_source_names_a_blessed_dspark_download():
    sources = registry._registry()["sources"]
    for profile_id in registry.profile_ids():
        speculator = sources[registry.describe(profile_id)["source"]]["speculator"]
        assert speculator.get("base_url", "https://huggingface.co/").startswith(
            "https://huggingface.co/"
        )


def test_every_source_declares_its_live_smoke_modalities():
    sources = registry._registry()["sources"]
    assert sources["glm52-vision"]["modalities"] == ["text", "image"]
    assert sources["kimi-k3"]["modalities"] == ["text", "image"]
    assert sources["dsv4-flash"]["modalities"] == ["text"]


def test_live_smoke_matrix_discovers_every_compatible_mi300x_profile():
    machine = Machine("mi300x", "AMD Instinct MI300X", 8)
    expected = {
        profile_id
        for profile_id in registry.profile_ids()
        if "mi300x" in registry.describe(profile_id)["platforms"]
    }

    compatible = compatible_profile_ids(machine)

    assert set(compatible) == expected
    assert {"k3-xxs-6", "k3-xxs-8"}.issubset(compatible)
    assert {"dsv4-xxs-1", "dsv4-q4ktail-2", "dsv4-mxfp4-4", "dsv4-q4k-8"}.issubset(compatible)


def test_live_smoke_matrix_requires_dspark_and_turboquant_for_every_profile():
    machine = Machine("mi300x", "AMD Instinct MI300X", 8)
    for profile_id in compatible_profile_ids(machine):
        plan = resolve(profile_id, "mi300x", 8, None)
        speculative = validate_acceleration(plan)
        assert speculative["method"] == "dspark"
        assert speculative["attention_backend"] == "TURBOQUANT"
        assert speculative["kv_cache_dtype"] == "turboquant_k8v4"


def test_registry_rejects_duplicate_json_keys():
    with pytest.raises(ValueError, match="duplicate key.*speculator"):
        registry._unique_object([("speculator", 1), ("speculator", 2)])


def test_deepseek_profiles_use_only_the_matching_0731_dspark_drafter():
    sources = registry._registry()["sources"]
    expected_repo = "alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF"
    expected_revision = "799216bd6a33457ae41a26968773d7cb47e157b6"
    expected_file = "DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf"
    for profile_id, platform in (
        ("dsv4-xxs-1", "mi300x"),
        ("dsv4-q4ktail-2", "mi300x"),
        ("dsv4-mxfp4-4", "mi300x"),
        ("dsv4-q4k-8", "mi300x"),
        ("dsv4-q4ktail-2", "a100"),
        ("dsv4-q4ktail-4", "a100"),
    ):
        entry = registry.describe(profile_id)
        assert entry["source"] == "dsv4-flash"
        speculator = sources[entry["source"]]["speculator"]
        assert speculator["repo"] == expected_repo
        assert speculator["revision"] == expected_revision
        assert speculator["file"]["path"] == expected_file

        plan = resolve(profile_id, platform, 8, None)
        config = engine_kwargs(plan)["speculative_config"]
        assert config["num_speculative_tokens"] == 5
        assert config["quantization"] == "gguf"
        assert config["attention_backend"] == "TURBOQUANT"
        assert config["kv_cache_dtype"] == "turboquant_k8v4"
        assert config["model"].endswith(f"/{expected_file}")


def test_deepseek_drafter_is_fetched_once_with_the_plan(tmp_path, monkeypatch):
    payload = b"GGUF"
    filename = "draft.gguf"
    plan = resolve("dsv4-q4ktail-2", "a100", 2, None)
    source = {
        **plan.source,
        "speculator": {
            "repo": "publisher/drafter",
            "base_url": "https://huggingface.co/publisher/drafter/resolve/rev",
            "local_dir": "drafter",
            "file": {
                "path": filename,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            "engine": {
                "method": "dspark",
                "num_speculative_tokens": 5,
            },
        },
    }
    plan = replace(
        plan,
        source=source,
        quant=replace(plan.quant, files=[], assembly=None),
    )
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    calls = []

    def fake_download(entry, destination):
        calls.append((entry, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(fetch, "_download", fake_download)

    fetch.ensure(plan, assume_yes=True)
    fetch.ensure(plan, assume_yes=True)

    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "drafter" / filename


def test_kimi_drafter_is_scoped_only_to_kimi_profiles():
    sources = registry._registry()["sources"]
    for profile_id in ("k3-xxs-6", "k3-xxs-8"):
        entry = registry.describe(profile_id)
        assert entry["source"] == "kimi-k3"
        speculator = sources[entry["source"]]["speculator"]
        assert speculator["repo"] == "Lucebox/Kimi-K3-DSpark-Q8_0-GGUF"


def test_quant_too_large_for_the_profile_names_a_bigger_one():
    with pytest.raises(ProfileError, match="try glm52-q2k-4"):
        resolve("glm52-q2k-2", "mi300x", 8, "Q4_K")


def test_profile_rejected_on_an_unsupported_platform():
    with pytest.raises(ProfileError, match="not supported"):
        resolve("k3-xxs-6", "a100", 8, None)


def test_profile_rejected_when_the_machine_is_too_small():
    with pytest.raises(ProfileError, match="needs 8 GPUs"):
        resolve("glm52-q2k-8", "mi300x", 6, None)


def test_unknown_names_are_rejected_with_the_legal_set():
    with pytest.raises(ProfileError, match="glm52-q2k-2"):
        registry.describe("nope")
    with pytest.raises(ProfileError, match="IQ2_XXS"):
        resolve("glm52-q2k-2", "mi300x", 2, "Q9_Z")


def test_platform_override_replaces_the_mi300x_kv_budget():
    """A100 has 80 GB cards; the MI300X byte budget would not fit."""
    amd = resolve("glm52-q2k-4", "mi300x", 4, "Q2_K")
    nvidia = resolve("glm52-q2k-4", "a100", 4, "Q2_K")
    assert "kv_cache_memory_bytes" in amd.engine
    assert "kv_cache_memory_bytes" not in nvidia.engine
    assert nvidia.engine["gpu_memory_utilization"] == 0.92
    assert nvidia.env == {}, "AITER is a ROCm switch"


def test_deepseek_v4_a100_tp2_and_tp4_profiles_are_legal():
    tp2 = resolve("dsv4-q4ktail-2", "a100", 2, "Q4K-tail")
    assert tp2.engine["tensor_parallel_size"] == 2
    assert tp2.engine["block_size"] == 256
    assert tp2.engine["kv_cache_dtype"] == "fp8"
    assert tp2.env == {
        "VLLM_DSV4_ALIGNED_Q8": "1",
        "VLLM_DSV4_MHC_SCHEDULE": "async",
        # Seed mitigation 2026-08-12: the multi-stream attention overlap is
        # the only component whose removal silences the rare NaN seed, and
        # it measured faster off. See the profile note and perf notebook.
        "VLLM_DSV4_AUX_STREAMS": "0",
    }

    tp4 = resolve("dsv4-q4ktail-4", "a100", 4, "MXFP4")
    assert tp4.engine["tensor_parallel_size"] == 4
    assert tp4.engine["block_size"] == 256
    assert tp4.engine["kv_cache_dtype"] == "fp8"
    assert tp4.env == {
        "VLLM_DSV4_ALIGNED_Q8": "1",
        "VLLM_DSV4_MHC_SCHEDULE": "async",
        "VLLM_DSV4_AUX_STREAMS": "0",
        # qwarp8 IQ2 W1 decode variant: +2.7% c1 step rate on the TP4 shard
        # (2026-08-15 dial sweep, exact-verified both runs).
        "VLLM_DSV4_W1_QWARP8": "1",
    }


def test_kimi_needs_the_native_kv_dtype():
    """This fork defaults the cache to fp8, which K3 cannot use."""
    for profile_id in ("k3-xxs-6", "k3-xxs-8"):
        assert resolve(profile_id, "mi300x", 8, None).engine["kv_cache_dtype"] == "auto"


def test_registry_contains_only_the_supported_model_artifacts():
    data = registry._registry()
    assert set(data["sources"]) == {
        "glm52-vision",
        "kimi-k3",
        "dsv4-flash",
        "muse-glimmer",
        "qwen38-27b-nvfp4",
    }
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
    assert deepseek["speculator"]["file"]["path"] == (
        "DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf"
    )
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
    plans = [
        resolve("k3-xxs-6", "mi300x", 8, None),
        resolve("k3-xxs-8", "mi300x", 8, None),
    ]
    for plan in plans:
        speculative = engine_kwargs(plan)["speculative_config"]

        assert speculative["model"].endswith(
            "/Kimi-K3-DSpark-Q8_0-GGUF/Kimi-K3-DSpark-Q8_0.gguf"
        )
        assert speculative["method"] == "dspark"
        assert speculative["num_speculative_tokens"] == 7
        assert speculative["quantization"] == "gguf"
        assert speculative["attention_backend"] == "TURBOQUANT"
        assert speculative["kv_cache_dtype"] == "turboquant_k8v4"
        assert speculative["disable_draft_cudagraphs"] is True
        assert "draft_tensor_parallel_size" not in speculative

    assert (
        engine_kwargs(plans[0])["speculative_config"]["replicate_draft_backbone"]
        is True
    )
    assert (
        "replicate_draft_backbone" not in engine_kwargs(plans[1])["speculative_config"]
    )

    [draft] = [entry for entry in files_for(plans[1]) if entry["role"] == "speculator"]
    assert draft["bytes"] == 2390153888
    assert draft["url"] == (
        "https://huggingface.co/Lucebox/Kimi-K3-DSpark-Q8_0-GGUF/"
        "resolve/main/Kimi-K3-DSpark-Q8_0.gguf"
    )


GB = 1 << 30


def test_nvfp4_directory_entry_resolves_to_the_model_dir():
    """The NVFP4 quant is an HF-format directory checkpoint: --model must be
    the folder, not files[0], and unknown entry modes must refuse loudly."""
    plan = resolve("qwen38-nvfp4-1", "metal", 1, "NVFP4", 128 * GB)
    assert plan.entry_file == plan.model_dir
    with pytest.raises(ProfileError, match="entry mode"):
        registry._validated_entry("dir")


def test_metal_gates_on_memory_not_on_gpu_count():
    """One Mac is always one GPU, so the card count must not decide anything."""
    assert registry.platform_gate("metal") == "memory"
    plan = resolve("dsv4-xxs-1", "metal", 1, "IQ2_XXS", 128 * GB)
    assert plan.engine["tensor_parallel_size"] == 1
    # The same single "GPU" cannot hold the 145 GiB build.
    with pytest.raises(ProfileError, match="unified memory"):
        resolve("dsv4-xxs-1", "metal", 1, "MXFP4", 128 * GB)


def test_metal_suggests_a_smaller_quant_not_a_bigger_machine():
    """More RAM is not a choice the user can make at the prompt; a quant is."""
    with pytest.raises(ProfileError, match="try Q4K-tail"):
        resolve("dsv4-xxs-1", "metal", 1, "MXFP4", 128 * GB)


def test_a_laptop_cannot_hold_glm_at_any_quant():
    with pytest.raises(ProfileError, match="no quant of this model fits"):
        resolve("glm52-xxs-1", "metal", 1, "IQ2_XXS", 128 * GB)


def test_kimi_is_absent_from_metal_because_no_mac_is_large_enough():
    """K3 is 800 GiB and the largest Mac is 512 GB; claiming it would be a lie."""
    for profile_id in ("k3-xxs-6", "k3-xxs-8"):
        assert "metal" not in registry.describe(profile_id)["platforms"]
    with pytest.raises(ProfileError, match="not supported"):
        resolve("k3-xxs-6", "metal", 1, None, 512 * GB)


def test_deepseek_metal_is_runnable_while_glm_stays_gated():
    assert registry.platform_blocked("metal") is None
    assert registry.profile_blocked("dsv4-xxs-1", "metal") is None
    assert registry.profile_blocked("glm52-xxs-1", "metal")
    assert registry.platform_blocked("mi300x") is None
    assert registry.platform_blocked("a100") is None


def test_deepseek_metal_uses_measured_dspark_turboquant_settings():
    plan = resolve("dsv4-xxs-1", "metal", 1, "IQ2_XXS", 128 * GB)
    # 256K metal resize (2026-08-11 Metal-side commit).
    assert plan.engine["max_model_len"] == 262144
    assert plan.engine["max_num_seqs"] == 32
    assert plan.engine["kv_cache_memory_bytes"] == 16 * GB
    assert plan.engine["kv_cache_dtype"] == "fp8_ds_mla"
    speculative = engine_kwargs(plan)["speculative_config"]
    assert speculative["method"] == "dspark"
    assert speculative["attention_backend"] == "TURBOQUANT"
    assert speculative["kv_cache_dtype"] == "turboquant_k8v4"
    assert speculative["disable_draft_cudagraphs"] is True


def test_deepseek_dspark_fetches_the_pinned_0731_drafter():
    plan = resolve("dsv4-mxfp4-4", "mi300x", 8, None)
    speculative = engine_kwargs(plan)["speculative_config"]
    assert speculative["model"].endswith(
        "/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF/"
        "DeepSeek-V4-Flash-0731-DSpark-Drafter-Q2_K-Q8_0-dflash.gguf"
    )
    [drafter] = [entry for entry in files_for(plan) if entry["role"] == "speculator"]
    assert drafter["bytes"] == 6971243008
    assert drafter["sha256"] == (
        "3e2be643b7881ac61e49c9907a963bdbbfcffe89c4d15c5f0e99e827e0305914"
    )
    assert drafter["local_dir"] == "DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF"


def test_deepseek_metal_verifier_is_checksum_pinned():
    plan = resolve("dsv4-xxs-1", "metal", 1, "IQ2_XXS", 128 * GB)
    [target] = [entry for entry in files_for(plan) if entry["role"] == "model"]
    assert target["bytes"] == 86720111488
    assert target["sha256"] == (
        "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0"
    )


def test_deepseek_profiles_cover_all_supported_tensor_parallel_sizes():
    # MI300X/Metal lineage keeps the numeric names; A100 serves the
    # user-confirmed hybrid/mxfp4 family (see perf/optimization_status.md).
    # (platform, gpus, engine tensor_parallel_size, default quant);
    # dsv4-q4ktail-8 is the TP4 x DP2 throughput tier, so gpus != tp there.
    # One namespace: <model>-<quant>-<gpus>; a profile resolves on every
    # platform it lists and nowhere else. dsv4-q4ktail-8 is the TP4 x DP2
    # throughput tier, so gpus != tensor_parallel_size there.
    expected = {
        "dsv4-xxs-1": [("mi300x", 1, 1, "IQ2_XXS")],
        "dsv4-q4ktail-2": [
            ("mi300x", 2, 2, "Q4K-tail"),
            ("a100", 2, 2, "Q4K-tail"),
        ],
        "dsv4-q4ktail-4": [("a100", 4, 4, "Q4K-tail")],
        "dsv4-q4ktail-8": [("a100", 8, 4, "Q4K-tail")],
        "dsv4-mxfp4-4": [
            ("mi300x", 4, 4, "MXFP4"),
            ("a100", 4, 4, "MXFP4"),
        ],
        "dsv4-mxfp4-8": [("a100", 8, 8, "MXFP4")],
        "dsv4-q4k-8": [("mi300x", 8, 8, "Q4_K")],
    }

    assert {
        profile_id
        for profile_id in registry.profile_ids()
        if profile_id.startswith("dsv4-")
    } == set(expected)
    for profile_id, cases in expected.items():
        for platform, gpus, tp_size, default_quant in cases:
            plan = resolve(profile_id, platform, 8, None)
            assert plan.gpus == gpus
            assert plan.engine["tensor_parallel_size"] == tp_size
            assert plan.quant.name == default_quant


def test_serve_argv_renders_flags_the_api_server_accepts():
    argv = serve_argv(resolve("glm52-q2k-2", "mi300x", 2, None), "127.0.0.1", 8000)
    assert "--enable-prefix-caching" in argv
    assert "--tensor-parallel-size" in argv
    # Booleans are flags, never "--flag True".
    assert "True" not in argv
    assert (
        argv[argv.index("--attention-config") + 1] == '{"sparse_mla_force_mqa": true}'
    )


def test_engine_kwargs_drop_server_only_settings():
    kwargs = engine_kwargs(resolve("k3-xxs-6", "mi300x", 6, None))
    assert "served_model_name" not in kwargs
    assert kwargs["tensor_parallel_size"] == 6
    assert kwargs["model"].endswith(".gguf")


def _all_plans():
    for profile_id in registry.profile_ids():
        for platform in registry.describe(profile_id)["platforms"]:
            yield resolve(
                profile_id, platform, 8, None, memory_bytes=512 * 1024**3
            )


def test_every_profile_serves_thinking_and_tool_calling_by_default():
    for plan in _all_plans():
        engine = plan.engine
        assert engine["enable_auto_tool_choice"] is True, plan.profile_id
        kwargs = engine["default_chat_template_kwargs"]
        assert kwargs["thinking"] is True, plan.profile_id
        assert kwargs["enable_thinking"] is True, plan.profile_id
        assert engine["reasoning_parser"], plan.profile_id
        # Muse-Glimmer has no tool parser in this fork; auto tool choice
        # stays enabled globally and no-ops without a registered parser.
        if plan.source_key != "muse-glimmer":
            assert engine["tool_call_parser"], plan.profile_id
        # No profile forces the chat client back out of thinking mode.
        assert plan.chat_template_kwargs.get("thinking") is not False, (
            plan.profile_id
        )


def test_thinking_and_tool_defaults_are_serve_only():
    plan = resolve("dsv4-q4ktail-4", "a100", 8, None)
    argv = serve_argv(plan, "127.0.0.1", 8000)
    assert "--enable-auto-tool-choice" in argv
    assert (
        argv[argv.index("--default-chat-template-kwargs") + 1]
        == '{"thinking": true, "enable_thinking": true}'
    )
    kwargs = engine_kwargs(plan)
    assert "default_chat_template_kwargs" not in kwargs
    assert "enable_auto_tool_choice" not in kwargs


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
    """Graph mode is per-platform evidence: DSpark/TurboQuant DSV4 runs eager
    on MI300X/Metal, while the A100 profiles qualified PIECEWISE capture
    (128K lifecycle, perf/optimization_status.md)."""
    for profile_id in registry.profile_ids():
        for platform in registry.describe(profile_id)["platforms"]:
            engine = resolve(
                profile_id,
                platform,
                registry.describe(profile_id)["gpus"],
                None,
                _big_enough(profile_id, platform),
            ).engine
            cudagraph_mode = engine.get("compilation_config", {}).get("cudagraph_mode")
            if profile_id.startswith("dsv4-") and platform == "a100":
                assert cudagraph_mode in ("PIECEWISE", "FULL_DECODE_ONLY"), profile_id
            elif (
                profile_id.startswith("dsv4-")
                or profile_id in ("k3-xxs-6", "k3-xxs-8")
                or platform == "metal"
            ):
                assert cudagraph_mode == "NONE", (profile_id, platform)
            else:
                assert cudagraph_mode not in (None, "NONE"), profile_id


def test_a100_deepseek_profiles_default_to_the_hybrid_quant():
    """Datacenter GPUs serve Q4K-tail; IQ2_XXS is the MacBook-footprint quant."""
    for profile_id in ("dsv4-q4ktail-2", "dsv4-q4ktail-4"):
        plan = resolve(profile_id, "a100", 8, None)
        assert plan.quant.name == "Q4K-tail", profile_id
    assert resolve("dsv4-mxfp4-4", "a100", 8, None).quant.name == "MXFP4"
    eight = resolve("dsv4-mxfp4-8", "a100", 8, None)
    assert eight.quant.name == "MXFP4"
    assert eight.engine["tensor_parallel_size"] == 8


def test_dsv4_hybrid_8_is_the_tp4_dp2_throughput_tier():
    """The box-record layout (858-926 tok/s hot c8): two TP4 replicas keep
    each engine's verify batches on CUDA graphs."""
    plan = resolve("dsv4-q4ktail-8", "a100", 8, None)
    assert plan.quant.name == "Q4K-tail"
    assert plan.engine["tensor_parallel_size"] == 4
    assert plan.engine["data_parallel_size"] == 2
    # FULL capture-64 restored 2026-08-11 with the bt_per_token persistence
    # fix (0/8 storm campaign); see the profile note and perf notebook.
    assert plan.engine["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    capture = plan.engine["compilation_config"]["max_cudagraph_capture_size"]
    assert capture == 64


def test_dsv4_mxfp4_8_is_tp8_with_wide_capture():
    """Eight-GPU MXFP4 serves TP8; graph capture 64 keeps the 48-token c8
    verify batches on CUDA graphs. FULL restored 2026-08-11 with the
    bt_per_token persistence fix (0/8 storm campaign; see notebook)."""
    mxfp4 = resolve("dsv4-mxfp4-8", "a100", 8, None)
    assert mxfp4.quant.name == "MXFP4"
    assert mxfp4.engine["tensor_parallel_size"] == 8
    assert "data_parallel_size" not in mxfp4.engine
    assert mxfp4.engine["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
    capture = mxfp4.engine["compilation_config"]["max_cudagraph_capture_size"]
    assert capture == 64


def test_dsv4_2_a100_kv_budget_is_per_quant():
    """A KV byte budget measured for one artifact must not follow another."""
    hybrid = resolve("dsv4-q4ktail-2", "a100", 2, None)
    xxs = resolve("dsv4-q4ktail-2", "a100", 2, "IQ2_XXS")
    assert hybrid.engine["kv_cache_memory_bytes"] == 13958643712
    assert xxs.engine["kv_cache_memory_bytes"] == 20401094656


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


def test_dsv4_flash_artifacts_are_0731_and_checksum_pinned():
    """The repo hosts same-size non-0731 twins of every 0731 build, so byte
    counts cannot distinguish them; only the -0731 path plus a sha256 pin
    gates 'we only support 0731'. Q4_K's pin is pending a hash on the MI300X
    box that holds the file."""
    quants = registry._registry()["sources"]["dsv4-flash"]["quants"]
    pending_pin = {"Q4_K"}
    for name, info in quants.items():
        for entry in info["files"]:
            assert entry["path"].endswith("-0731.gguf"), (name, entry["path"])
            if name not in pending_pin:
                sha = entry.get("sha256")
                assert sha and len(sha) == 64, f"{name} missing sha256 pin"
