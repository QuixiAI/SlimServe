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
    default_quant = registry.variant(profile_id, platform)["default_quant"]
    return source["quants"][default_quant]["min_memory_bytes"][platform]


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
            # The variant may override the source drafter (qwen38-nvfp4-1:
            # MTP on MI300X, DFlash2 on Metal).
            registered = plan.speculator["engine"]["method"]
            assert config["method"] == registered
            if registered == "dspark":
                assert config["attention_backend"] == "TURBOQUANT"
                assert config["kv_cache_dtype"] == "turboquant_k8v4"


def test_no_spec_cli_flag_disables_the_resolved_speculator(monkeypatch):
    plan = resolve("dsv4-q4ktail-2", "a100", 2, "IQ2_XXS")
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        Mock(
            return_value=Mock(
                known=True,
                platform="a100",
                count=2,
                memory_bytes=0,
                device_name="A100",
            )
        ),
    )
    monkeypatch.setattr(cli.registry, "resolve", Mock(return_value=plan))
    monkeypatch.setattr(cli.fetch, "ensure", Mock())
    seen = []

    def _capture_chat(resolved, *_args):
        seen.append(resolved)
        return 0

    monkeypatch.setattr(cli, "_chat", _capture_chat)

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
    assert sources["muse-glimmer"]["modalities"] == ["text", "image"]
    assert sources["qwen38-27b"]["modalities"] == ["text", "image"]
    assert sources["qwen38-27b-nvfp4"]["modalities"] == ["text", "image"]


def test_qwen38_uses_measured_metal_speculation_settings():
    plan = resolve("qwen38-q2kxl-1", "metal", 1, None, 2**37)
    speculative = validate_acceleration(plan)

    assert speculative["method"] == "dflash"
    assert speculative["num_speculative_tokens"] == 3
    assert speculative["quantization"] == "gguf"
    assert plan.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_qwen38_uses_measured_mi300x_speculation_settings():
    plan = resolve("qwen38-q2kxl-1", "mi300x", 1, None, 2**38)
    speculative = validate_acceleration(plan)

    assert speculative["method"] == "dflash"
    assert speculative["num_speculative_tokens"] == 3
    assert speculative["quantization"] == "gguf"
    assert plan.env["VLLM_ROCM_USE_AITER"] == "1"
    assert plan.env["VLLM_USE_V2_MODEL_RUNNER"] == "1"


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
    assert {"dsv4-xxs-1", "dsv4-q4ktail-2", "dsv4-mxfp4-4", "dsv4-q4k-8"}.issubset(
        compatible
    )


def test_live_smoke_matrix_requires_dspark_and_turboquant_for_every_profile():
    machine = Machine("mi300x", "AMD Instinct MI300X", 8)
    for profile_id in compatible_profile_ids(machine):
        plan = resolve(profile_id, "mi300x", 8, None)
        speculative = validate_acceleration(plan)
        if profile_id == "qwen38-nvfp4-1":
            # The checkpoint ships its own MTP head; there is no separate
            # DSpark artifact or TurboQuant draft cache to require.
            assert speculative["method"] == "qwen3_5_mtp"
            continue
        if profile_id == "qwen38-q2kxl-1":
            # The GGUF artifact's blessed drafter is the published DFlash 2
            # block model, which shares the target's KV layout and so has no
            # TurboQuant draft cache of its own.
            assert speculative["method"] == "dflash"
            continue
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
    assert nvidia.engine["gpu_memory_utilization"] == 0.95
    assert nvidia.env == {}, "AITER is a ROCm switch"


def test_deepseek_v4_a100_tp2_and_tp4_profiles_are_legal():
    tp2 = resolve("dsv4-q4ktail-2", "a100", 2, "Q4K-tail")
    assert tp2.engine["tensor_parallel_size"] == 2
    assert tp2.engine["block_size"] == 256
    # bf16 main KV (serving policy): the Ampere bf16 sparse-MLA page path
    # is the NFP8=0 instantiation; see the profile note and
    # csrc/quixicore/dsv4_bf16_kv_design.md.
    assert tp2.engine["kv_cache_dtype"] == "auto"
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
    assert tp4.engine["kv_cache_dtype"] == "auto"  # same policy as tp2 above
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
        "qwen38-27b",
        "qwen38-27b-nvfp4",
        "qwen38-flash-next-fp8",
    }
    glm = data["sources"]["glm52-vision"]
    kimi = data["sources"]["kimi-k3"]
    deepseek = data["sources"]["dsv4-flash"]
    muse = data["sources"]["muse-glimmer"]
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
        (
            "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp-"
            "Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf"
        ),
        (
            "DeepSeek-V4-Flash-MXFP4Experts-F16HC-F16Compressor-F16Indexer-Q8Attn-"
            "Q8Shared-Q8Out-chat-v2-mxfp4-0731.gguf"
        ),
        (
            "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-"
            "Q8Shared-Q8Out-chat-v2-imatrix-0731.gguf"
        ),
    }

    assert [entry["path"] for entry in muse["shared"]] == [
        "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
    ]
    assert {
        name: (quant["bytes"], quant["files"][0]["path"])
        for name, quant in muse["quants"].items()
    } == {
        "kquant-dynamic": (
            19653960832,
            "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf",
        ),
        "kquant-17gb": (
            16756683904,
            "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        ),
    }
    assert muse["speculator"]["file"] == {
        "path": "dflash-Muse-Glimmer-30B-Q4_K_M.gguf",
        "bytes": 1631208128,
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
    """The NVFP4 quant is an HF-format directory checkpoint (source format
    safetensors): --model must be the folder, not files[0]."""
    plan = resolve("qwen38-nvfp4-1", "metal", 1, "NVFP4", 128 * GB)
    assert plan.entry_file == plan.model_dir


def test_qwen38_nvfp4_platforms_diverge_on_the_measured_drafter():
    """MI300X reuses the checkpoint's own MTP head; Metal serves the DFlash2
    drafter that measured +23% at c1 (variant-level speculator override)."""
    metal = resolve("qwen38-nvfp4-1", "metal", 1, "NVFP4", 128 * GB)
    assert metal.speculator["engine"]["method"] == "dflash"
    assert metal.speculator["repo"] == "z-lab/Qwen3.8-27B-DFlash2"
    mi300x = resolve("qwen38-nvfp4-1", "mi300x", 1, None)
    assert mi300x.speculator["engine"]["method"] == "qwen3_5_mtp"


def test_tq_profile_pins_both_drafter_turboquant_fields():
    """The -tq drafter must pin attention_backend AND kv_cache_dtype
    together: unset, the drafter inherits the engine-global turboquant_k8v4
    dtype but keeps metal_attn, whose 5-dim cache shape cannot view the
    TQ-sized page (boot reshape failure documented in the profile notes)."""
    from slimserve.engine import _speculative_config

    plan = resolve("qwen38-nvfp4-1-tq", "metal", 1, "NVFP4", 128 * GB)
    cfg = _speculative_config(plan)
    assert cfg["attention_backend"] == "TURBOQUANT"
    assert cfg["kv_cache_dtype"] == "turboquant_k8v4"


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
            yield resolve(profile_id, platform, 8, None, memory_bytes=512 * 1024**3)


def test_every_profile_serves_thinking_and_tool_calling_by_default():
    for plan in _all_plans():
        engine = plan.engine
        assert engine["enable_auto_tool_choice"] is True, plan.profile_id
        kwargs = engine["default_chat_template_kwargs"]
        assert kwargs["thinking"] is True, plan.profile_id
        assert kwargs["enable_thinking"] is True, plan.profile_id
        assert engine["reasoning_parser"], plan.profile_id
        assert engine["tool_call_parser"], plan.profile_id
        # No profile forces the chat client back out of thinking mode.
        assert plan.chat_template_kwargs.get("thinking") is not False, plan.profile_id


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
            elif profile_id == "qwen38-nvfp4-1":
                # Qualified 2026-08-18: FULL_DECODE_ONLY capture 64 measured
                # 1.4x at c1 on the hybrid GDN + MTP decode.
                assert cudagraph_mode == "FULL_DECODE_ONLY", profile_id
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


def test_a_profile_is_one_config_per_platform():
    """A profile is model x quant x platform x config.

    The id a user types carries no platform because the CLI detects it, so
    every id stores one record per platform and each record states its own
    platform. Nothing may span platforms: that is what the retired
    `platform_overrides` block used to paper over, and it let a config tuned
    for one platform silently stand in for another.
    """
    for profile_id, entry in registry._registry()["profiles"].items():
        variants = entry.get("variants")
        assert variants, f"{profile_id} has no per-platform records"
        assert "platforms" not in entry, (
            f"{profile_id} carries a platforms list; platform belongs to the "
            "record, not the profile"
        )
        assert "platform_overrides" not in entry, (
            f"{profile_id} uses platform_overrides; give each platform its "
            "own record instead"
        )
        for platform, record in variants.items():
            assert record.get("platform") == platform, (
                f"{profile_id}/{platform} does not state its own platform"
            )
            assert "engine" in record and record["engine"], (
                f"{profile_id}/{platform} has no engine settings"
            )
            assert "default_quant" in record, (
                f"{profile_id}/{platform} has no default quant"
            )


def test_no_profile_carries_another_platforms_environment():
    """A ROCm switch on an a100 or Metal record is a config that leaked."""
    marker = {
        "mi300x": ("ROCM", "AITER", "HIP_"),
        "a100": ("CUDA_",),
    }
    for profile_id, entry in registry._registry()["profiles"].items():
        for platform, record in entry["variants"].items():
            for key in record.get("env") or {}:
                for owner, tokens in marker.items():
                    if owner == platform:
                        continue
                    assert not any(tok in key for tok in tokens), (
                        f"{profile_id}/{platform} sets {key}, which belongs to {owner}"
                    )


# vllm/config/scheduler.py::SchedulerConfig.DEFAULT_MAX_NUM_SEQS. Mirrored
# rather than imported so profile-registry tests stay free of a vllm import;
# test_mirrored_vllm_defaults_have_not_drifted below pins it to the real value.
DEFAULT_MAX_NUM_SEQS = 128


# (profile, platform) -> (validated concurrency, why the max_num_seqs rule is
# not applied). These records pin a capture ceiling that was MEASURED against a
# specific concurrency band rather than against max_num_seqs, and raising it is
# coupled to the KV budget on hardware not present here.
#
# dsv4 A100 tiers: capture 64 was the fix for this very bug (2026-08-10, "TP8 c8
# Cliff Root Cause: Graph Capture Width") -- the c8 verify batch is 8 reqs x 6
# spec tokens = 48 rows, and the list topped out at 32, so every decode step ran
# eager. 64 makes the derived list [1,2,4,8,16,24,32,40,48,56,64], which
# contains 48. These records leave max_num_seqs unpinned, so it inherits 128 and
# batches past ~10 concurrent requests still fall off the graphs. The notebook's
# own follow-up ("extend cudagraph_capture_sizes to include 48 and re-derive the
# KV budget") is the open work; larger graphs measurably shrink the KV pool
# there, so the fix needs an A100, not an edit. Until then this asserts the band
# that WAS measured.
_CAPTURE_BAND_EXEMPT = {
    ("dsv4-q4ktail-4", "a100"): (8, "capture measured against the c8 verify width"),
    ("dsv4-q4ktail-8", "a100"): (8, "capture measured against the c8 verify width"),
    ("dsv4-mxfp4-4", "a100"): (8, "capture measured against the c8 verify width"),
    ("dsv4-mxfp4-8", "a100"): (8, "capture measured against the c8 verify width"),
}


def test_full_decode_graphs_cover_the_largest_speculative_batch():
    """capture >= (k+1) * max_num_seqs, or the biggest batches run eager.

    Measured on 8x3090: c32 at 425 tok/s with a 64-token capture ceiling vs
    880 with the ceiling covering seqs x (1+k) -- and the failure is silent.
    A note is not a guard; this is.

    Both sides are resolved to their EFFECTIVE values. An unset capture
    inherits min(max_num_seqs * 2, 512), a ceiling derived with no knowledge
    of speculation: it budgets one row per sequence with 2x headroom, while a
    speculative step submits k+1 rows per sequence. At k>=2 the default is
    therefore always short, so treating "unset" as "nothing to check" skipped
    precisely the records where nobody had made the decision at all.
    """
    raw = registry._registry()
    for profile_id, profile in raw["profiles"].items():
        if not profile.get("speculative"):
            continue
        source = raw["sources"][profile["source"]]
        base_k = (
            (source.get("speculator") or {})
            .get("engine", {})
            .get("num_speculative_tokens", 0)
        )
        for platform, record in profile["variants"].items():
            engine = record.get("engine", {})
            compilation = engine.get("compilation_config") or {}
            mode = str(compilation.get("cudagraph_mode", ""))
            if "FULL" not in mode:
                continue
            overrides = record.get("speculative_overrides") or {}
            k = overrides.get("num_speculative_tokens", base_k)
            schedule = overrides.get("num_speculative_tokens_per_batch_size")
            if schedule:
                k = max(k, max(entry[2] for entry in schedule))
            if not k:
                continue
            # Resolve what the engine will ACTUALLY use, not just what the
            # record spells out. Skipping the unset cases would skip exactly
            # the silent ones: an omitted capture inherits a default derived
            # from max_num_seqs with no knowledge of speculation, which is
            # where this bug hides rather than where it is absent.
            max_num_seqs = engine.get("max_num_seqs") or DEFAULT_MAX_NUM_SEQS
            capture = compilation.get("max_cudagraph_capture_size")
            if capture is None:
                # vllm/config/vllm.py::_set_cudagraph_sizes.
                capture = min(max_num_seqs * 2, 512)
                source_note = (
                    f"the inherited default min({max_num_seqs} x 2, 512) = {capture}"
                )
            else:
                source_note = f"the pinned {capture}"
            needed = (k + 1) * max_num_seqs
            if (profile_id, platform) in _CAPTURE_BAND_EXEMPT:
                band, why = _CAPTURE_BAND_EXEMPT[(profile_id, platform)]
                assert capture >= (k + 1) * band, (
                    f"{profile_id}/{platform} is exempt from the full "
                    f"max_num_seqs rule ({why}), but its capture {capture} no "
                    f"longer covers even the validated c{band} band "
                    f"({(k + 1) * band} rows)"
                )
                continue
            assert capture >= needed, (
                f"{profile_id}/{platform}: max_cudagraph_capture_size "
                f"{source_note} < ({k}+1) x max_num_seqs {max_num_seqs} = "
                f"{needed}; the largest decode batches would silently run "
                "eager"
            )


def test_host_offload_profiles_declare_a_host_ram_gate():
    """PLE-host pins ~48 GiB of system RAM per rank; the GPU gate can't see
    that. Any variant that turns on host offload must carry a
    min_host_ram_bytes entry for its platform, sized at least to the pinned
    tables across the tensor-parallel ranks."""
    raw = registry._registry()
    # One shared /dev/shm segment across TP ranks (see
    # Qwen4ExpHostNGramEmbedding._shared_pinned_table): the floor is one
    # table, not one per rank.
    ple_table_bytes = 47_700_000_000  # 47.7 GiB, model-defined
    for profile_id, profile in raw["profiles"].items():
        source = raw["sources"][profile["source"]]
        for platform, record in profile["variants"].items():
            env = record.get("env") or {}
            if env.get("VLLM_QWEN4_EXP_PLE_HOST") != "1":
                continue
            floor = ple_table_bytes
            gated = [
                quant
                for quant in source["quants"].values()
                if (quant.get("min_host_ram_bytes") or {}).get(platform, 0) >= floor
            ]
            assert gated, (
                f"{profile_id}/{platform} enables PLE host offload "
                f"(~{floor / 2**30:.0f} GiB pinned, shared across ranks) "
                "but no quant declares a min_host_ram_bytes gate covering it"
            )


def test_mirrored_vllm_defaults_have_not_drifted():
    """The capture guard resolves unset fields against vLLM's own defaults.

    Those defaults are mirrored as literals so the registry tests do not import
    vllm; this is the one test that pays the import, so a vLLM-side change to
    either default fails loudly here instead of silently weakening the guard.
    """
    from vllm.config.scheduler import SchedulerConfig

    assert SchedulerConfig.DEFAULT_MAX_NUM_SEQS == DEFAULT_MAX_NUM_SEQS

    # The capture default itself: vllm/config/vllm.py::_set_cudagraph_sizes
    # computes min(max_num_seqs * 2, 512).
    import inspect

    from vllm.config import VllmConfig

    source = inspect.getsource(VllmConfig._set_cudagraph_sizes)
    assert "min(max_num_seqs * 2, 512)" in source, (
        "vLLM's default cudagraph capture ceiling changed; update the "
        "min(max_num_seqs * 2, 512) model in "
        "test_full_decode_graphs_cover_the_largest_speculative_batch"
    )


def test_every_profile_states_prefix_caching_explicitly():
    """Prefix caching is always ON, and always stated.

    vLLM defaults prefix caching OFF for hybrid (mamba/GDN) models, so a
    profile that omits enable_prefix_caching silently pays full-history
    re-prefill on every chat turn -- exactly how qwen38fn-fp8-8 shipped
    with a 0.0% hit rate. Policy since 2026-08-28: every record states the
    setting explicitly. Tightened 2026-08-30 by operator directive: it must
    also be true everywhere, so this no longer accepts an opt-out with a
    note. The engine supports it -- ModelConfig.is_prefix_caching_supported
    returns True for hybrid generative models -- so an off record is a
    stale default, never a capability limit.
    """
    for profile_id, entry in registry._registry()["profiles"].items():
        for platform, record in entry.get("variants", {}).items():
            engine = record.get("engine", {})
            assert "enable_prefix_caching" in engine, (
                f"{profile_id}/{platform} does not state enable_prefix_caching; "
                "it would silently inherit vLLM's per-model default"
            )
            assert engine["enable_prefix_caching"] is True, (
                f"{profile_id}/{platform} sets enable_prefix_caching="
                f"{engine['enable_prefix_caching']!r}; SlimServe serving always "
                "has prefix caching enabled"
            )


def test_every_profile_serves_with_tool_calling_and_thinking():
    """Automatic tool calling and thinking are always on.

    Every profile names its tool-call and reasoning parsers, and the
    registry's _SERVING_DEFAULTS force enable_auto_tool_choice plus
    thinking-on template kwargs (and prefix caching) into every resolved
    plan unless a record overrides the key itself.
    """
    from slimserve.registry import _SERVING_DEFAULTS

    assert _SERVING_DEFAULTS.get("enable_auto_tool_choice") is True
    assert _SERVING_DEFAULTS.get("enable_prefix_caching") is True
    kwargs = _SERVING_DEFAULTS.get("default_chat_template_kwargs", {})
    assert kwargs.get("thinking") is True and kwargs.get("enable_thinking") is True

    for profile_id, entry in registry._registry()["profiles"].items():
        for platform, record in entry.get("variants", {}).items():
            engine = record.get("engine", {})
            assert engine.get("tool_call_parser"), (
                f"{profile_id}/{platform} has no tool_call_parser"
            )
            assert engine.get("reasoning_parser"), (
                f"{profile_id}/{platform} has no reasoning_parser"
            )
            assert engine.get("enable_auto_tool_choice", True) is True, (
                f"{profile_id}/{platform} disables automatic tool calling"
            )


def test_no_profile_quantizes_main_kv():
    """Main KV is bf16 on rtx3090; aspirational elsewhere (operator 2026-08-29).

    Quantized main KV was implicated in multi-turn tracking errors on
    Qwen3.8-Flash-Next, so no rtx3090 profile may set a quantized
    kv_cache_dtype. Other platforms keep their qualified configs until an
    on-box requalification pass flips them. Draft-model KV (DSpark
    TurboQuant) is exempt everywhere because rejection sampling verifies
    drafts against the target.
    """
    # Enforced on rtx3090 (operator 2026-08-29): quantized main KV was
    # implicated in multi-turn tracking errors on Qwen3.8-Flash-Next, and
    # this box is where that was root-caused and validated. Other platforms
    # keep their qualified configs; bf16 main KV is aspirational there and
    # flips only with an on-box requalification pass.
    # glm52-q2k-4/a100 is the sole carve-out (operator-approved 2026-08-30):
    # 65.8 GiB of Q2K weights per 80 GB rank leave no room for bf16 KV at
    # 131072 (12.2 GiB/rank), a physical impossibility, so that record
    # serves fp8 main KV, requalified by the WildChat deep-context sweep's
    # marker-recall probes. The record's note states the arithmetic.
    carve_outs = {("glm52-q2k-4", "a100"): {"fp8"}}
    for profile_id, entry in registry._registry()["profiles"].items():
        for platform, record in entry.get("variants", {}).items():
            if platform not in ("rtx3090", "a100"):
                continue
            dtype = record.get("engine", {}).get("kv_cache_dtype", "auto")
            allowed = {"auto", "bfloat16"} | carve_outs.get(
                (profile_id, platform), set()
            )
            # 'auto' resolves to the model dtype (bf16 for every supported
            # model); an explicit 'bfloat16' is the same commitment spelled
            # out and equally compliant.
            assert dtype in allowed, (
                f"{profile_id}/{platform} sets kv_cache_dtype={dtype!r}; "
                "main KV must be bf16 (auto) on rtx3090/a100 profiles "
                "outside the noted carve-outs"
            )


def test_every_a100_profile_carries_the_host_kv_tier():
    """Every A100 variant declares the HostTierConnector config.

    The connector requires the packed cross-layer KV slab: the DSV4
    records get it via the allocator's is_dsv4 gate, and the GLM records
    force it with enable_cross_layers_blocks in the connector extra
    config, because their group specs are not verified to resolve
    all-uniform on their own.
    """
    seen = 0
    for profile_id, entry in registry._registry()["profiles"].items():
        record = entry.get("variants", {}).get("a100")
        if record is None:
            continue
        seen += 1
        transfer = record["engine"]["kv_transfer_config"]
        assert transfer["kv_connector"] == "HostTierConnector", profile_id
        assert transfer["kv_role"] == "kv_both", profile_id
        extra = transfer["kv_connector_extra_config"]
        assert extra["host_tier_gb_per_rank"] > 0, profile_id
        if entry["source"] == "glm52-vision":
            assert extra["enable_cross_layers_blocks"] == "True", profile_id
    assert seen == 7, "expected all seven A100 variants to be checked"
