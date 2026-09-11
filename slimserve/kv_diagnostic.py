# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM53 KV arithmetic isolation; never a production compiler policy."""

import json
import os
import subprocess
import sys
from pathlib import Path

from slimserve import glm53_ordering, rmsnorm_diagnostic, rmsnorm_geometry

FLAG = "SLIMSERVE_GLM53_KV_DIAGNOSTIC"
MANIFEST = "SLIMSERVE_GLM53_KV_MANIFEST"
SERVING_SCHEMA = "glm53-kv-serving-v1"
AOT_PAIR_SHA = "e650a2a5f806085ab6f748169b1d54cf8950a2fd7656dfeda11d5e6f25a018f7"
CASES = (("control", "control"), ("kv", "kv"), ("return-control", "control"))


def mode():
    value = os.getenv(FLAG, "")
    if value not in ("", "control", "kv"):
        raise ValueError(f"{FLAG} must be absent, control or kv")
    return value


def validate_environment():
    rmsnorm_geometry.validate_environment()
    if os.getenv(rmsnorm_geometry.FLAG):
        raise ValueError("KV cannot combine with the RMSNorm geometry diagnostic")


def read_manifest(*, workflow=None, preparation=None):
    from benchmarks.kernels.check_glm53_attention_norms import require, verify
    from benchmarks.kernels.glm53_kv_loader import SCHEMA as KV_SCHEMA

    if preparation is None:
        from benchmarks.kernels import prepare_glm53_kv_serving as preparation
    workflow = workflow or sys.modules[__name__]
    schema = getattr(workflow, "SCHEMA", KV_SCHEMA)

    selected = workflow.mode()
    workflow.validate_environment()
    require(
        bool(selected) and bool(os.getenv(workflow.MANIFEST)),
        "diagnostic manifest required",
    )
    path = Path(os.environ[workflow.MANIFEST]).resolve()
    data = json.loads(path.read_text())
    private, original = (
        Path(data["private_namespace"]),
        Path(data["original_namespace"]),
    )
    cache = path.parent / "cache"
    require(
        data["serving_schema"] == workflow.SERVING_SCHEMA
        and data["schema"] == schema
        and data["mode"] == selected
        and data["namespace"] == rmsnorm_diagnostic.NAMESPACE
        and data["aot_qualification_sha256"] == workflow.AOT_PAIR_SHA
        and data["cache_root"] == str(cache)
        and private
        == cache / "torch_compile_cache/torch_aot_compile" / data["namespace"]
        and data["receipts"] == str(path.parent / "receipts")
        and data["worker_receipts"] == str(path.parent / "worker-receipts")
        and path.parent.name == data["label"]
        and (data["label"], selected) in workflow.CASES,
        "unexpected KV serving manifest or private paths",
    )
    require(
        private.resolve() == private
        and original.resolve() == original
        and not private.is_relative_to(original)
        and not original.is_relative_to(private)
        and os.getenv("VLLM_CACHE_ROOT") == str(cache)
        and os.getenv("TORCHINDUCTOR_CACHE_DIR") == str(private / "inductor_cache"),
        "KV requires exact nonoverlapping private caches",
    )
    prepared = json.loads((path.parent.parent / "preparation.json").read_text())
    require(
        prepared["status"] == "prepared"
        and prepared["sources"] == data["sources"]
        and prepared["integration_changes"] == data["integration_changes"]
        and prepared["git_commit"]
        == data["git_commit"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        and [(r["label"], r["mode"]) for r in prepared["runs"]] == list(workflow.CASES),
        "prepared KV serving identity or case order changed",
    )
    for row in prepared["runs"]:
        expected = path.parent.parent / row["label"] / "manifest.json"
        require(
            row["manifest"] == str(expected)
            and row["manifest_sha256"] == rmsnorm_diagnostic.sha(expected),
            "prepared KV serving manifest changed",
        )
    qualified, graphs, reference, receipts = preparation.completed_evidence(
        Path(data["aot_pair_path"])
    )
    sources, changes = preparation.serving_sources(qualified["sources"])
    sources.update(receipts)
    require(
        data["qualified_manifest"] == reference
        and data["qualified_graphs"] == graphs
        and data["integration_changes"] == changes
        and all(data["sources"].get(p) == h for p, h in sources.items())
        and all(
            data[k] == qualified[k] for k in preparation.COMMON_FIELDS if k != "sources"
        ),
        "KV serving differs from completed AOT qualification",
    )
    verify(data)
    return data, path


def validate_plan(plan):
    if not mode():
        return
    validate_environment()
    glm53_ordering.validate_plan(plan)
    options = plan.engine.get("compilation_config", {}).get(
        "inductor_compile_config", {}
    )
    if (
        plan.engine.get("kv_cache_dtype", "auto") != "auto"
        or plan.engine.get("dtype", "auto") not in ("auto", "bfloat16", "bf16")
        or options.get("deterministic")
        or options.get("combo_kernels") is False
    ):
        raise ValueError(
            "KV diagnostic preserves BF16 activation/KV and original compiler policy"
        )
    read_manifest()


def install(runner):
    if not mode():
        return
    validate_environment()
    import torch

    from benchmarks.kernels.glm53_kv_serving import ServingKV

    config, parallel = runner.model_config.hf_text_config, runner.parallel_config
    options = runner.compilation_config.inductor_compile_config
    if (
        config.hidden_size != 4096
        or config.num_hidden_layers != 45
        or parallel.tensor_parallel_size != 4
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_expert_parallel
        or runner.speculative_config is not None
        or torch.cuda.get_device_capability() != (12, 0)
        or runner.dtype is not torch.bfloat16
        or runner.kv_cache_dtype is not torch.bfloat16
        or options.get("deterministic")
        or options.get("combo_kernels") is False
        or getattr(runner, "_slimserve_kv_diagnostic", None) is not None
    ):
        raise ValueError(
            "KV diagnostic requires original-policy GLM53 SM120 TP4, installed once"
        )
    manifest, path = read_manifest()
    diagnostic = ServingKV(parallel.rank, manifest, path)
    diagnostic.install(runner)
    runner._slimserve_kv_diagnostic = diagnostic
