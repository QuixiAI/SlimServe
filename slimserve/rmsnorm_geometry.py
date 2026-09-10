# SPDX-License-Identifier: Apache-2.0
"""Opt-in, source-qualified GLM53 RMSNorm geometry causal experiment.

No compiler policy or production default. The historical four-source diagnostic
has its own flag/schema and cannot be combined with this thirteen-source test.
"""

import json
import os
import subprocess
from pathlib import Path

from slimserve import glm53_ordering, rmsnorm_diagnostic

FLAG = "SLIMSERVE_GLM53_RMSNORM_GEOMETRY"
MANIFEST = "SLIMSERVE_GLM53_GEOMETRY_MANIFEST"
SERVING_SCHEMA = "glm53-rmsnorm-geometry-serving-v1"
AOT_PAIR_SHA = "46b9c695c2734759deb55d3c15ee085dc8e3bd8eea67153ab074cdd1cb4fe946"
AOT_CLOSURE_SHA = "ff9b29e175d6d8ac2f1c787d6ca35ee522786e20541a4ac5f3ffdf6fdfd98242"


def mode():
    value = os.getenv(FLAG, "")
    if value not in ("", "control", "geometry"):
        raise ValueError(f"{FLAG} must be absent, control or geometry")
    return value


def validate_environment():
    rmsnorm_diagnostic.validate_environment()
    if os.getenv(rmsnorm_diagnostic.FLAG):
        raise ValueError("geometry cannot combine with the legacy RMSNorm diagnostic")
    for name in (
        "TORCHINDUCTOR_DETERMINISTIC",
        "TORCHINDUCTOR_BATCH_INVARIANT",
        "TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS",
        "VLLM_BATCH_INVARIANT",
    ):
        if os.getenv(name, "0") not in ("", "0"):
            raise ValueError(f"geometry forbids conflicting compiler flag: {name}")


def read_manifest():
    from benchmarks.kernels.check_glm53_attention_norms import require, verify
    from benchmarks.kernels.glm53_rmsnorm_geometry import SCHEMA

    selected = mode()
    validate_environment()
    require(bool(selected) and bool(os.getenv(MANIFEST)), "geometry manifest required")
    path = Path(os.environ[MANIFEST]).resolve()
    data = json.loads(path.read_text())
    private = Path(data["private_namespace"])
    original = Path(data["original_namespace"])
    cache = path.parent / "cache"
    require(
        data["serving_schema"] == SERVING_SCHEMA
        and data["schema"] == SCHEMA
        and data["mode"] == selected
        and data["namespace"] == rmsnorm_diagnostic.NAMESPACE
        and data["aot_qualification_sha256"] == AOT_PAIR_SHA
        and data["aot_closure_sha256"] == AOT_CLOSURE_SHA
        and data["cache_root"] == str(cache)
        and private
        == cache / "torch_compile_cache/torch_aot_compile" / data["namespace"]
        and data["receipts"] == str(path.parent / "receipts")
        and data["worker_receipts"] == str(path.parent / "worker-receipts"),
        "unexpected serving geometry manifest or private paths",
    )
    require(
        private.resolve() == private
        and original.resolve() == original
        and not private.is_relative_to(original)
        and not original.is_relative_to(private)
        and os.getenv("VLLM_CACHE_ROOT") == str(cache)
        and os.getenv("TORCHINDUCTOR_CACHE_DIR") == str(private / "inductor_cache"),
        "geometry requires exact nonoverlapping private caches",
    )
    require(
        set(data["targets"])
        == set(data["artifact_roots"])
        == set(data["expected_graphs"])
        == {str(r) for r in range(4)}
        and [len(data["targets"][str(r)]) for r in range(4)] == [3, 3, 3, 4]
        and all(
            len(data["artifact_roots"][str(r)])
            == len(data["expected_graphs"][str(r)])
            == 7
            and sum(len(t["submodules"]) for t in data["artifact_roots"][str(r)]) == 46
            for r in range(4)
        ),
        "incomplete all-rank geometry qualification",
    )
    from benchmarks.kernels.prepare_glm53_geometry_serving import (
        CASES,
        INTEGRATION_SITES,
        completed_evidence,
    )

    preparation = json.loads((path.parent.parent / "preparation.json").read_text())
    require(
        preparation["status"] == "prepared"
        and preparation["sources"] == data["sources"]
        and preparation["git_commit"]
        == data["git_commit"]
        == subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        and [(r["label"], r["mode"]) for r in preparation["runs"]] == list(CASES),
        "prepared serving source identity or case order changed",
    )
    for row in preparation["runs"]:
        expected = path.parent.parent / row["label"] / "manifest.json"
        require(
            row["manifest"] == str(expected)
            and row["manifest_sha256"] == rmsnorm_diagnostic.sha(expected),
            "prepared serving manifest changed",
        )
    require(
        path.parent.name == data["label"] and (data["label"], selected) in CASES,
        "wrong serving case or intervention mode",
    )
    qualified, graphs, reference, _ = completed_evidence(
        Path(data["aot_pair_path"]), Path(data["aot_closure_path"])
    )
    require(
        data["qualified_manifest"] == reference
        and data["qualified_graphs"] == graphs
        and set(qualified["sources"]) <= set(data["sources"])
        and all(
            data["sources"][name] == digest
            for name, digest in qualified["sources"].items()
            if Path(name).resolve() not in INTEGRATION_SITES
        )
        and all(
            data[k] == qualified[k]
            for k in (
                "targets",
                "artifact_roots",
                "expected_graphs",
                "original_files",
                "original_namespace",
            )
        ),
        "serving targets/roots or frozen implementation differ from AOT qualification",
    )
    verify(data)
    return data, path


def validate_plan(plan):
    if not mode():
        return
    validate_environment()
    glm53_ordering.validate_plan(plan)
    if plan.engine.get("kv_cache_dtype", "auto") != "auto" or plan.engine.get(
        "dtype", "auto"
    ) not in ("auto", "bfloat16", "bf16"):
        raise ValueError("geometry preserves the recipe's BF16 activation/KV settings")
    options = plan.engine.get("compilation_config", {}).get(
        "inductor_compile_config", {}
    )
    if options.get("deterministic") or options.get("combo_kernels") is False:
        raise ValueError(
            "geometry requires original compiler and attention combo policy"
        )
    read_manifest()


def install(runner):
    # Inert for every default profile; no Torch or diagnostic-loader import here.
    if not mode():
        return
    validate_environment()
    import torch

    from benchmarks.kernels.glm53_geometry_serving import ServingGeometry

    config = runner.model_config.hf_text_config
    parallel = runner.parallel_config
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
        or getattr(runner, "_slimserve_geometry", None) is not None
    ):
        raise ValueError(
            "geometry requires original-policy GLM53 SM120 TP4, installed once"
        )
    manifest, path = read_manifest()
    diagnostic = ServingGeometry(parallel.rank, manifest, path)
    diagnostic.install(runner)
    runner._slimserve_geometry = diagnostic
