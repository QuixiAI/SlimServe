# SPDX-License-Identifier: Apache-2.0
"""Explicit indexer cancellation diagnostic; inert on production profiles."""

import os
import sys

from slimserve import glm53_ordering, kv_diagnostic

FLAG = "SLIMSERVE_GLM53_INDEXER_CORRECTION"
MANIFEST = "SLIMSERVE_GLM53_INDEXER_CORRECTION_MANIFEST"
SCHEMA = "glm53-indexer-correction-loader-v1"
SERVING_SCHEMA = "glm53-indexer-correction-serving-v1"
AOT_PAIR_SHA = "89fecf567bfe98ebcdb8ae6b948db7ad7387f4492877cba52c1f90ba65206061"
CASES = (
    ("control", "control"),
    ("correction", "correction"),
    ("return-control", "control"),
)


def mode():
    value = os.getenv(FLAG, "")
    if value not in ("", "control", "correction"):
        raise ValueError(f"{FLAG} must be absent, control or correction")
    return value


def validate_environment():
    kv_diagnostic.validate_environment()
    if os.getenv(kv_diagnostic.FLAG):
        raise ValueError("indexer correction cannot combine with KV diagnostic")


def read_manifest():
    from benchmarks.kernels import prepare_glm53_indexer_correction_serving

    return kv_diagnostic.read_manifest(
        workflow=sys.modules[__name__],
        preparation=prepare_glm53_indexer_correction_serving,
    )


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
            "indexer correction preserves BF16 and original compiler policy"
        )
    read_manifest()


def install(runner):
    if not mode():
        return
    validate_environment()
    import torch

    from benchmarks.kernels.glm53_indexer_correction_serving import (
        ServingIndexerCorrection,
        runtime_envelope,
    )

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
        or getattr(runner, "_slimserve_indexer_correction", None) is not None
    ):
        raise ValueError(
            "indexer correction requires original-policy GLM53 SM120 TP4, once"
        )
    manifest, path = read_manifest()
    runtime_envelope(runner, manifest["selection_capacity"])
    diagnostic = ServingIndexerCorrection(parallel.rank, manifest, path)
    diagnostic.install(runner)
    runner._slimserve_indexer_correction = diagnostic
