# SPDX-License-Identifier: Apache-2.0
"""Prompt-score memory comparison with unchanged, qualified model kernels."""

import os
import sys

from slimserve import glm53_ordering, indexer_correction_diagnostic, kv_diagnostic

FLAG = "SLIMSERVE_GLM53_PROMPT_SCORE_DIAGNOSTIC"
MANIFEST = "SLIMSERVE_GLM53_PROMPT_SCORE_MANIFEST"
CHUNKS = "SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS"
SCHEMA = indexer_correction_diagnostic.SCHEMA
SERVING_SCHEMA = "glm53-prompt-score-serving-v1"
AOT_PAIR_SHA = indexer_correction_diagnostic.AOT_PAIR_SHA
CASES = (("control", "control"), ("chunked", "control"), ("return-control", "control"))


def mode():
    value = os.getenv(FLAG, "")
    if value not in ("", "control"):
        raise ValueError(f"{FLAG} permits only the unchanged control loader")
    return value


def validate_environment():
    indexer_correction_diagnostic.validate_environment()
    if os.getenv(indexer_correction_diagnostic.FLAG):
        raise ValueError("prompt scoring cannot combine with indexer correction")


def read_manifest():
    from benchmarks.kernels import prepare_glm53_prompt_score_serving as preparation

    data, path = kv_diagnostic.read_manifest(
        workflow=sys.modules[__name__], preparation=preparation
    )
    if os.getenv(CHUNKS) != ("1" if data["label"] == "chunked" else "0"):
        raise ValueError("prompt-score chunk flag differs from prescribed arm")
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
        raise ValueError("prompt scoring preserves BF16 and original compiler policy")
    read_manifest()


def install(runner):
    if not mode():
        return
    validate_environment()
    import torch

    from benchmarks.kernels.glm53_indexer_correction_serving import (
        ServingIndexerCorrection,
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
        or getattr(runner, "_slimserve_prompt_score_diagnostic", None) is not None
    ):
        raise ValueError(
            "prompt scoring diagnostic requires original GLM53 SM120 TP4, once"
        )
    manifest, path = read_manifest()
    # All three arms use the exact qualified control loader and lifecycle.
    # No indexer correction kernel or selection arena is activated.
    diagnostic = ServingIndexerCorrection(parallel.rank, manifest, path)
    diagnostic.install(runner)
    runner._slimserve_prompt_score_diagnostic = diagnostic
