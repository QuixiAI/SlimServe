# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM53 compiler-policy candidate, expressed in the recorded engine plan.

Uses a supported, cache-keyed Inductor option. No generated-source mutation,
runtime replacement, global PyTorch determinism or production default change.
"""

import copy
import os
from dataclasses import replace

from slimserve import glm53_ordering


def diagnostic_plan(plan):
    if not glm53_ordering.enabled():
        raise ValueError("deterministic reductions require native GLM53 ordering")
    glm53_ordering.validate_plan(plan)
    if os.getenv("SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC"):
        raise ValueError(
            "deterministic reductions cannot combine with cached RMSNorm intervention"
        )
    for flag in (
        "TORCHINDUCTOR_DETERMINISTIC",
        "TORCHINDUCTOR_BATCH_INVARIANT",
        "TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS",
        "VLLM_BATCH_INVARIANT",
    ):
        if os.getenv(flag, "0") not in ("", "0"):
            raise ValueError(f"use the explicit compiler plan without global {flag}")
    engine = copy.deepcopy(plan.engine)
    options = engine.setdefault("compilation_config", {}).setdefault(
        "inductor_compile_config", {}
    )
    if "deterministic" in options and options["deterministic"] is not True:
        raise ValueError("conflicting deterministic compiler option")
    options["deterministic"] = True
    return replace(plan, engine=engine)
