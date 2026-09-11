# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM53 compiler-policy candidate, expressed in the recorded engine plan.

Uses a supported, cache-keyed Inductor option. No generated-source mutation,
runtime replacement, global PyTorch determinism or production default change.

Not promoted: the 2026-09-10 full-model no-combo start repeated its scores exactly
but failed 12/32 historical per-window quality gates. Against normal compilation,
the hypothesis was repeatable reduction scheduling. Diagnostic c1/c8/c16 rates
were 155.740/574.245/777.768 tok/s, not a speedup or accepted baseline. Raw:
perf/results/2026-09-10/deterministic-no-combo-serving/fresh-a/.
Full baseline, limits and decision: docs/glm53-flash-sm120-review.md,
"Qualification record map". Do not resume this closed diagnostic campaign.
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
    # vLLM otherwise enables timed combo fusion. Inductor deliberately rejects
    # that benchmarking under deterministic=True (fresh-model qualification,
    # 2026-09-09). Disable this optional fusion in the candidate only. Merely
    # disabling its timing gate would accept previously unqualified combinations.
    policy = {
        "deterministic": True,
        "combo_kernels": False,
        "benchmark_combo_kernel": False,
    }
    for key, value in policy.items():
        if key in options and options[key] is not value:
            raise ValueError(f"conflicting deterministic compiler option: {key}")
    options.update(policy)
    return replace(plan, engine=engine)
