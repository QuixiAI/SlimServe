# SPDX-License-Identifier: Apache-2.0
"""Single opt-in native ordering policy for the fixed GLM53 SM120 recipe.

This selects the full-model-qualified kernels without enabling observers:
small fused stable routing, large stable alignment, and fused pool ordering
with smaller-ID cutoff ties. It is not a blanket determinism guarantee.
Legacy diagnostic switches remain separate controls and cannot be combined.
"""

import os

FLAG = "SLIMSERVE_GLM53_NATIVE_ORDER"
RECIPE = "glm53-redhatai-nvfp4-fp8-kda-tp4-v1"
LEGACY_FLAGS = (
    "SLIMSERVE_GLM53_CANONICAL_MOE",
    "SLIMSERVE_GLM53_STABLE_ROUTE",
    "SLIMSERVE_GLM53_STABLE_ALIGN",
    "SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER",
    "SLIMSERVE_GLM53_CANONICAL_INDEX_TIES",
    "SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED",
)


def enabled() -> bool:
    value = os.getenv(FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{FLAG} must be 0 or 1")
    if value == "1":
        conflicts = [key for key in LEGACY_FLAGS if os.getenv(key, "0") != "0"]
        if conflicts:
            raise ValueError(
                f"native ordering cannot combine with diagnostic flags: {conflicts}"
            )
        if os.getenv("SLIMSERVE_GLM_ROUTE_ALIGN", "1") == "0":
            raise ValueError("native ordering requires fused small-M routing")
    return value == "1"


def cache_factor() -> str:
    return f"glm53_native_order_v1={int(enabled())}"


def validate_plan(plan) -> None:
    if not enabled():
        return
    if not (
        plan.profile_id == "glm53-nvfp4-4"
        and plan.platform == "rtx6000"
        and plan.gpus == 4
        and plan.quant.name == "NVFP4"
        and (plan.weight_recipe or {}).get("id") == RECIPE
        and plan.engine.get("tensor_parallel_size") == 4
        and not plan.engine.get("enable_expert_parallel", False)
        and not plan.speculative
        and plan.engine.get("moe_backend") == "marlin"
    ):
        raise ValueError(
            "native ordering requires glm53-nvfp4-4/rtx6000 recipe v1, "
            "TP4, no speculation"
        )
