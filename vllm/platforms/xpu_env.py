# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process environment fixes that must precede the first ``torch.xpu`` call.

Kept free of torch and vLLM imports so XPU platform detection
(``vllm.platforms.xpu_platform_plugin``) can apply them before torch.xpu
initialises the SYCL / Unified Runtime loader, which reads these variables
once.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# The UR layer that intercepts every call for tracing. Parameter validation
# and leak-checking layers are left alone.
_UR_TRACING_LAYER = "UR_LAYER_TRACING"
_XPTI_TRACING_VARS = (
    "XPTI_TRACE_ENABLE",
    "XPTI_SUBSCRIBERS",
    "XPTI_FRAMEWORK_DISPATCHER",
)
KEEP_UR_TRACING_ENV = "VLLM_XPU_KEEP_UR_TRACING"


def disable_ur_tracing() -> None:
    """Turn off UR/XPTI call tracing in this process (idempotent).

    The B70 hang investigation found the UR tracing layer active in workers
    even when the launching shell disabled it: runtime imports can populate
    these variables after exec. Only ``UR_LAYER_TRACING`` is removed from
    ``UR_ENABLE_LAYERS``; any other configured layer (validation, leak
    checking) stays enabled. ``VLLM_XPU_KEEP_UR_TRACING=1`` keeps tracing
    for debugging.
    """
    if os.environ.get(KEEP_UR_TRACING_ENV, "0").lower() in ("1", "true", "yes", "on"):
        return

    layers = os.environ.get("UR_ENABLE_LAYERS")
    if layers:
        separator = ";" if ";" in layers else ","
        kept = [
            layer
            for layer in (part.strip() for part in re.split(r"[;,]", layers))
            if layer and layer.upper() != _UR_TRACING_LAYER
        ]
        new_layers = separator.join(kept)
        if new_layers != layers:
            logger.info("XPU: removing %s from UR_ENABLE_LAYERS.", _UR_TRACING_LAYER)
            if new_layers:
                os.environ["UR_ENABLE_LAYERS"] = new_layers
            else:
                del os.environ["UR_ENABLE_LAYERS"]

    for var in _XPTI_TRACING_VARS:
        value = "0" if var == "XPTI_TRACE_ENABLE" else ""
        if os.environ.get(var, value) != value:
            logger.info("XPU: disabling XPTI tracing (%s).", var)
        os.environ[var] = value
