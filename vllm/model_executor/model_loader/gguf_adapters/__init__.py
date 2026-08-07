# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight adapters, one per GGUF architecture this fork serves.

Selection happens in `GGUFModelLoader._prepare_adapter`, on the file's
`general.architecture` rather than through `matches()`: the config is itself
built from the GGUF, so the architecture string is the one authority both
halves of loading agree on.
"""

from .base import BaseGGUFWeightsAdapter
from .deepseek4 import Deepseek4GGUFAdapter
from .default import GGUFWeightsAdapter
from .glm_dsa import GlmDsaGGUFAdapter
from .kimi_k3 import KimiK3GGUFAdapter
from .kimi_k3_dspark import KimiK3DSparkGGUFAdapter

__all__ = [
    "BaseGGUFWeightsAdapter",
    "Deepseek4GGUFAdapter",
    "GGUFWeightsAdapter",
    "GlmDsaGGUFAdapter",
    "KimiK3GGUFAdapter",
    "KimiK3DSparkGGUFAdapter",
]
