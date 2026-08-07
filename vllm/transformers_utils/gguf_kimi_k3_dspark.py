# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the Kimi K3 DSpark draft config from its GGUF metadata."""

from __future__ import annotations

import json
from functools import cache

from transformers import Qwen3Config

from vllm.transformers_utils.gguf_native import _field
from vllm.transformers_utils.gguf_utils import gguf_reader

ARCH = "dflash-draft"
NAME = "Kimi-K3-DSpark-Q8_0"
TARGET = "moonshotai/Kimi-K3"


@cache
def build_kimi_k3_dspark_config_from_gguf(gguf_path: str) -> Qwen3Config:
    """Read the exact Kimi K3 DSpark configuration embedded by its converter."""
    reader = gguf_reader(gguf_path)
    name = str(_field(reader, "general.name", ""))
    target = str(_field(reader, f"{ARCH}.dflash.target.repository", ""))
    if name != NAME or target != TARGET:
        raise ValueError(
            f"unsupported dflash-draft GGUF {name!r} for target {target!r}; "
            f"expected {NAME!r} for {TARGET!r}"
        )

    raw = _field(reader, f"{ARCH}.source.config_json")
    if not raw:
        raise ValueError(f"{NAME} has no embedded source config")
    values = json.loads(str(raw))
    block_size = int(_field(reader, f"{ARCH}.dflash.block_size"))
    values.update(
        architectures=["Qwen3DSparkModel"],
        draft_vocab_size=int(_field(reader, f"{ARCH}.vocab_size")),
        dspark_block_size=block_size,
    )
    return Qwen3Config(**values)
