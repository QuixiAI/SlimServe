# SPDX-License-Identifier: Apache-2.0
"""Official LTX-2.5 safetensors: file set, headers, and load-time dtype policy.

Precision policy on M1-M4 (perf/ltx25_metal_campaign.md section 9):

* every GEMM operand (linear weights and their biases) is fp16. The shipped
  weights are bf16, and bf16 -> fp16 is exact for these checkpoints (no value
  outside the fp16 range, no mantissa bits lost: bf16 has 8, fp16 has 11);
* the AdaLN scale/shift tables ship as fp32 and stay fp32. They are what keeps
  the residual stream, the norms and the modulation in fp32.

Nothing here quantizes anything.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

# Relative to the model root (the layout of the Lightricks/LTX-2.5 repo).
FILES = {
    "dit-distilled": (
        "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    ),
    "dit-dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "distilled-lora": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    "detail-lora": (
        "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    ),
    "text-encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video-vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "video-vae-diffusion": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio-vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial-upscaler": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
    "temporal-upscaler": (
        "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    ),
    "duration-head": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    # Not a Lightricks file: google/gemma-4-E2B-it, the generative instruct
    # Gemma upstream's --prompt-enhancer-gemma-root names for 2.5. Its
    # config.json, generation_config.json and tokenizer.json sit next to it.
    "enhancer": "prompt_enhancer/gemma-4-E2B-it/model.safetensors",
}

DIT_PREFIX = "model.diffusion_model."
# The text connectors ship inside the DiT file but run once per prompt, on the
# text path; they are split out so the DiT forward never walks them.
CONNECTOR_PREFIXES = ("video_embeddings_connector.", "audio_embeddings_connector.")


def model_root() -> Path:
    root = os.environ.get("SLIMSERVE_LTX25_ROOT")
    if root:
        return Path(root)
    cache = os.environ.get("SLIMSERVE_CACHE")
    base = Path(cache) if cache else Path.home() / "models"
    return base / "ltx-2.5" / "official"


def path_of(component: str, root: Path | None = None) -> Path:
    return (root or model_root()) / FILES[component]


@dataclass(frozen=True)
class Header:
    tensors: dict[str, dict[str, Any]]  # name -> {dtype, shape, data_offsets}
    metadata: dict[str, str]

    def config(self, key: str = "config") -> dict[str, Any]:
        raw = self.metadata.get(key)
        return json.loads(raw) if raw else {}


def read_header(path: Path) -> Header:
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    meta = header.pop("__metadata__", {}) or {}
    return Header(tensors=header, metadata=meta)


def _is_float(a: mx.array) -> bool:
    return a.dtype in (mx.bfloat16, mx.float16, mx.float32)


def load_raw(path: Path) -> dict[str, mx.array]:
    """Lazy arrays straight from the file; nothing is read until evaluated."""
    return mx.load(str(path))


def cast_operands(
    tensors: dict[str, mx.array],
    operand: mx.Dtype = mx.float16,
    chunk_bytes: int = 4 << 30,
) -> dict[str, mx.array]:
    """Apply the load-time policy: bf16 -> `operand`, fp32 stays fp32.

    Evaluated in bounded chunks so the bf16 source and the fp16 copy of a
    42 GB transformer never coexist (two dtype copies of the DiT in one
    process pushed this box into swap during the investigation).
    """
    out: dict[str, mx.array] = {}
    pending: list[mx.array] = []
    pending_bytes = 0
    for name in list(tensors):
        a = tensors.pop(name)
        if a.dtype == mx.bfloat16:
            a = a.astype(operand)
        out[name] = a
        pending.append(a)
        pending_bytes += a.nbytes
        del a
        if pending_bytes >= chunk_bytes:
            mx.eval(pending)
            pending, pending_bytes = [], 0
            mx.clear_cache()  # return the freed bf16 source buffers to the OS
    if pending:
        mx.eval(pending)
    mx.clear_cache()
    return out


def load_dit(
    variant: str = "distilled", root: Path | None = None
) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, Any]]:
    """Returns (dit_weights, connector_weights, transformer_config).

    Keys keep the upstream spelling with `model.diffusion_model.` stripped.
    """
    path = path_of(f"dit-{variant}", root)
    header = read_header(path)
    raw = load_raw(path)
    stripped = {}
    for name in list(raw):
        if not name.startswith(DIT_PREFIX):
            raise ValueError(
                f"unexpected tensor outside the DiT in {path.name}: {name}"
            )
        stripped[name[len(DIT_PREFIX) :]] = raw.pop(name)
    weights = cast_operands(stripped)
    connectors = {
        k: weights.pop(k) for k in list(weights) if k.startswith(CONNECTOR_PREFIXES)
    }
    return weights, connectors, header.config()["transformer"]
