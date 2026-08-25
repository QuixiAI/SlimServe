# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF weight adapter for the DFlash 2 drafter (Qwen3.8-27B).

The backbone tensors use exactly the dense-GQA `dflash` schema the
Muse-Glimmer drafter adapter already maps (split q/k/v, per-head q/k norms,
SwiGLU, `fc`, `enc.output_norm`). DFlash 2 adds per-layer two-tap conv
tensors and three top-level selector tensors; the conv projections and the
selector tables arrive Q4_K-quantized and are dequantized at load -- they
run through plain fp16 modules (`DFlash2QwenDecoderLayer` /
`DFlash2QwenDraftModel`), not the GGUF quant method. Sizes are small next
to the backbone: 2 x ~127 MB for the A/B tables, ~13 MB per conv
projection.
"""

from collections.abc import Iterable

import gguf
import torch

from vllm.config import ModelConfig
from vllm.transformers_utils.gguf_utils import gguf_reader

from .muse_glimmer import MuseGlimmerDFlashGGUFAdapter

_DFLASH2_TOP_RENAMES = {
    "selector_predecessor.weight": "selector_predecessor.weight",
    "selector_successor.weight": "selector_successor.weight",
    "selector_hidden.weight": "selector_hidden.weight",
}

_DFLASH2_BLK_KEYS = (
    "attn_conv_base",
    "attn_conv_proj.weight",
    "ffn_conv_base",
    "ffn_conv_proj.weight",
)

# HF-name stems whose weights bypass the GGUF quant method (dequantized at
# load into plain modules).
_DEQUANT_SUBSTRS = ("selector_", "_conv_proj", "_conv_base")


class DFlash2QwenGGUFAdapter(MuseGlimmerDFlashGGUFAdapter):
    """The DFlash 2 drafter GGUF (`dflash` schema + selector metadata)."""

    def build_name_map(self, model_config: ModelConfig) -> dict[str, str]:
        config = model_config.hf_config
        name_map = super().build_name_map(model_config)
        name_map.update(_DFLASH2_TOP_RENAMES)
        for idx in range(config.num_hidden_layers):
            for key in _DFLASH2_BLK_KEYS:
                name_map[f"blk.{idx}.{key}"] = f"model.layers.{idx}.{key}"
        return name_map

    def prepare_weights(
        self, model_config: ModelConfig
    ) -> Iterable[tuple[str, torch.Tensor]]:
        assert self.load_spec is not None
        reverse = {
            hf: g for g, hf in (self.load_spec.gguf_to_hf_name_map or {}).items()
        }
        for name, weight in super().prepare_weights(model_config):
            stem = None
            if name.endswith(".qweight"):
                stem = name.removesuffix(".qweight")
            elif name.endswith(".qweight_type"):
                stem = name.removesuffix(".qweight_type")
            if stem is not None and any(s in stem for s in _DEQUANT_SUBSTRS):
                if name.endswith(".qweight_type"):
                    continue
                gguf_name = reverse[stem + ".weight"]
                yield stem + ".weight", self._dequantized_tensor(gguf_name)
                continue
            yield name, weight

    def _dequantized_tensor(self, gguf_name: str) -> torch.Tensor:
        assert self.load_spec is not None
        for tensor in gguf_reader(self.load_spec.weights_source[0]).tensors:
            if tensor.name == gguf_name:
                data = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                return torch.from_numpy(data)
        raise RuntimeError(f"{gguf_name} missing from the drafter GGUF")
