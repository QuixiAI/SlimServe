# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF-processor facade for Qwen3.5-family GGUF artifacts.

A GGUF checkpoint carries weights, tokenizer data and hyperparameters but no
`preprocessor_config.json`, so `AutoProcessor` has nothing to load. The vision
hyperparameters we need are all present in the mmproj's `clip.*` metadata,
which `gguf_qwen35.py` already folds into the composite config's
`vision_config`; this module turns those into a processor object shaped like
the HF one so a single model implementation serves both artifact formats.

The patch layout produced here is byte-for-byte HF's Qwen2VLImageProcessor
output for still images: rows of `[C, T, P, P]` in spatial-merge order, which
is exactly what `Qwen3_VisionPatchEmbed` reinterprets as its Conv3d input.
"""

import math
from typing import Any

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

__all__ = [
    "Qwen3_5GGUFImageProcessor",
    "Qwen3_5GGUFProcessor",
    "build_qwen3_5_gguf_processor",
    "smart_resize",
]


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """HF qwen2_vl smart_resize: round to `factor`, clamp total pixels."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen3_5GGUFImageProcessor:
    """Still-image preprocessing to flattened temporal-merged patches.

    Mirrors HF's Qwen2VLImageProcessor: smart-resize (bicubic), rescale +
    normalize, duplicate the frame across the two temporal taps, then flatten
    `patch_size` patches in spatial-merge order.

    `size`, `patch_size`, `merge_size` and `temporal_patch_size` are exposed
    under the HF names because `Qwen3VLProcessingInfo._get_vision_info` and
    `_get_prompt_updates` read them off the image processor.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, vision_config: Any) -> None:
        v = vision_config
        self.patch_size = int(v.patch_size)
        self.merge_size = int(v.spatial_merge_size)
        self.temporal_patch_size = int(v.temporal_patch_size)
        image_size = int(getattr(v, "image_size", 768))
        factor = self.patch_size * self.merge_size
        # Trained envelope: the position grid at the artifact's image_size.
        self.max_pixels = int(getattr(v, "image_max_pixels", image_size * image_size))
        self.min_pixels = int(getattr(v, "image_min_pixels", 4 * factor * factor))
        self.image_mean = [float(x) for x in getattr(v, "image_mean", [0.5] * 3)]
        self.image_std = [float(x) for x in getattr(v, "image_std", [0.5] * 3)]
        # HF processing info reads bounds from here, not from our attributes.
        self.size = {
            "shortest_edge": self.min_pixels,
            "longest_edge": self.max_pixels,
        }

    def _patchify(self, img: Any) -> tuple[np.ndarray, tuple[int, int, int]]:
        from PIL import Image

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        width, height = img.size
        factor = self.patch_size * self.merge_size
        h_bar, w_bar = smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        img = img.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # [C, H, W]

        t, p, m = self.temporal_patch_size, self.patch_size, self.merge_size
        grid_h, grid_w = h_bar // p, w_bar // p
        # Still image: duplicate the frame across the temporal taps
        # (HF np.tile; llama.cpp applies both conv taps to the same frame).
        patches = np.tile(arr[np.newaxis], (t, 1, 1, 1))  # [T, C, H, W]
        patches = patches.reshape(1, t, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_h * grid_w, 3 * t * p * p)
        return flat, (1, grid_h, grid_w)

    def preprocess(self, images: Any, **kwargs: object) -> BatchFeature:
        if not isinstance(images, (list, tuple)):
            images = [images]
        flats, grids = [], []
        for img in images:
            flat, grid = self._patchify(img)
            flats.append(torch.from_numpy(flat))
            grids.append(grid)
        return BatchFeature(
            {
                "pixel_values": torch.cat(flats, dim=0),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
            },
            tensor_type=None,
        )

    def __call__(self, images: Any, **kwargs: object) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class Qwen3_5GGUFProcessor:
    """Processor facade over the GGUF tokenizer and image processor.

    Deliberately does NOT expand the single `<|image_pad|>` per image: the
    multimodal processor reports `_hf_processor_applies_updates() is False`
    for GGUF artifacts, so vLLM performs the expansion itself from the
    returned `image_grid_thw`.
    """

    # Placeholder strings, matching the HF processor surface the shared
    # Qwen3VL prompt-update code reads (`hf_processor.image_token` /
    # `.video_token`). `video_token` exists only so that code can name a
    # target it will never find: a GGUF mmproj carries no video path.
    image_token = "<|image_pad|>"
    video_token = "<|video_pad|>"

    def __init__(self, tokenizer: Any, image_processor: Any, image_token_id: int):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id

    def __call__(
        self,
        text: Any = None,
        images: Any = None,
        return_tensors: Any = None,
        **kwargs: object,
    ) -> BatchFeature:
        data: dict[str, Any] = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            encodings = [self.tokenizer(t, add_special_tokens=False) for t in text]
            data["input_ids"] = [e["input_ids"] for e in encodings]
        if images:
            data.update(self.image_processor.preprocess(images))
        return BatchFeature(data, tensor_type=None)


def build_qwen3_5_gguf_processor(
    tokenizer: Any,
    hf_config: Any,
    image_placeholder: str = Qwen3_5GGUFProcessor.image_token,
) -> Qwen3_5GGUFProcessor:
    """Build the facade, checking the tokenizer agrees with the config."""
    image_token_id = tokenizer.convert_tokens_to_ids(image_placeholder)
    if image_token_id != hf_config.image_token_id:
        raise ValueError(
            f"tokenizer maps {image_placeholder!r} to {image_token_id}, "
            f"config says {hf_config.image_token_id}"
        )
    return Qwen3_5GGUFProcessor(
        tokenizer,
        Qwen3_5GGUFImageProcessor(hf_config.vision_config),
        image_token_id,
    )


def is_gguf_multimodal(hf_config: Any) -> bool:
    """True when this config came from a GGUF text+mmproj pair."""
    return getattr(hf_config, "vision_config", None) is not None
