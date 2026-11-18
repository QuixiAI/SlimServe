"""Image inputs for the q38 native engine.

The language model runs in the q38 library; the vision tower (a 27-block ViT,
0.45 B parameters) runs here through the reference PyTorch module, in the
server process, and hands the engine one projected embedding row per image
token. Rows are cached by the image's content hash, so an image that stays in
a conversation is encoded once.

What the engine needs for a prompt with images (`plan`):
  tokens  the prompt ids with every image row replaced by a negative id that
          is a function of (image hash, row). The engine's prefix snapshots are
          keyed by token ids, so two different images never share a prefix.
  embeds  uint16 [rows, hidden]: BF16 bits of the image rows, in prompt order.
  pos3    int32 [n, 3]: (temporal, height, width) rotary position per token.
          Text advances all three together; an image at position p with an
          h x w grid of merged patches puts row (y, x) at (p, p + y, p + x) and
          moves the next text token to p + max(h, w) (the reference
          get_rope_index of Qwen3.5).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

import numpy as np
import torch


class VisionTower:
    def __init__(self, model_dir: str, *, device: str | None = None, cache_bytes: int = 2 << 30) -> None:
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        cfg = AutoConfig.from_pretrained(model_dir)
        self.merge = int(cfg.vision_config.spatial_merge_size)
        self.out_dim = int(cfg.vision_config.out_hidden_size)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        model = Qwen3_5VisionModel(cfg.vision_config)
        prefix = "model.visual."
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for key, fname in wmap.items():
            if key.startswith(prefix):
                by_file.setdefault(fname, []).append(key)
        state = {}
        for fname, keys in by_file.items():
            with safe_open(os.path.join(model_dir, fname), framework="pt") as sf:
                for key in keys:
                    state[key[len(prefix):]] = sf.get_tensor(key)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision tower weights do not match: missing {missing[:4]}, unexpected {unexpected[:4]}")
        self.model = model.to(self.device, torch.float32).eval()
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes = cache_bytes

    @torch.inference_mode()
    def encode(self, key: str, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> np.ndarray:
        """BF16 bits [rows, out_dim] of one image; `key` is its content hash."""
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        grid = grid_thw.reshape(1, 3).to(torch.long)
        out = self.model(pixel_values.to(self.device, torch.float32), grid_thw=grid.to(self.device),
                         return_dict=True).pooler_output
        rows = out.to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)
        rows = np.ascontiguousarray(rows)
        self._cache[key] = rows
        while len(self._cache) > 1 and sum(v.nbytes for v in self._cache.values()) > self._cache_bytes:
            self._cache.popitem(last=False)
        return rows

    def cached(self, key: str) -> np.ndarray | None:
        return self._cache.get(key)


def _row_ids(key: str, n: int) -> np.ndarray:
    base = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little")
    return (-1 - ((base + np.arange(n, dtype=np.uint64)) % np.uint64(0x7FFFFFFF)).astype(np.int64)).astype(np.int32)


def plan(ids: list[int], images: list[tuple[int, int, np.ndarray | None, str, np.ndarray, tuple[int, int]]]):
    """Engine inputs for a prompt with images.

    `images`: one (offset, length, is_embed mask or None, key, rows, (h, w))
    per image, where [offset, offset + length) is its placeholder range in
    `ids`, `rows` its embedding rows and (h, w) its merged-patch grid.
    Returns (tokens int32 [n], embeds uint16 [rows, hidden], pos3 int32 [n, 3]).
    """
    n = len(ids)
    tokens = np.asarray(ids, dtype=np.int32).copy()
    pos3 = np.zeros((n, 3), dtype=np.int32)
    parts = []
    cur = 0   # next rotary position
    at = 0    # next prompt index
    for offset, length, mask, key, rows, (h, w) in sorted(images, key=lambda im: im[0]):
        where = np.arange(offset, offset + length)
        if mask is not None:
            where = where[np.asarray(mask, dtype=bool)]
        if offset < at or len(where) != h * w or rows.shape[0] != h * w:
            raise ValueError("image placeholder ranges do not match the image grids")
        # text up to the first image row, and any non-image tokens inside the range
        first = int(where[0])
        pos3[at:first] = (cur + np.arange(first - at, dtype=np.int32))[:, None]
        cur += first - at
        if not np.array_equal(where, np.arange(first, first + h * w)):
            raise ValueError("image rows are not contiguous in the prompt")
        k = np.arange(h * w, dtype=np.int32)
        pos3[where, 0] = cur
        pos3[where, 1] = cur + k // w
        pos3[where, 2] = cur + k % w
        cur += max(h, w)
        tokens[where] = _row_ids(key, h * w)
        parts.append(rows)
        at = first + h * w
    pos3[at:] = (cur + np.arange(n - at, dtype=np.int32))[:, None]
    embeds = np.ascontiguousarray(np.concatenate(parts, axis=0))
    return tokens, embeds, np.ascontiguousarray(pos3)
