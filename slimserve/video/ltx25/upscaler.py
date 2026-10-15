# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 latent upscalers (LatentUpsampler): spatial x2 and temporal x2.

conv3d -> GroupNorm(32) -> SiLU -> 4 res blocks -> upsample (2-D conv +
pixel shuffle per frame, or 3-D conv + temporal shuffle dropping the first
frame) -> 4 res blocks -> conv3d. They run on DE-normalized latents:
`upscale_normalized` wraps the denormalize / renormalize the pipelines need.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.vae import conv3d_core

G = mx.float32


def _group_norm(
    x: mx.array, weight: mx.array, bias: mx.array, groups: int = 32, eps: float = 1e-5
) -> mx.array:
    """PyTorch GroupNorm on channels-last x (B, ..., C): groups of adjacent channels."""
    shape = x.shape
    b, c = shape[0], shape[-1]
    g = (
        x.reshape(b, -1, groups, c // groups)
        .transpose(0, 2, 1, 3)
        .reshape(b, groups, -1)
    )
    mean = mx.mean(g, axis=-1, keepdims=True)
    var = mx.var(g, axis=-1, keepdims=True)
    g = (g - mean) * mx.rsqrt(var + eps)
    g = g.reshape(b, groups, -1, c // groups).transpose(0, 2, 1, 3).reshape(shape)
    return g * weight + bias


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class LatentUpscaler:
    def __init__(
        self, kind: str = "spatial", root: Path | None = None, dtype: mx.Dtype = G
    ):
        if kind not in ("spatial", "temporal"):
            raise ValueError(kind)
        self.kind, self.dtype = kind, dtype
        self.path = checkpoints.path_of(f"{kind}-upscaler", root)
        self.w: dict[str, mx.array] = {}

    def load(self) -> LatentUpscaler:
        cfg = checkpoints.read_header(self.path).config()
        if (
            bool(cfg["spatial_upsample"]) != (self.kind == "spatial")
            or cfg["dims"] != 3
        ):
            raise ValueError(f"unexpected upscaler config for {self.kind}: {cfg}")
        if self.kind == "spatial" and (
            cfg["spatial_scale"] != 2.0 or cfg["rational_resampler"]
        ):
            raise ValueError(
                f"engine implements the x2 pixel-shuffle spatial upscaler, got {cfg}"
            )
        self.blocks = cfg["num_blocks_per_stage"]
        w = {}
        for name, a in checkpoints.load_raw(self.path).items():
            if a.ndim >= 4:  # (O, I, *k) -> (O, *k, I)
                a = mx.moveaxis(a, 1, -1)
            w[name] = a.astype(self.dtype)
        mx.eval(w)
        self.w = w
        return self

    def _conv(self, name: str, x: mx.array) -> mx.array:
        w = self.w[name + ".weight"]
        if w.ndim == 5:
            padded = mx.pad(x, [(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)])
            return conv3d_core(padded, w, self.w[name + ".bias"])
        return mx.conv2d(x, w, padding=1) + self.w[name + ".bias"]

    def _norm(self, name: str, x: mx.array) -> mx.array:
        return _group_norm(x, self.w[name + ".weight"], self.w[name + ".bias"])

    def _res(self, p: str, x: mx.array) -> mx.array:
        h = _silu(self._norm(p + ".norm1", self._conv(p + ".conv1", x)))
        h = self._norm(p + ".norm2", self._conv(p + ".conv2", h))
        return _silu(h + x)

    def __call__(self, latent: mx.array) -> mx.array:
        """DE-normalized latent (B, 128, F, H, W) -> (B, 128, F, 2H, 2W) or
        (B, 128, 2F-1, H, W), fp32."""
        x = latent.transpose(0, 2, 3, 4, 1).astype(self.dtype)
        x = _silu(self._norm("initial_norm", self._conv("initial_conv", x)))
        for i in range(self.blocks):
            x = self._res(f"res_blocks.{i}", x)
        b, d, h, w, c = x.shape
        if self.kind == "spatial":
            x = self._conv("upsampler.0", x.reshape(b * d, h, w, c))
            x = (
                x.reshape(b * d, h, w, c, 2, 2)
                .transpose(0, 1, 4, 2, 5, 3)
                .reshape(b, d, h * 2, w * 2, c)
            )
        else:
            x = self._conv("upsampler.0", x)
            x = (
                x.reshape(b, d, h, w, c, 2)
                .transpose(0, 1, 5, 2, 3, 4)
                .reshape(b, d * 2, h, w, c)[:, 1:]
            )
        for i in range(self.blocks):
            x = self._res(f"post_upsample_res_blocks.{i}", x)
        x = self._conv("final_conv", x).transpose(0, 4, 1, 2, 3).astype(G)
        mx.eval(x)
        return x

    def upscale_normalized(self, latent: mx.array, vae) -> mx.array:
        """Normalized latent in, normalized latent out (`vae` is a loaded VideoVAE)."""
        out = vae.normalize(self(vae.denormalize(latent)))
        mx.eval(out)
        return out
