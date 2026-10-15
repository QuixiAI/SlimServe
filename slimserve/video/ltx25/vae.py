# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 video VAE (conv decoder + causal conv3d encoder), official checkpoint.

`vae/ltx-2.5-video-vae-conv-bf16.safetensors`: 128 latent channels, 32x spatial
and 8x temporal compression, pixel-norm residual stages, depth-to-space
upsamples, no timestep conditioning, non-causal decoder (replicate temporal
padding), causal encoder. Reads against Lightricks'
`ltx_core/model/video_vae/`; weights keep their upstream names.

Activations are channels-last (B, D, H, W, C); the public surface takes and
returns the PyTorch layout (B, C, F, H, W). Every convolution goes through
`conv3d` below, which is where a custom kernel replaces `mx.conv3d`.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints

G = mx.float32
SCALE_T, SCALE_S = 8, 32
SLAB_SCRATCH_BYTES = 4 << 30  # im2col-equivalent scratch allowed per conv slab

# Optional profiler: called as hook(tag, x_shape, w_shape, seconds) with the conv
# evaluated synchronously. Leave None in production (it serializes the graph).
CONV_HOOK: Callable[[str, tuple, tuple, float], None] | None = None


def conv3d(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    causal: bool = False,
    tag: str = "",
    act: bool = False,
    operand: mx.Dtype | None = None,
    stream: mx.Dtype | None = None,
) -> mx.array:
    """3x3x3 stride-1 conv, optionally preceded by pixel-norm + SiLU (`act`).

    Temporal padding replicates the edge frame (front only when causal),
    spatial padding is zeros. x (B, D, H, W, I), w (O, 3, 3, 3, I).

    Evaluated in temporal slabs: output frame t reads input frames t-1..t+1
    (t-2..t when causal), so slabs are exact. MLX's conv3d allocates scratch
    proportional to its whole input (a 768x512x121 decode peaked near 60 GiB
    for activations of 1.5 GiB each). Each slab gathers its own frames, applies
    the activation, pads and convolves, so no full-size padded or activated
    copy of the input ever exists and equal-shaped slabs reuse buffers.
    """
    d = x.shape[1]
    front = 2 if causal else 1
    per_frame = (
        (x.shape[2] + 2)
        * (x.shape[3] + 2)
        * max(x.shape[4], w.shape[0])
        * (operand or x.dtype).size
    )
    frames = max(1, int(SLAB_SCRATCH_BYTES // (27 * per_frame)))
    mx.eval(x)
    t_start = time.perf_counter() if CONV_HOOK is not None else 0.0
    out = []
    for t in range(0, d, frames):
        hi = min(t + frames, d)
        idx = [min(max(i - front, 0), d - 1) for i in range(t, hi + 2)]
        xs = (
            x[:, idx[0] : idx[-1] + 1]
            if idx == list(range(idx[0], idx[-1] + 1))
            else mx.take(x, mx.array(idx), axis=1)
        )
        if act:
            xs = silu(pixel_norm(xs))
        if operand is not None:
            xs = xs.astype(operand)
        y = conv3d_core(mx.pad(xs, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)]), w, b)
        if stream is not None:
            y = y.astype(stream)
        mx.eval(y)
        out.append(y)
    y = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
    if CONV_HOOK is not None:
        mx.eval(y)
        mx.synchronize()
        CONV_HOOK(
            tag,
            (x.shape[0], d + 2, x.shape[2] + 2, x.shape[3] + 2, x.shape[4]),
            tuple(w.shape),
            time.perf_counter() - t_start,
        )
    return y


# ---- 3x3x3 valid conv on a padded block: MLX conv3d or a per-tap split-K GEMM --
# MLX's conv3d is a Winograd path that runs at an effective 18-25 TF/s on the
# wide-grid, narrow-channel layers but at 1.4-3 TF/s on the small-grid,
# 1024-channel ones (the VAE's first stages, the latent upscaler). Those are
# GEMM-shaped: 27 taps x C_in. Summing one GEMM per three taps (K = 3 C_in,
# below the K cliff of section 13) runs them at 8-14 TF/s in fp32 with
# fp32-level agreement (rel 3e-6). The choice is measured once per shape in
# this process and cached.
TAPS_PER_GEMM = 3
_CONV_CHOICE: dict[tuple, str] = {}
_CONV_TIMINGS: dict[tuple, dict] = {}
_CHOICE_FILE = Path(
    os.environ.get(
        "SLIMSERVE_LTX25_TUNE", "~/.cache/slimserve/ltx25_conv3d_choice.json"
    )
).expanduser()


def _choice_key(key: tuple) -> str:
    return json.dumps(key)


def _load_choices() -> None:
    """The table is per machine (chip and MLX version), persisted so a cold
    process pays no timing. 2-3 s of measurement per process otherwise."""
    if _CONV_CHOICE:
        return
    try:
        data = json.loads(_CHOICE_FILE.read_text())
        if data.get("device") == _device_tag():
            _CONV_CHOICE.update({k: v for k, v in data["choices"].items()})
    except (OSError, ValueError, KeyError):
        pass
    _CONV_CHOICE.setdefault("__loaded__", "1")


def _device_tag() -> str:
    info = mx.device_info()
    return f"{info.get('device_name')} mlx{mx.__version__}"


def _save_choices() -> None:
    try:
        _CHOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CHOICE_FILE.with_suffix(".tmp")
        choices = {k: v for k, v in _CONV_CHOICE.items() if k != "__loaded__"}
        tmp.write_text(
            json.dumps({"device": _device_tag(), "choices": choices}, indent=0)
        )
        tmp.replace(_CHOICE_FILE)
    except OSError:
        pass


_TAPS = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]


def _conv_tap_gemm(xp: mx.array, w: mx.array, b: mx.array) -> mx.array:
    bsz, dp, hp, wp, c = xp.shape
    d, h, wd = dp - 2, hp - 2, wp - 2
    y = None
    for s in range(0, 27, TAPS_PER_GEMM):
        group = _TAPS[s : s + TAPS_PER_GEMM]
        cols = mx.concatenate(
            [xp[:, i : i + d, j : j + h, k : k + wd, :] for (i, j, k) in group], axis=-1
        ).reshape(bsz * d * h * wd, -1)
        wm = mx.concatenate([w[:, i, j, k, :] for (i, j, k) in group], axis=-1)
        part = cols @ wm.T
        y = part if y is None else y + part
    return (y + b).reshape(bsz, d, h, wd, -1)


def conv3d_core(xp: mx.array, w: mx.array, b: mx.array) -> mx.array:
    """Valid 3x3x3 conv of an already padded block, by the faster of the two
    formulations for this shape (measured on first use)."""
    _load_choices()
    key = _choice_key((tuple(xp.shape), tuple(w.shape), str(xp.dtype)))
    choice = _CONV_CHOICE.get(key)
    if choice is None:
        if xp.shape[-1] < 256:  # narrow channels: Winograd wins by far, no need to time
            choice = "mlx"
        else:
            timings = {}
            for name, fn in (
                ("mlx", lambda: mx.conv3d(xp, w) + b),
                ("taps", lambda: _conv_tap_gemm(xp, w, b)),
            ):
                mx.eval(fn())
                mx.synchronize()
                t0 = time.perf_counter()
                mx.eval(fn())
                mx.synchronize()
                timings[name] = time.perf_counter() - t0
            choice = min(timings, key=timings.get)
            _CONV_TIMINGS[key] = timings
        _CONV_CHOICE[key] = choice
        _save_choices()
    return mx.conv3d(xp, w) + b if choice == "mlx" else _conv_tap_gemm(xp, w, b)


def pixel_norm(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, None, 1e-8)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def depth_to_space(x: mx.array, sf: int, tf: int) -> mx.array:
    """b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3), channels-last."""
    b, d, h, w, c = x.shape
    c //= sf * sf * tf
    x = x.reshape(b, d, h, w, c, tf, sf, sf).transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, d * tf, h * sf, w * sf, c)


def space_to_depth(x: mx.array, st: int, sh: int, sw: int) -> mx.array:
    b, d, h, w, c = x.shape
    x = x.reshape(b, d // st, st, h // sh, sh, w // sw, sw, c).transpose(
        0, 1, 3, 5, 7, 2, 4, 6
    )
    return x.reshape(b, d // st, h // sh, w // sw, c * st * sh * sw)


# ---- tiling (upstream video_vae/tiling.py; latent tiles, trapezoidal blends) ----
@dataclass(frozen=True)
class Tiling:
    """Tile sizes/overlaps in pixels and frames; None disables that axis."""

    spatial: tuple[int, int] | None = None  # (tile px, overlap px), multiples of 32
    temporal: tuple[int, int] | None = (
        None  # (tile frames, overlap frames), multiples of 8
    )

    def describe(self) -> str:
        parts = []
        if self.temporal:
            parts.append(f"frames={self.temporal[0]}/{self.temporal[1]}")
        if self.spatial:
            parts.append(f"px={self.spatial[0]}/{self.spatial[1]}")
        return " ".join(parts) or "untiled"


def _trapezoid(
    length: int, ramp_left: int, ramp_right: int, left_from_0: bool
) -> mx.array:
    ramp_left, ramp_right = (
        max(0, min(ramp_left, length)),
        max(0, min(ramp_right, length)),
    )
    mask = mx.ones(length)
    if ramp_left > 0:
        fade = mx.linspace(0.0, 1.0, ramp_left + (1 if left_from_0 else 2))[:-1]
        if not left_from_0:
            fade = fade[1:]
        mask = mx.concatenate([mask[:ramp_left] * fade, mask[ramp_left:]])
    if ramp_right > 0:
        fade = mx.linspace(1.0, 0.0, ramp_right + 2)[1:-1]
        mask = mx.concatenate([mask[:-ramp_right], mask[-ramp_right:] * fade])
    return mx.clip(mask, 0.0, 1.0)


def _split(length: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """[(start, end, left_ramp, right_ramp)] with symmetric overlaps."""
    if length <= size:
        return [(0, length, 0, 0)]
    n = (length + size - 2 * overlap - 1) // (size - overlap)
    out = []
    for i in range(n):
        s = i * (size - overlap)
        out.append(
            (
                s,
                length if i == n - 1 else s + size,
                overlap if i else 0,
                overlap if i < n - 1 else 0,
            )
        )
    return out


def _axis_mask(mask: mx.array | None, axis: int) -> mx.array:
    shape = [1] * 5
    if mask is None:
        return mx.ones(1).reshape(shape)
    shape[axis] = mask.shape[0]
    return mask.reshape(shape)


def decode_tiles(
    latent_shape: tuple[int, ...], tiling: Tiling
) -> list[tuple[tuple, tuple, mx.array]]:
    """[(latent slices, pixel slices, blend mask)], temporal-major order."""
    _, _, f, h, w = latent_shape
    axes: list[list[tuple[slice, slice, mx.array | None]]] = [
        [(slice(0, None), slice(0, None), None)]
    ] * 2
    if tiling.temporal:
        size, ov = tiling.temporal[0] // SCALE_T, tiling.temporal[1] // SCALE_T
        t_axis = []
        for i, (s, e, lr, rr) in enumerate(_split(f, size, ov)):
            if i and f > size:  # causal: later tiles start one latent early
                s, lr = s - 1, lr + 1
            start, stop = s * SCALE_T, 1 + (e - 1) * SCALE_T
            lrf = 0 if lr == 0 else 1 + (lr - 1) * SCALE_T
            t_axis.append(
                (
                    slice(s, e),
                    slice(start, stop),
                    _trapezoid(stop - start, lrf, rr * SCALE_T, True),
                )
            )
        axes.append(t_axis)
    else:
        axes.append([(slice(0, None), slice(0, None), None)])
    for n in (h, w):
        if tiling.spatial:
            size, ov = tiling.spatial[0] // SCALE_S, tiling.spatial[1] // SCALE_S
            size = max(max(2, ov + 1), round(size * n / max(h, w)))
            axes.append(
                [
                    (
                        slice(s, e),
                        slice(s * SCALE_S, e * SCALE_S),
                        _trapezoid(
                            (e - s) * SCALE_S, lr * SCALE_S, rr * SCALE_S, False
                        ),
                    )
                    for s, e, lr, rr in _split(n, size, ov)
                ]
            )
        else:
            axes.append([(slice(0, None), slice(0, None), None)])
    tiles = []
    for combo in itertools.product(*axes):
        mask = _axis_mask(combo[0][2], 0)
        for ax in range(1, 5):
            mask = mask * _axis_mask(combo[ax][2], ax)
        tiles.append((tuple(c[0] for c in combo), tuple(c[1] for c in combo), mask))
    return tiles


# fp32 untiled decode peak = DECODE_BASE_BYTES + BYTES_PER_PIXEL_FRAME x pixel-frames.
# Measured 2026-10-02 with slab convolutions: 768x512x121 peaks at 10.1 GiB and
# 1536x1024x121 at 27.3 GiB (slope 129 B, intercept 4.4 GiB: weights + slab scratch).
# Before slabs the short clip peaked near 60 GiB.
BYTES_PER_PIXEL_FRAME = 135
DECODE_BASE_BYTES = 5 << 30
_ACCUM_BYTES_PER_PIXEL = 4 * 3 * 4


def estimate_peak_bytes(
    latent_shape: tuple[int, ...],
    tiling: Tiling | None,
    bytes_per: int = BYTES_PER_PIXEL_FRAME,
) -> int:
    _, _, f, h, w = latent_shape
    fp, hp, wp = SCALE_T * f - 7, SCALE_S * h, SCALE_S * w
    if tiling is None:
        return DECODE_BASE_BYTES + bytes_per * fp * hp * wp
    tf, th, tw = fp, hp, wp
    if tiling.temporal:
        tf = min(fp, tiling.temporal[0])
    if tiling.spatial:
        size, ov = tiling.spatial[0] // SCALE_S, tiling.spatial[1] // SCALE_S
        px = lambda n: (
            min(n, max(max(2, ov + 1), round(size * n / max(h, w)))) * SCALE_S
        )  # noqa: E731
        th, tw = px(h), px(w)
    return (
        DECODE_BASE_BYTES
        + bytes_per * tf * th * tw
        + tf * hp * wp * _ACCUM_BYTES_PER_PIXEL
    )


def plan_tiling(
    latent_shape: tuple[int, ...],
    frame_rate: float = 24.0,
    budget_bytes: int | None = None,
    bytes_per: int = BYTES_PER_PIXEL_FRAME,
) -> Tiling | None:
    """None when the whole clip fits the budget (default: half of unified
    memory), else the first rung that fits: temporal tiles 80 -> 40 frames,
    then spatial 768 -> 512 -> 256 px at 40 frames, then temporal down to 16."""
    budget = (
        budget_bytes
        if budget_bytes is not None
        else int(mx.device_info()["memory_size"]) // 2
    )
    if estimate_peak_bytes(latent_shape, None, bytes_per) <= budget:
        return None
    fp = SCALE_T * latent_shape[2] - 7

    def temporal(n: int) -> tuple[int, int]:
        return (n, min(max(8, (int(frame_rate) // 8) * 8), (int(n * 0.3) // 8) * 8))

    sizes = [n for n in range(80, 15, -8) if n < fp]
    preferred = [n for n in sizes if n >= 40]
    ladder = [Tiling(temporal=temporal(n)) for n in preferred]
    base = temporal(preferred[-1]) if preferred else None
    ladder += [
        Tiling(spatial=s, temporal=base) for s in ((768, 64), (512, 32), (256, 32))
    ]
    ladder += [Tiling(spatial=(256, 32), temporal=temporal(n)) for n in sizes if n < 40]
    for t in ladder:
        if estimate_peak_bytes(latent_shape, t, bytes_per) <= budget:
            return t
    return ladder[-1]


def to_uint8(pixels: mx.array) -> np.ndarray:
    """(1, 3, T, H, W) in [-1, 1] -> uint8 (T, H, W, 3)."""
    x = ((mx.clip(pixels[0], -1.0, 1.0) + 1.0) * 127.5).astype(mx.uint8)
    return np.array(x.transpose(1, 2, 3, 0))


class VideoVAE:
    """`dtype` is the conv operand/activation dtype. fp32 is the output-
    preserving default; see perf/ltx25_metal_campaign.md for the fp16 numbers."""

    # decoder: (kind, blocks | (spatial factor, temporal factor))
    DEC = (
        ("res", 2),
        ("up", (2, 2)),
        ("res", 2),
        ("up", (2, 2)),
        ("res", 4),
        ("up", (1, 2)),
        ("res", 6),
        ("up", (2, 1)),
        ("res", 4),
    )
    # encoder: (kind, blocks | (stride t, h, w))
    ENC = (
        ("res", 4),
        ("down", (1, 2, 2)),
        ("res", 6),
        ("down", (2, 1, 1)),
        ("res", 4),
        ("down", (2, 2, 2)),
        ("res", 2),
        ("down", (2, 2, 2)),
        ("res", 2),
    )

    def __init__(
        self,
        root: Path | None = None,
        dtype: mx.Dtype = G,
        stream: mx.Dtype | None = None,
    ):
        self.path = checkpoints.path_of("video-vae", root)
        self.dtype = dtype  # conv operands (weights and conv inputs)
        self.stream = stream or dtype  # activations between convs: norm, SiLU, residual
        self.w: dict[str, mx.array] = {}
        self.config: dict[str, Any] = {}

    def load(self, encoder: bool = True) -> VideoVAE:
        cfg = checkpoints.read_header(self.path).config()["vae"]
        want = {
            "norm_layer": "pixel_norm",
            "patch_size": 4,
            "causal_decoder": False,
            "timestep_conditioning": False,
            "spatial_padding_mode": "zeros",
            "latent_channels": 128,
        }
        for k, v in want.items():
            if cfg.get(k) != v:
                raise ValueError(
                    f"LTX-2.5 video VAE config {k}={cfg.get(k)!r}, "
                    f"engine implements {v!r}"
                )
        self.config = cfg
        raw = checkpoints.load_raw(self.path)
        w = {}
        for name, a in raw.items():
            if name.startswith("encoder.") and not encoder:
                continue
            if a.ndim == 5:  # (O, I, D, H, W) -> MLX (O, D, H, W, I)
                a = a.transpose(0, 2, 3, 4, 1)
            w[name] = a.astype(G if name.startswith("per_channel") else self.dtype)
        mx.eval(w)
        self.w = w
        self.mean = w["per_channel_statistics.mean-of-means"].reshape(1, -1, 1, 1, 1)
        self.std = w["per_channel_statistics.std-of-means"].reshape(1, -1, 1, 1, 1)
        return self

    # latents are (B, 128, F, H, W) fp32, normalized (the DiT's space)
    def normalize(self, latent: mx.array) -> mx.array:
        return (latent.astype(G) - self.mean) / self.std

    def denormalize(self, latent: mx.array) -> mx.array:
        return latent.astype(G) * self.std + self.mean

    def _conv(
        self, name: str, x: mx.array, causal: bool, act: bool = False
    ) -> mx.array:
        return conv3d(
            x,
            self.w[name + ".conv.weight"],
            self.w[name + ".conv.bias"],
            causal,
            name,
            act=act,
            operand=self.dtype,
            stream=self.stream,
        )

    def _res(self, p: str, n: int, x: mx.array, causal: bool) -> mx.array:
        for i in range(n):
            h = self._conv(f"{p}.res_blocks.{i}.conv1", x, causal, act=True)
            x = x + self._conv(f"{p}.res_blocks.{i}.conv2", h, causal, act=True)
            del h
            mx.eval(x)
        return x

    def decode_raw(self, latent: mx.array, materialize: bool = True) -> mx.array:
        """Normalized latent (B, 128, F, H, W) -> pixels (B, 3, 8F-7, 32H, 32W) fp32 in
        ~[-1, 1].

        `materialize` evaluates after every stage so the previous stage's
        activations are released before the next (larger) one; values are
        unchanged.
        """
        x = self.denormalize(latent).transpose(0, 2, 3, 4, 1).astype(self.stream)
        x = self._conv("decoder.conv_in", x, False)
        for i, (kind, arg) in enumerate(self.DEC):
            p = f"decoder.up_blocks.{i}"
            if kind == "res":
                x = self._res(p, arg, x, False)
            else:
                sf, tf = arg
                x = depth_to_space(self._conv(p + ".conv", x, False), sf, tf)
                if tf > 1:
                    x = x[:, 1:]
            if materialize:
                mx.eval(x)
        x = self._conv("decoder.conv_out", x, False, act=True)
        b, f, h, w, _ = (
            x.shape
        )  # unpatchify: b (c p r q) f h w -> b c (f p) (h q) (w r), q = r = 4
        x = (
            x.reshape(b, f, h, w, 3, 4, 4)
            .transpose(0, 1, 2, 6, 3, 5, 4)
            .reshape(b, f, h * 4, w * 4, 3)
        )
        return x.transpose(0, 4, 1, 2, 3).astype(G)

    def decode_chunks(
        self,
        latent: mx.array,
        tiling: Tiling | None | str = "auto",
        frame_rate: float = 24.0,
        budget_bytes: int | None = None,
    ) -> Iterator[mx.array]:
        """Yield evaluated pixel chunks (B, 3, T, H, W) fp32 in temporal order."""
        if tiling == "auto":
            per = (
                BYTES_PER_PIXEL_FRAME if self.dtype == G else BYTES_PER_PIXEL_FRAME // 2
            )
            tiling = plan_tiling(latent.shape, frame_rate, budget_bytes, per)
        if tiling is None:
            px = self.decode_raw(latent)
            mx.eval(px)
            yield px
            return
        yield from self._decode_tiled(latent, tiling)

    def _decode_tiled(self, latent: mx.array, tiling: Tiling) -> Iterator[mx.array]:
        b, _, f, h, w = latent.shape
        fp, hp, wp = SCALE_T * f - 7, SCALE_S * h, SCALE_S * w
        groups: list[list] = []
        for tile in decode_tiles(latent.shape, tiling):
            if groups and groups[-1][0][1][2] == tile[1][2]:
                groups[-1].append(tile)
            else:
                groups.append([tile])
        prev = prev_w = prev_start = prev_stop = None
        for group in groups:
            start, stop = group[0][1][2].indices(fp)[:2]
            buf = mx.zeros((b, 3, stop - start, hp, wp))
            wts = mx.zeros_like(buf)
            for lat_sl, px_sl, mask in group:
                tile_px = self.decode_raw(latent[lat_sl])
                n = min(stop - start, tile_px.shape[2])
                m = mask[:, :, :n] if mask.shape[2] > 1 else mask
                at = (slice(None), slice(None), slice(0, n), px_sl[3], px_sl[4])
                buf[at] = buf[at] + tile_px[:, :, :n] * m
                wts[at] = wts[at] + m
                mx.eval(buf, wts)
                del tile_px
                mx.clear_cache()
            if prev is not None:
                if prev_stop > start:
                    ov, cut = prev_stop - start, start - prev_start
                    merged = prev[:, :, cut:] + buf[:, :, :ov]
                    merged_w = prev_w[:, :, cut:] + wts[:, :, :ov]
                    prev = mx.concatenate([prev[:, :, :cut], merged], axis=2)
                    prev_w = mx.concatenate([prev_w[:, :, :cut], merged_w], axis=2)
                    buf = mx.concatenate([merged, buf[:, :, ov:]], axis=2)
                    wts = mx.concatenate([merged_w, wts[:, :, ov:]], axis=2)
                if start > prev_start:
                    chunk = (prev / mx.maximum(prev_w, 1e-8))[
                        :, :, : start - prev_start
                    ]
                    mx.eval(chunk)
                    yield chunk
            prev, prev_w, prev_start, prev_stop = buf, wts, start, stop
        chunk = prev / mx.maximum(prev_w, 1e-8)
        mx.eval(chunk)
        yield chunk

    def decode(
        self,
        latent: mx.array,
        tiling: Tiling | None | str = "auto",
        frame_rate: float = 24.0,
        budget_bytes: int | None = None,
    ) -> np.ndarray:
        """Normalized latent (1, 128, F, H, W) -> uint8 frames (8F-7, 32H, 32W, 3).

        The conv decoder is deterministic: no seed, no decode timestep.
        """
        return np.concatenate(
            [
                to_uint8(c)
                for c in self.decode_chunks(latent, tiling, frame_rate, budget_bytes)
            ]
        )

    def encode(self, pixels: mx.array) -> mx.array:
        """Pixels (B, 3, 8k+1, H, W) in [-1, 1] -> normalized latent
        (B, 128, k+1, H/32, W/32) fp32."""
        x = pixels.transpose(0, 2, 3, 4, 1).astype(self.stream)
        b, f, h, w, c = x.shape  # patchify: b c (f p) (h q) (w r) -> b (c p r q) f h w
        x = (
            x.reshape(b, f, h // 4, 4, w // 4, 4, c)
            .transpose(0, 1, 2, 4, 6, 5, 3)
            .reshape(b, f, h // 4, w // 4, c * 16)
        )
        x = self._conv("encoder.conv_in", x, True)
        for i, (kind, arg) in enumerate(self.ENC):
            p = f"encoder.down_blocks.{i}"
            if kind == "res":
                x = self._res(p, arg, x, True)
            else:
                st, sh, sw = arg
                if st == 2:
                    x = mx.concatenate([x[:, :1], x], axis=1)
                out_ch = self.w[p + ".conv.conv.weight"].shape[0] * st * sh * sw
                skip = space_to_depth(x, st, sh, sw)
                group = skip.shape[-1] // out_ch
                if group > 1:
                    skip = skip.reshape(*skip.shape[:-1], out_ch, group).mean(axis=-1)
                x = space_to_depth(self._conv(p + ".conv", x, True), st, sh, sw) + skip
            mx.eval(x)
        x = self._conv("encoder.conv_out", x, True, act=True)[..., :128]
        return self.normalize(x.transpose(0, 4, 1, 2, 3))

    def encode_tiled(self, pixels: mx.array, tiling: Tiling) -> mx.array:
        """Upstream VideoEncoder.tiled_encode with a TileSizeConfig (the
        retake / IC-LoRA source encode: TileSizeConfig.default() is frames
        80/24, height and width 768/64): the latent grid is split per axis
        (`to_splitters`: tile = max(2, overlap + 1, size // factor), the
        temporal axis causal: every tile after the first starts one latent
        frame earlier with its left ramp one longer), each tile's pixels are
        encoded on their own (frames [8 b, 1 + 8 (e - 1)), pixels [32 b,
        32 e)) and blended on the latent grid with per-axis trapezoid masks
        (the temporal one starting from 0); the masks partition unity, so no
        denominator unless they do not (then the summed weights)."""
        b, _, f, h, w = pixels.shape
        if (f - 1) % 8:
            pixels = pixels[:, :, : f - (f - 1) % 8]
            f = pixels.shape[2]
        lf, lh, lw = (f - 1) // SCALE_T + 1, h // SCALE_S, w // SCALE_S

        def axis(cfg, factor, length, temporal):
            if cfg is None:
                return [(0, length, 0, 0)]
            size, overlap = cfg[0] // factor, cfg[1] // factor
            size = max(2, overlap + 1, size)
            ivs = _split(length, size, overlap)
            if temporal and len(ivs) > 1:
                ivs = [ivs[0]] + [(s - 1, e, lr + 1, rr) for s, e, lr, rr in ivs[1:]]
            return ivs

        t_ivs = axis(tiling.temporal, SCALE_T, lf, True)
        h_ivs = axis(tiling.spatial, SCALE_S, lh, False)
        w_ivs = axis(tiling.spatial, SCALE_S, lw, False)
        buf = mx.zeros((b, 128, lf, lh, lw))
        wts = mx.zeros((lf, lh, lw))
        for (ts, te, tl, tr), (hs, he, hl, hr), (ws, we, wl, wr) in itertools.product(
            t_ivs, h_ivs, w_ivs
        ):
            tile = pixels[
                :,
                :,
                ts * SCALE_T : 1 + (te - 1) * SCALE_T,
                hs * SCALE_S : he * SCALE_S,
                ws * SCALE_S : we * SCALE_S,
            ]
            lat = self.encode(tile)
            mt = _trapezoid(te - ts, tl, tr, True)
            mh = _trapezoid(he - hs, hl, hr, False)
            mw = _trapezoid(we - ws, wl, wr, False)
            mask = mt[:, None, None] * mh[None, :, None] * mw[None, None, :]
            at = (slice(None), slice(None), slice(ts, te), slice(hs, he), slice(ws, we))
            buf[at] = buf[at] + lat * mask[None, None]
            wts[ts:te, hs:he, ws:we] = wts[ts:te, hs:he, ws:we] + mask
            mx.eval(buf, wts)
        if bool(mx.all(mx.abs(wts - 1.0) <= 1e-5).item()):
            return buf
        return buf / mx.maximum(wts, 1e-8)[None, None]
