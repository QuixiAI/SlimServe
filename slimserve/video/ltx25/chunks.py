# SPDX-License-Identifier: Apache-2.0
"""Long clips in overlapping temporal windows: the layout, carry and seam
blending of Lightricks' `ltx_pipelines.chunks` (layout.py, planner.py,
operators.py). The denoising per window is the engine's
(`pipeline.LTX25Engine._chunked`); this module is the pure arithmetic.

A clip of F frames is cut into windows of `chunk_pixel_frames` (8k + 1) that
share `carry_frames` (8k + 1, at least 17) with the next one: after a window
is denoised, its last carry latent frames are pinned (clean, latent index 0)
at the start of the next window, which therefore re-renders them. The kept
output of a window starts after its incoming carry; the decoded seam is
crossfaded over `blend_frames` (default the whole carry) with linear weights
and the audio over 40 ms with an equal-power fade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

T = 8  # the VAE's temporal scale


@dataclass(frozen=True)
class ChunkConfig:
    """Upstream ChunkConfig: window, carry and decoded blend, in pixel frames."""

    chunk_pixel_frames: int = 97
    carry_frames: int = 25
    blend_frames: int | None = None

    def __post_init__(self) -> None:
        blend = self.carry_frames if self.blend_frames is None else self.blend_frames
        object.__setattr__(self, "blend_frames", blend)
        if blend < 0 or blend > self.carry_frames:
            raise ValueError(
                f"blend_frames ({blend}) must be between 0 and carry_frames "
                f"({self.carry_frames})"
            )


@dataclass(frozen=True)
class Layout:
    """Upstream ChunkLayout: one window on the stitched pixel timeline."""

    pixel_frames: int
    start_pixel_frame: int
    prev_carry: int
    next_carry: int
    blend: int

    @property
    def owned(self) -> tuple[int, int]:
        """The stitched range this window ships (after the incoming carry)."""
        return (
            self.start_pixel_frame + self.prev_carry,
            self.start_pixel_frame + self.pixel_frames,
        )

    def local_frame(self, global_idx: int) -> int | None:
        """Upstream local_frame_index: the window-local index of a stitched
        frame this window owns, else None."""
        lo, hi = self.owned
        return global_idx - self.start_pixel_frame if lo <= global_idx < hi else None

    @property
    def latent_frames(self) -> int:
        return (self.pixel_frames - 1) // T + 1


def _on_grid(name: str, value: int) -> None:
    if value != 0 and (value - 1) % T:
        raise ValueError(f"{name} ({value}) must be on the causal grid (8k + 1), or 0")


def chunk_lengths(target_frames: int, chunk_frames: int, carry: int) -> list[int]:
    """Upstream _split_target_pixel_frames_into_chunk_lengths."""
    _on_grid("chunk_pixel_frames", chunk_frames)
    _on_grid("carry_frames", carry)
    chunk_lat = (chunk_frames - 1) // T + 1
    target_lat = (target_frames - 1) // T + 1
    if target_lat <= chunk_lat:
        return [(target_lat - 1) * T + 1]
    carry_lat = (carry - 1) // T + 1 if carry else 0
    if carry_lat < 3:
        raise ValueError("carry_frames must be at least 17 when chunking")
    if carry_lat >= chunk_lat:
        raise ValueError("carry_frames must be less than chunk_pixel_frames")
    new_lat = chunk_lat - carry_lat
    extra_full, remainder = divmod(target_lat - chunk_lat, new_lat)
    sizes = [chunk_frames] * (1 + extra_full)
    if remainder:
        sizes.append((carry_lat + remainder - 1) * T + 1)
    return sizes


def layouts(target_frames: int, config: ChunkConfig) -> list[Layout]:
    """Upstream uniform_chunk_layouts (the target floors to the grid)."""
    sizes = chunk_lengths(target_frames, config.chunk_pixel_frames, config.carry_frames)
    out: list[Layout] = []
    start = 0
    last = len(sizes) - 1
    for i, size in enumerate(sizes):
        prev = 0 if i == 0 else out[i - 1].next_carry
        nxt = 0 if i == last else config.carry_frames
        out.append(
            Layout(
                pixel_frames=size,
                start_pixel_frame=start - prev,
                prev_carry=prev,
                next_carry=nxt,
                blend=0 if i == last else config.blend_frames,
            )
        )
        start += size - prev
    return out


def spaced(n: int, frames: int) -> list[int]:
    """Upstream evenly_spaced_keyframe_positions."""
    if n < 0:
        raise ValueError("keyframe count must be non-negative")
    if n == 0:
        return []
    if frames < n + 2:
        raise ValueError(f"{n} keyframes need at least {n + 2} frames, got {frames}")
    return [int(x) for x in np.rint(np.linspace(0, frames - 1, n + 2))[1:-1]]


def plan_keyframes(generated, num_frames: int, plan: list[Layout]) -> list[int]:
    """Upstream plan_chunk_keyframes: a count is a per-window budget (a
    non-final window keeps one slot at its seam and spreads the rest over the
    frames it owns; the final window spreads them all); a list is global."""
    if isinstance(generated, bool):
        raise ValueError("generated keyframes is a count or a list of frames")
    if isinstance(generated, int):
        if generated <= 0:
            return []
        if len(plan) <= 1:
            return spaced(generated, num_frames)
        positions: set[int] = set()
        for i, layout in enumerate(plan):
            final = i == len(plan) - 1
            start, end = layout.owned
            if end <= start:
                continue
            count = generated if final else generated - 1
            try:
                inner = [start + p for p in spaced(count, end - start)]
            except ValueError as exc:
                raise ValueError(
                    f"generated_keyframes={generated} does not fit in the window "
                    f"[{start}, {end})"
                ) from exc
            positions.update(inner if final else [*inner, end - 1])
        return sorted(positions)
    frames = sorted({int(x) for x in generated})
    if frames and (frames[0] < 0 or frames[-1] >= num_frames):
        raise ValueError(f"generated keyframes must lie in [0, {num_frames})")
    return frames


def carry_audio_frames(carry_px: int, fps: float) -> int:
    """Upstream: max(1, AudioLatentShape.from_duration(carry / fps).frames)."""
    return max(1, round(carry_px / fps * 25.0))


def audio_window(full: np.ndarray | None, layout: Layout, fps: float, tokens: int):
    """Upstream audio_latent_for_layout on (1, T, C) tokens: the slice
    covering the window, zero-padded on the right."""
    if full is None:
        return None
    start = round(layout.start_pixel_frame / fps * 25.0)
    piece = full[:, start : start + tokens]
    if piece.shape[1] < tokens:
        pad = np.zeros(
            (piece.shape[0], tokens - piece.shape[1], piece.shape[2]), piece.dtype
        )
        piece = np.concatenate([piece, pad], axis=1)
    return piece


def crossfade_video(previous: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Upstream _crossfade_video_overlap: the pending tail of the last window
    lerped into the next window's re-rendered overlap with weights
    1 / (n + 1) ... n / (n + 1)."""
    count = min(previous.shape[0], overlap.shape[0])
    if count == 0:
        return previous
    w = (np.arange(1, count + 1, dtype=np.float32) / (count + 1)).reshape(
        count, *([1] * (previous.ndim - 1))
    )
    a = previous[-count:].astype(np.float32)
    b = overlap[-count:].astype(np.float32)
    blended = a + (b - a) * w
    if previous.dtype == np.uint8:
        blended = np.rint(blended)
    return np.concatenate([previous[:-count], blended.astype(previous.dtype)])


AUDIO_SEAM_CROSSFADE_MS = 40.0


def crossfade_audio(left: np.ndarray, right: np.ndarray, fade: int):
    """Upstream _crossfade_audio_seam: an equal-power (cos / sin) fade over
    `fade` samples; returns the adjusted left and right waveforms."""
    fade = min(fade, left.shape[-1], right.shape[-1])
    if fade <= 0:
        return left, right
    t = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out, inn = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    mixed = left[..., -fade:] * out + right[..., :fade] * inn
    return np.concatenate([left[..., :-fade], mixed], axis=-1), right[..., fade:]
