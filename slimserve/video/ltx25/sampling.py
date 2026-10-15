# SPDX-License-Identifier: Apache-2.0
"""Latent state, positions, noise and samplers for LTX-2.5.

Everything here is fp32: latents, noise, the x0 blend and the sampler steps.
Constants and formulas follow Lightricks' `ltx_pipelines` (utils/constants.py,
utils/samplers.py, components/diffusion_steps.py).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import numpy as np

G = mx.float32

# 8 steps; front-loaded near sigma 1 (trained with the ancestral sampler).
DISTILLED_SIGMAS = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]
STAGE_2_DISTILLED_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
ANCESTRAL_NOISE_SEED_OFFSET = 10000
# LTX-2.5 checkpoints sample stage 2 with the ancestral loop too (upstream
# distilled.py ANCESTRAL_SAMPLER_SINCE_VERSION = (2, 5)), from its own offset.
ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET = 20000

VIDEO_TEMPORAL_SCALE = 8
VIDEO_SPATIAL_SCALE = 32
AUDIO_LATENTS_PER_SECOND = 16000 / 160 / 4  # sample rate / hop / downsample = 25


def video_latent_shape(
    num_frames: int, height: int, width: int
) -> tuple[int, int, int]:
    return (num_frames + 7) // 8, height // 32, width // 32


def snap_dimensions(height: int, width: int, two_stage: bool = True) -> tuple[int, int]:
    m = 64 if two_stage else 32
    return max(m, height // m * m), max(m, width // m * m)


def audio_token_count(num_frames: int, fps: float) -> int:
    return round(num_frames / fps * AUDIO_LATENTS_PER_SECOND)


def video_positions(f: int, h: int, w: int, fps: float) -> mx.array:
    """(1, F*H*W, 3): seconds, pixel row, pixel column; midpoints, causal first
    frame."""
    idx = mx.arange(f).astype(G)
    starts = mx.maximum(idx * 8 + 1 - 8, 0.0)
    ends = mx.maximum((idx + 1) * 8 + 1 - 8, 0.0)
    t = (starts + ends) / 2.0 / fps
    y = mx.arange(h).astype(G) * 32 + 16.0
    x = mx.arange(w).astype(G) * 32 + 16.0
    grid = mx.stack(
        [
            mx.broadcast_to(t[:, None, None], (f, h, w)),
            mx.broadcast_to(y[None, :, None], (f, h, w)),
            mx.broadcast_to(x[None, None, :], (f, h, w)),
        ],
        axis=-1,
    )
    return grid.reshape(1, -1, 3)


def audio_positions(tokens: int) -> mx.array:
    """(1, T, 1): midpoint of each audio latent in seconds (causal)."""
    idx = mx.arange(tokens).astype(G)
    starts = mx.maximum(idx * 4 + 1 - 4, 0.0) * 160 / 16000
    ends = mx.maximum((idx + 1) * 4 + 1 - 4, 0.0) * 160 / 16000
    return ((starts + ends) / 2.0)[None, :, None]


def patchify(latent: mx.array) -> mx.array:
    b, c, f, h, w = latent.shape
    return latent.transpose(0, 2, 3, 4, 1).reshape(b, f * h * w, c)


def unpatchify(tokens: mx.array, dims: tuple[int, int, int]) -> mx.array:
    f, h, w = dims
    return tokens.reshape(tokens.shape[0], f, h, w, -1).transpose(0, 4, 1, 2, 3)


@dataclass(frozen=True)
class LatentState:
    latent: mx.array  # (B, N, C) current noisy tokens
    clean: mx.array  # (B, N, C) values restored where denoise_mask == 0
    denoise_mask: mx.array  # (B, N, 1): 1 generate, 0 preserve
    positions: mx.array
    keyframes_mask: mx.array | None = None  # (B, N, 1): tokens encoding one pixel frame
    attention_mask: mx.array | None = None
    frozen: bool = False

    @property
    def uniform(self) -> bool:
        return bool(mx.all(self.denoise_mask == 1.0).item())


def _normal(shape: tuple[int, ...], seed: int, bf16_noise: bool) -> mx.array:
    mx.random.seed(seed)
    n = mx.random.normal(shape)
    # The baseline runner rounds its noise to bf16; reproducing that gives the
    # same starting point for same-seed comparisons against it.
    return n.astype(mx.bfloat16).astype(G) if bf16_noise else n


def noised_state(
    shape: tuple[int, int, int],
    positions: mx.array,
    seed: int,
    sigma: float = 1.0,
    initial: mx.array | None = None,
    tokens_per_frame: int | None = None,
    bf16_noise: bool = False,
) -> LatentState:
    """noise * sigma + clean * (1 - sigma) on every token; conditioning is applied
    by the caller."""
    clean = mx.zeros(shape, dtype=G) if initial is None else initial.astype(G)
    noise = _normal(shape, seed, bf16_noise)
    kf = None
    if (
        tokens_per_frame is not None
    ):  # video: the causal first latent frame is one pixel frame
        b, n, _ = shape
        kf = (mx.arange(n) < tokens_per_frame).astype(G).reshape(1, n, 1)
        kf = mx.broadcast_to(kf, (b, n, 1))
    return LatentState(
        latent=noise * sigma + clean * (1.0 - sigma),
        clean=clean,
        denoise_mask=mx.ones((shape[0], shape[1], 1), dtype=G),
        positions=positions,
        keyframes_mask=kf,
    )


def blend(x0: mx.array, state: LatentState) -> mx.array:
    return x0 * state.denoise_mask + state.clean * (1.0 - state.denoise_mask)


# A denoiser maps (video_state, audio_state, video_x, audio_x, sigma)
# -> (video_x0, audio_x0).
Denoiser = Callable[
    [LatentState, LatentState, mx.array, mx.array, float], tuple[mx.array, mx.array]
]


def euler_loop(
    denoise: Denoiser,
    video: LatentState,
    audio: LatentState,
    sigmas: list[float],
    on_step: Callable[[int, float], None] | None = None,
    step_cache: Any = None,
) -> tuple[mx.array, mx.array]:
    """`step_cache` (dit.StepCache) is handed to the denoiser each step; its
    last step is always computed."""
    vx, ax = video.latent, audio.latent
    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        if step_cache is not None:
            step_cache.force = s_next == 0
            v0, a0 = denoise(video, audio, vx, ax, s, step_cache=step_cache, step=i)
        else:
            v0, a0 = denoise(video, audio, vx, ax, s, step=i)
        v0, a0 = blend(v0, video), blend(a0, audio)
        dt = s_next - s
        vx = vx + (vx - v0) / s * dt
        ax = ax + (ax - a0) / s * dt
        mx.async_eval(vx, ax)
        if on_step:
            on_step(i, s)
    return vx, ax


def euler_loop_audio(
    denoise,
    audio: LatentState,
    sigmas: list[float],
    on_step: Callable[[int, float], None] | None = None,
) -> mx.array:
    """`euler_loop` for an audio-only state (T2A): the denoiser is called with
    no video and returns (None, audio x0)."""
    ax = audio.latent
    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        _, a0 = denoise(None, audio, None, ax, s, step=i)
        a0 = blend(a0, audio)
        ax = ax + (ax - a0) / s * (s_next - s)
        mx.async_eval(ax)
        if on_step:
            on_step(i, s)
    return ax


def euler_loop_video(
    denoise,
    video: LatentState,
    sigmas: list[float],
    on_step: Callable[[int, float], None] | None = None,
    step_cache: Any = None,
) -> mx.array:
    """`euler_loop` for a video-only state (alpha-gen): the denoiser is called
    with no audio and returns (video x0, None)."""
    vx = video.latent
    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        if step_cache is not None:
            step_cache.force = s_next == 0
            v0, _ = denoise(video, None, vx, None, s, step_cache=step_cache, step=i)
        else:
            v0, _ = denoise(video, None, vx, None, s, step=i)
        v0 = blend(v0, video)
        vx = vx + (vx - v0) / s * (s_next - s)
        mx.async_eval(vx)
        if on_step:
            on_step(i, s)
    return vx


def euler_ancestral_loop(
    denoise: Denoiser,
    video: LatentState,
    audio: LatentState,
    sigmas: list[float],
    noise_seed: int,
    eta: float = 1.0,
    s_noise: float = 1.0,
    on_step: Callable[[int, float], None] | None = None,
    step_cache: Any = None,
) -> tuple[mx.array, mx.array]:
    """Rectified-flow ancestral Euler (alpha = 1 - sigma), one generator, video
    noise drawn first. `step_cache` as in `euler_loop`. As upstream's
    _ancestral_euler_denoising_loop: x0 is blended with the clean latent on
    the denoise mask before the step and the stepped latent again after it."""
    vx, ax = video.latent, audio.latent
    mx.random.seed(noise_seed)

    def step(
        x: mx.array, x0: mx.array, s: float, s_next: float, noise: mx.array | None
    ) -> mx.array:
        sigma_down = s_next * (1.0 + (s_next / s - 1.0) * eta)
        r = sigma_down / s
        nxt = r * x + (1.0 - r) * x0
        if eta > 0:
            a_next, a_down = 1.0 - s_next, 1.0 - sigma_down
            coeff = max(s_next**2 - sigma_down**2 * a_next**2 / a_down**2, 0.0) ** 0.5
            nxt = (a_next / a_down) * nxt + noise * (s_noise * coeff)
        return nxt

    for i, (s, s_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        if step_cache is not None:
            step_cache.force = s_next == 0
            v0, a0 = denoise(video, audio, vx, ax, s, step_cache=step_cache, step=i)
        else:
            v0, a0 = denoise(video, audio, vx, ax, s, step=i)
        v0, a0 = blend(v0, video), blend(a0, audio)
        if s_next == 0:
            vx, ax = v0, a0
        else:
            vn = mx.random.normal(vx.shape) if eta > 0 else None
            an = mx.random.normal(ax.shape) if eta > 0 else None
            vx, ax = step(vx, v0, s, s_next, vn), step(ax, a0, s, s_next, an)
            if eta > 0:
                vx, ax = blend(vx, video), blend(ax, audio)
        mx.async_eval(vx, ax)
        if on_step:
            on_step(i, s)
        if s_next == 0:
            break
    return vx, ax


def with_latent(state: LatentState, latent: mx.array) -> LatentState:
    return replace(state, latent=latent)


def video_time_bounds(f: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """[start, end) seconds of each latent frame (causal first frame), the
    temporal row of upstream get_pixel_coords(causal_fix=True) / fps."""
    idx = np.arange(f, dtype=np.float64)
    starts = np.maximum(idx * 8 + 1 - 8, 0.0) / fps
    ends = np.maximum((idx + 1) * 8 + 1 - 8, 0.0) / fps
    return starts, ends


def audio_time_bounds(tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """[start, end) seconds of each audio latent (upstream AudioPatchifier
    _compute_audio_timings, causal)."""
    idx = np.arange(tokens, dtype=np.float64)
    starts = np.maximum(idx * 4 + 1 - 4, 0.0) * 160 / 16000
    ends = np.maximum((idx + 1) * 4 + 1 - 4, 0.0) * 160 / 16000
    return starts, ends


def region_mask(
    state: LatentState,
    start_time: float,
    end_time: float,
    fps: float,
    tokens_per_frame: int | None,
) -> LatentState:
    """Upstream TemporalRegionMask: denoise mask 1 for the tokens whose time
    span overlaps [start_time, end_time) (end > start_time and start <
    end_time), 0 elsewhere, over the whole state (it is the only conditioning
    of the retake); `tokens_per_frame` None means an audio state. The noisy
    latent is re-blended as the noiser does (lerp(clean, noised, mask)) so the
    kept region is the source."""
    n = state.latent.shape[1]
    if tokens_per_frame is None:
        starts, ends = audio_time_bounds(n)
    else:
        f = n // tokens_per_frame
        starts, ends = video_time_bounds(f, fps)
        starts, ends = (
            np.repeat(starts, tokens_per_frame),
            np.repeat(ends, tokens_per_frame),
        )
    inside = (ends > start_time) & (starts < end_time)
    mask = mx.array(inside.astype(np.float32)).reshape(1, n, 1)
    mask = mx.broadcast_to(mask, (state.latent.shape[0], n, 1))
    return replace(
        state,
        denoise_mask=mask,
        latent=state.clean * (1.0 - mask) + state.latent * mask,
    )


# ---- stills (upstream ImageConditioningInput) --------------------------------
KEYFRAME_NOISE_SEED_OFFSET = 50000  # 40000 is the hq pipeline's res_2s stream


@dataclass(frozen=True)
class Still:
    """One conditioning still, upstream's `--image PATH FRAME_IDX STRENGTH
    [CRF]`: `image` is a path or encoded bytes, `frame` the target pixel frame,
    `crf` None means the checkpoint's H.264 round trip (18), 0 none."""

    image: Any
    frame: int = 0
    strength: float = 1.0
    crf: int | None = None


def rebase_stills(
    conds: list[tuple[Any, int, float]],
    scale: int,
    start: int = 0,
    end: int | None = None,
    resume: int | None = None,
) -> list[tuple[Any, int, float]]:
    """Upstream dfr_helpers/ops.py rebase_image_conditionings: after r temporal
    rounds a still's moment sits at frame * 2**r; with `end` only stills inside
    [start, end] are kept, their frames made window-local; `resume` (the
    epilogue's filter) additionally drops stills before the window's resume
    pixel."""
    out = []
    for latent, frame, strength in conds:
        scaled = frame * scale
        if end is not None and not start <= scaled <= end:
            continue
        if resume is not None and scaled < resume:
            continue
        out.append((latent, scaled - start, strength))
    return out


def condition_stills(
    state: LatentState,
    conds: list[tuple[mx.array, int, float]],
    fps: float,
    sigma: float,
    seed: int,
    append_all: bool = False,
) -> LatentState:
    """Upstream combined_image_conditionings: each (latent (1, C, 1, h, w),
    pixel frame, strength) at frame 0 replaces latent frame 0 (`VideoCondition
    ByLatentIndex`); any other frame is appended as a clean single-frame
    keyframe token block (`VideoConditionByKeyframeIndex`), in list order.
    `append_all` is image_conditionings_by_adding_guiding_latent (the keyframe
    interpolation pipeline): frame 0 is appended too. Apply before generated
    slots and reference tokens."""
    for i, (latent, frame, strength) in enumerate(conds):
        if frame == 0 and not append_all:
            state = condition_latent_frame(state, latent, strength)
            continue
        _, _, _, h, w = latent.shape
        state = append_anchor_keyframes(
            state,
            latent,
            [frame],
            h,
            w,
            fps,
            sigma,
            seed + i,
            strength=strength,
            seed_offset=KEYFRAME_NOISE_SEED_OFFSET,
        )
    return state


def condition_latent_frame(
    state: LatentState,
    latent: mx.array,
    strength: float = 1.0,
    latent_idx: int = 0,
) -> LatentState:
    """Upstream VideoConditionByLatentIndex (ltx_core latent_cond.py
    _apply_condition_by_latent_index): the patchified `latent` (B, C, F', H, W)
    replaces `clean` on the token span of latent frames [latent_idx, latent_idx
    + F') and denoise_mask there becomes 1 - strength. The current latent on
    the span is re-blended (the noiser's lerp(clean, noised, denoise_mask)), so
    a strength-1 frame starts and stays clean. Apply to the plain video state,
    before any slots or reference tokens are appended."""
    tokens = patchify(latent).astype(G)
    b, count, c = tokens.shape
    _, _, _, h, w = latent.shape
    start = latent_idx * h * w
    stop = start + count
    if stop > state.latent.shape[1]:
        raise ValueError(
            f"conditioning span [{start}, {stop}) exceeds "
            f"{state.latent.shape[1]} tokens"
        )
    tokens = mx.broadcast_to(tokens, (state.latent.shape[0], count, c))
    clean = mx.concatenate(
        [state.clean[:, :start], tokens, state.clean[:, stop:]], axis=1
    )
    mask = mx.concatenate(
        [
            state.denoise_mask[:, :start],
            mx.full((state.latent.shape[0], count, 1), 1.0 - strength, dtype=G),
            state.denoise_mask[:, stop:],
        ],
        axis=1,
    )
    # only the span is re-blended (upstream rewrites those tokens' noisy latent
    # as lerp(clean, noised, mask)); other tokens keep their own blend
    span = state.latent[:, start:stop] * (1.0 - strength) + tokens * strength
    latent = mx.concatenate(
        [state.latent[:, :start], span, state.latent[:, stop:]], axis=1
    )
    return replace(state, latent=latent, clean=clean, denoise_mask=mask)


# ---- dev (guided) ---------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "has_subtitles, has_blurbox, transition from black, transition to black, "
    "speech_ending_short, "
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out "
    "colors, excessive noise, grainy texture, poor lighting, flickering, motion "
    "blur, distorted proportions, unnatural skin tones, deformed facial features, "
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera "
    "shake, incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, color "
    "banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted "
    "voice, robotic voice, echo, background noise, off-sync audio, incorrect "
    "dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts."
)


def ltx2_schedule(
    steps: int,
    num_tokens: int,
    base_shift: float = 0.95,
    max_shift: float = 2.05,
    terminal: float = 0.1,
) -> list[float]:
    """Token-count-shifted schedule (LinearQuadratic family), stretched so the last
    non-zero sigma is `terminal`."""
    import math

    import numpy as np

    sigmas = np.linspace(1.0, 0.0, steps + 1)
    slope = (max_shift - base_shift) / (4096 - 1024)
    shift = num_tokens * slope + base_shift - slope * 1024
    nz = sigmas != 0
    sigmas[nz] = math.exp(shift) / (math.exp(shift) + (1.0 / sigmas[nz] - 1.0))
    one_minus = 1.0 - sigmas[nz]
    scale = one_minus[-1] / (1.0 - terminal)
    if scale != 0:
        sigmas[nz] = 1.0 - one_minus / scale
    return sigmas.tolist()


@dataclass(frozen=True)
class Guidance:
    cfg: float = 3.0
    stg: float = 1.0
    modality: float = 3.0
    rescale: float = 0.7
    # Upstream resolves params by checkpoint version: a 2.5.0 checkpoint gets
    # LTX_2_4_PARAMS (constants.py _PARAMS_SINCE_VERSION), whose stg_blocks is
    # [28]. The [29] in the PipelineParams dataclass is the LTX-2.0 default.
    stg_blocks: tuple[int, ...] = (28,)
    # upstream MultiModalGuiderParams.skip_step: with N > 0 only steps with
    # index % (N + 1) == 0 run this modality; a skipped step keeps its last x0
    skip_step: int = 0

    def skips(self, step: int | None) -> bool:
        """upstream MultiModalGuider.should_skip_step"""
        if not self.skip_step or step is None:
            return False
        return step % (self.skip_step + 1) != 0

    def combine(
        self,
        cond: mx.array,
        uncond: mx.array | None,
        perturbed: mx.array | None,
        isolated: mx.array | None,
    ) -> mx.array:
        """A pass that is None was not run (its scale is neutral)."""
        pred = cond
        if uncond is not None:
            pred = pred + (self.cfg - 1) * (cond - uncond)
        if perturbed is not None:
            pred = pred + self.stg * (cond - perturbed)
        if isolated is not None:
            pred = pred + (self.modality - 1) * (cond - isolated)
        if self.rescale:
            factor = mx.sqrt(mx.var(cond)) / (mx.sqrt(mx.var(pred)) + 1e-8)
            pred = pred * (self.rescale * factor + (1 - self.rescale))
        return pred


# ---- DFR: canvas, generated keyframe slots, reference tokens ---------------
SLOT_NOISE_SEED_OFFSET = 20000
SEGMENT_CANDIDATES = (24, 32)


def dfr_canvas(num_frames: int) -> tuple[int, int, list[int]]:
    """(canvas frames, segment, slot pixel frames): the clip padded to whole
    keyframe segments (24 or 32 frames, least padding, longer on ties) with
    one generated keyframe slot at every segment boundary."""
    if num_frames < 9 or (num_frames - 1) % 8:
        raise ValueError(f"num_frames must be 8k + 1 with k >= 1, got {num_frames}")
    content = num_frames - 1
    segment = min(SEGMENT_CANDIDATES, key=lambda s: ((-content) % s, -s))
    padded = content + (-content) % segment
    return padded + 1, segment, [segment * i for i in range(1, padded // segment + 1)]


def conditioning_fps(fps: float) -> float:
    return 60.0 if fps > 30.0 else fps


def append_slots(
    state: LatentState,
    pixel_frames: list[int],
    h: int,
    w: int,
    fps: float,
    initial: mx.array | None,
    sigma: float,
    seed: int,
) -> tuple[LatentState, slice]:
    """Append one single-frame keyframe slot per pixel frame (generated tokens,
    marked in keyframes_mask, noised from their own seed). Returns the slot token
    slice."""
    b, n, c = state.latent.shape
    per = h * w
    y = mx.arange(h).astype(G) * 32 + 16.0
    x = mx.arange(w).astype(G) * 32 + 16.0
    pos = []
    for (
        frame
    ) in pixel_frames:  # a single pixel frame: temporal midpoint (frame + 0.5) / fps
        t = mx.full((h, w), (frame + 0.5) / fps, dtype=G)
        pos.append(
            mx.stack(
                [
                    t,
                    mx.broadcast_to(y[:, None], (h, w)),
                    mx.broadcast_to(x[None, :], (h, w)),
                ],
                axis=-1,
            ).reshape(1, per, 3)
        )
    pos = mx.broadcast_to(mx.concatenate(pos, axis=1), (b, per * len(pixel_frames), 3))
    count = per * len(pixel_frames)
    slot = (
        mx.zeros((b, count, c), dtype=G)
        if initial is None
        else mx.concatenate(
            [patchify(initial[:, :, k : k + 1]) for k in range(initial.shape[2])],
            axis=1,
        ).astype(G)
    )
    mx.random.seed(seed + SLOT_NOISE_SEED_OFFSET)
    noise = mx.random.normal(slot.shape)
    noised = noise * sigma + slot * (1.0 - sigma)
    kf = (
        state.keyframes_mask
        if state.keyframes_mask is not None
        else mx.zeros((b, n, 1), dtype=G)
    )
    return replace(
        state,
        latent=mx.concatenate([state.latent, noised], axis=1),
        clean=mx.concatenate([state.clean, mx.zeros_like(slot)], axis=1),
        denoise_mask=mx.concatenate(
            [state.denoise_mask, mx.ones((b, count, 1), dtype=G)], axis=1
        ),
        positions=mx.concatenate([state.positions, pos], axis=1),
        keyframes_mask=mx.concatenate([kf, mx.ones((b, count, 1), dtype=G)], axis=1),
    ), slice(n, n + count)


def append_reference(
    state: LatentState,
    tokens: mx.array,
    positions: mx.array,
    downscale: int,
    strength: float = 1.0,
    temporal_scale: int = 1,
    fps: float | None = None,
    attention: float | np.ndarray | None = None,
    num_noisy: int | None = None,
) -> LatentState:
    """Upstream VideoConditionByReferenceLatent: append clean IC-LoRA reference
    tokens (denoise mask 1 - strength) with their spatial positions scaled to
    the target's pixel grid (`downscale`) and, for a reference at 1 / S of
    the target's frame rate (`temporal_scale` S), their times spread by S and
    shifted so the last reference patch ends with the target's last
    (t - (S - 1) / fps, clamped at 0). `attention` (upstream
    ConditioningItemAttentionStrengthWrapper) is a scalar in [0, 1] or a
    per-token weight (count,) controlling how strongly these tokens and the
    target's `num_noisy` tokens attend to each other: the state's attention
    mask becomes the log-space bias of upstream build_attention_mask."""
    b, n, _ = state.latent.shape
    count = tokens.shape[1]
    tokens = tokens.astype(G)
    pos = positions * mx.array([1.0, float(downscale), float(downscale)])
    if temporal_scale != 1:
        if fps is None:
            raise ValueError("temporal_scale needs the target fps")
        t = pos[..., 0:1] * float(temporal_scale) - (temporal_scale - 1) / fps
        pos = mx.concatenate([mx.maximum(t, 0.0), pos[..., 1:]], axis=-1)
    kf = state.keyframes_mask
    mask = state.attention_mask
    if attention is not None or mask is not None:
        mask = attention_bias(
            mask, n, count, num_noisy if num_noisy is not None else n, attention
        )
    return replace(
        state,
        latent=mx.concatenate([state.latent, tokens], axis=1),
        clean=mx.concatenate([state.clean, tokens], axis=1),
        denoise_mask=mx.concatenate(
            [state.denoise_mask, mx.full((b, count, 1), 1.0 - strength, dtype=G)],
            axis=1,
        ),
        positions=mx.concatenate(
            [state.positions, mx.broadcast_to(pos, (b, count, 3))], axis=1
        ),
        keyframes_mask=None
        if kf is None
        else mx.concatenate([kf, mx.zeros((b, count, 1), dtype=G)], axis=1),
        attention_mask=mask,
    )


AUDIO_REFERENCE_GAP = 0.04  # upstream patchify_audio_reference_latent


def append_audio_reference(
    state: LatentState, tokens: mx.array, strength: float = 1.0
) -> LatentState:
    """Upstream AudioConditionByReferenceLatent with patchify_audio_reference_
    latent(negative_positions=True): clean reference audio tokens appended
    after the target's, their times shifted before 0 by the reference's own
    duration plus 0.04 s (so they never overlap the target's timeline)."""
    b, n, _ = state.latent.shape
    count = tokens.shape[1]
    tokens = tokens.astype(G)
    starts, ends = audio_time_bounds(count)
    pos = mx.array(
        ((starts + ends) / 2.0 - ends.max() - AUDIO_REFERENCE_GAP).astype(np.float32)
    )
    pos = mx.broadcast_to(pos.reshape(1, count, 1), (b, count, 1))
    return replace(
        state,
        latent=mx.concatenate([state.latent, tokens], axis=1),
        clean=mx.concatenate([state.clean, tokens], axis=1),
        denoise_mask=mx.concatenate(
            [state.denoise_mask, mx.full((b, count, 1), 1.0 - strength, dtype=G)],
            axis=1,
        ),
        positions=mx.concatenate([state.positions, pos], axis=1),
    )


def attention_bias(
    existing: mx.array | None,
    n_existing: int,
    m_new: int,
    n_noisy: int,
    cross: float | np.ndarray | None,
) -> mx.array:
    """Upstream mask_utils.build_attention_mask followed by the transformer's
    _prepare_self_attention_mask: the (1, N + M, N + M) self-attention bias
    for a conditioning block of M tokens appended to N. Blocks: existing x
    existing kept (or full attention), new x new 1, noisy x new and new x
    noisy the per-token weight `cross` (None: 1), other prior conditioning
    tokens x new 0. Weights w become log(w) (w <= 0: -inf), fp16."""
    total = n_existing + m_new
    weights = np.ones((total, total), dtype=np.float32)
    if existing is not None:
        weights[:n_existing, :n_existing] = np.exp(
            np.array(existing[0], dtype=np.float32)
        )
    c = (
        np.ones(m_new, dtype=np.float32)
        if cross is None
        else np.broadcast_to(np.asarray(cross, dtype=np.float32), (m_new,))
    )
    weights[:, n_existing:] = 0.0
    weights[n_existing:, :] = 0.0
    weights[n_existing:, n_existing:] = 1.0
    weights[:n_noisy, n_existing:] = c[None, :]
    weights[n_existing:, :n_noisy] = c[:, None]
    with np.errstate(divide="ignore"):
        bias = np.where(weights > 0, np.log(np.maximum(weights, 1e-30)), -np.inf)
    return mx.array(bias.astype(np.float16))[None]


def mask_video_to_tokens(mask: np.ndarray, f: int, h: int, w: int) -> np.ndarray:
    """Upstream downsample_mask_video_to_latent: a pixel mask (F, H, W) in
    [0, 1] -> per-token weights (f h w) for an (f, h, w) latent: area
    downsampling to (h, w) per frame, the first frame kept, the rest averaged
    over each latent frame's (F - 1) / (f - 1) pixel frames."""
    F = mask.shape[0]
    sh, sw = mask.shape[1] // h, mask.shape[2] // w
    if mask.shape[1] != sh * h or mask.shape[2] != sw * w:
        raise ValueError("the mask video's size must be a multiple of the latent grid")
    spatial = mask.reshape(F, h, sh, w, sw).mean(axis=(2, 4))
    if F > 1 and f > 1:
        if (F - 1) % (f - 1):
            raise ValueError(f"mask frames {F} do not fit {f} latent frames")
        t = (F - 1) // (f - 1)
        rest = spatial[1:].reshape(f - 1, t, h, w).mean(axis=1)
        out = np.concatenate([spatial[:1], rest], axis=0)
    else:
        out = spatial[:1]
    return out.reshape(-1).astype(np.float32)


def slots_to_latent(
    tokens: mx.array, slots: slice, count: int, h: int, w: int
) -> mx.array:
    """Slot tokens -> (B, C, K, H, W)."""
    return unpatchify(tokens[:, slots], (count, h, w))


# ---- DFR temporal rounds (upstream dfr_helpers/layout.py, ops.py) ------------
ANCHOR_KEYFRAME_STRENGTH = 0.95
TEMPORAL_ANCESTRAL_ETA = 0.5
ANCHOR_NOISE_SEED_OFFSET = 30000


@dataclass(frozen=True)
class TemporalTile:
    """One temporal-round window: latent cells [start, end), its pixel span on
    the canvas, the keyframe seams inside it (anchors) and the segment
    midpoints it generates (slots)."""

    start: int
    end: int
    pixel_start: int
    pixel_end: int
    anchors: tuple[int, ...]
    slots: tuple[int, ...]


def split_at_seams(boundaries: list[int], num_tiles: int) -> list[tuple[int, int]]:
    """ltx_core.tiling.split_at_seams with no overlap: the K segments between
    consecutive boundary cells are dealt largest-first over min(num_tiles, K)
    tiles; each tile is [first boundary, last boundary + 1)."""
    k = len(boundaries) - 1
    n = min(num_tiles, k)
    base, leftover = divmod(k, n)
    counts = [base + (1 if i < leftover else 0) for i in range(n)]
    out, cursor = [], 0
    for count in counts:
        start = boundaries[cursor]
        cursor += count
        out.append((0 if not out else start + 1, boundaries[cursor] + 1))
    return out


def temporal_tile_plan(
    seam_positions: list[int], num_frames: int, num_tiles: int
) -> list[TemporalTile]:
    """Upstream TemporalTilePlan: cut the canvas on its keyframe seams into
    num_tiles windows (remainder segments to the leading ones); slots are the
    canvas segments' midpoints, handed to the window that contains them."""
    t = VIDEO_TEMPORAL_SCALE
    seams = [0, *(p // t for p in seam_positions)]
    latent_len = (num_frames - 1) // t + 1
    if seams[-1] != latent_len - 1:
        raise ValueError("the last keyframe seam must be the last canvas cell")
    edges = [0, *seam_positions]
    slots = [(a + b) // 2 for a, b in zip(edges[:-1], edges[1:])]
    tiles = []
    for start, end in split_at_seams(seams, num_tiles):
        ps, pe = start * t, (end - 1) * t
        tiles.append(
            TemporalTile(
                start,
                end,
                ps,
                pe,
                tuple(p for p in seam_positions if ps <= p <= pe),
                tuple(p for p in slots if ps <= p <= pe),
            )
        )
    return tiles


@dataclass(frozen=True)
class TilePrefix:
    """A non-first tile starts on the last keyframe plane before its seam
    (cell 0), carries the previous tile's cells up to the seam pinned, and
    resumes denoising on the pixel after the seam (upstream TilePrefix)."""

    keyframe_position: int
    video_start_cell: int
    cells: int
    resume_pixel: int


def tile_prefix(seam_pixel: int, plane_positions, scale: int = VIDEO_TEMPORAL_SCALE):
    before = [p for p in plane_positions if p < seam_pixel]
    if not before:
        raise RuntimeError(f"no keyframe plane before seam {seam_pixel}")
    keyframe = max(before)
    if keyframe % scale or seam_pixel % scale:
        raise RuntimeError("keyframe and seam must sit on the latent border")
    return TilePrefix(
        keyframe,
        keyframe // scale + 1,
        1 + (seam_pixel - keyframe) // scale,
        seam_pixel + 1,
    )


def resample_audio_tokens(
    tokens: mx.array, src_start: float, src_end: float, out_frames: int
) -> mx.array:
    """Linear resampling of (1, T, C) audio tokens over [src_start, src_end)
    latent cells into out_frames tokens (upstream resample_audio_time)."""
    full_t = tokens.shape[1]
    step = (src_end - src_start) / out_frames
    pos = mx.clip(
        src_start + step * mx.arange(out_frames).astype(G), 0.0, float(full_t - 1)
    )
    lo = mx.floor(pos).astype(mx.int32)
    hi = mx.minimum(lo + 1, full_t - 1)
    w = (pos - lo.astype(G))[None, :, None]
    return tokens[:, lo] * (1.0 - w) + tokens[:, hi] * w


def audio_tokens_for_tile(
    tokens: mx.array,
    pixel_start: int,
    local_frames: int,
    playback_fps: float,
    source_duration: float,
    cond_fps: float,
) -> mx.array:
    """Stage-1 audio for one tile: its wall-clock window of the source audio,
    resampled to the token count a clip of local_frames at cond_fps gets
    (upstream audio_latent_for_tile)."""
    full_t = tokens.shape[1]
    src_start = pixel_start / playback_fps / source_duration * full_t
    src_end = (pixel_start + local_frames) / playback_fps / source_duration * full_t
    return resample_audio_tokens(
        tokens, src_start, src_end, audio_token_count(local_frames, cond_fps)
    )


def append_anchor_keyframes(
    state: LatentState,
    planes: mx.array,
    pixel_frames: list[int],
    h: int,
    w: int,
    fps: float,
    sigma: float,
    seed: int,
    strength: float = ANCHOR_KEYFRAME_STRENGTH,
    seed_offset: int = ANCHOR_NOISE_SEED_OFFSET,
) -> LatentState:
    """Upstream VideoConditionByKeyframeIndex for each plane in `planes`
    ((B, C, K, H, W)): given keyframe content appended as single-frame tokens
    at pixel_frames (not marked as generated slots) with denoise mask
    1 - strength; the noisy tokens start at lerp(clean, sigma * noise, mask).
    A frame-0 keyframe gets upstream's causal fix ([0, 1) -> the same single
    pixel frame), so every index uses the (frame + 0.5) / fps midpoint."""
    b, n, c = state.latent.shape
    per = h * w
    y = mx.arange(h).astype(G) * 32 + 16.0
    x = mx.arange(w).astype(G) * 32 + 16.0
    pos = []
    for frame in pixel_frames:
        t = mx.full((h, w), (frame + 0.5) / fps, dtype=G)
        pos.append(
            mx.stack(
                [
                    t,
                    mx.broadcast_to(y[:, None], (h, w)),
                    mx.broadcast_to(x[None, :], (h, w)),
                ],
                axis=-1,
            ).reshape(1, per, 3)
        )
    count = per * len(pixel_frames)
    pos = mx.broadcast_to(mx.concatenate(pos, axis=1), (b, count, 3))
    clean = mx.concatenate(
        [patchify(planes[:, :, k : k + 1]) for k in range(planes.shape[2])], axis=1
    ).astype(G)
    mx.random.seed(seed + seed_offset)
    noise = mx.random.normal(clean.shape)
    mask = 1.0 - strength
    kf = state.keyframes_mask
    return replace(
        state,
        latent=mx.concatenate(
            [state.latent, clean * (1.0 - mask) + noise * (sigma * mask)], axis=1
        ),
        clean=mx.concatenate([state.clean, clean], axis=1),
        denoise_mask=mx.concatenate(
            [state.denoise_mask, mx.full((b, count, 1), mask, dtype=G)], axis=1
        ),
        positions=mx.concatenate([state.positions, pos], axis=1),
        keyframes_mask=None
        if kf is None
        else mx.concatenate([kf, mx.zeros((b, count, 1), dtype=G)], axis=1),
    )


# ---- DFR spatial epilogue: tiled denoising (upstream ltx_core/tiling.py) --------
EPILOGUE_SPATIAL_OVERLAP = 10
EPILOGUE_SPATIAL_COARSE_TILES = 2
EPILOGUE_SPATIAL_TILES = 4
EPILOGUE_KEYFRAME_STRENGTH = 1.0
EPILOGUE_NOISE_SEED_OFFSET = 2000
EPILOGUE_KEYFRAME_DECODE_SEED_OFFSET = 4000


def split_by_count(
    dim: int, num_tiles: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """ltx_core.tiling.split_by_count after clamp_dim_tiling: (start, end,
    left_ramp, right_ramp) per tile. Fewer cells than tiles -> one tile; the
    overlap is clamped to dim - num_tiles."""
    if num_tiles <= 1 or dim < num_tiles:
        return [(0, dim, 0, 0)]
    overlap = min(overlap, dim - num_tiles)
    total = dim + overlap * (num_tiles - 1)
    size = total // num_tiles
    remainder = total % num_tiles
    base_dim = dim - remainder
    if base_dim <= size:
        base = [(0, base_dim, 0, 0)]
    else:
        amount = (base_dim + size - 2 * overlap - 1) // (size - overlap)
        base = [(0, size, 0, overlap)]
        base += [
            (i * (size - overlap), i * (size - overlap) + size, overlap, overlap)
            for i in range(1, amount - 1)
        ]
        base.append(((amount - 1) * (size - overlap), base_dim, overlap, 0))
    out = []
    for i, (start, end, left, right) in enumerate(base):
        shift = min(i, remainder)
        grow = 1 if i < remainder else 0
        out.append((start + shift, end + shift + grow, left, right))
    return out


def trapezoid_mask(length: int, ramp_left: int, ramp_right: int) -> np.ndarray:
    """ltx_core compute_trapezoidal_mask_1d: linear fade-in over ramp_left cells
    (not starting from 0) and fade-out over ramp_right."""
    ramp_left, ramp_right = (
        max(0, min(ramp_left, length)),
        max(0, min(ramp_right, length)),
    )
    mask = np.ones(length, dtype=np.float32)
    if ramp_left:
        mask[:ramp_left] *= np.linspace(0.0, 1.0, ramp_left + 2, dtype=np.float32)[1:-1]
    if ramp_right:
        mask[-ramp_right:] *= np.linspace(1.0, 0.0, ramp_right + 2, dtype=np.float32)[
            1:-1
        ]
    return np.clip(mask, 0.0, 1.0)


@dataclass(frozen=True)
class SpatialTile:
    h0: int
    h1: int
    w0: int
    w1: int
    weights: np.ndarray  # (h1 - h0, w1 - w0) trapezoid blend


def split_by_size_pinned(
    dim: int, size: int, min_overlap: int
) -> list[tuple[int, int, int, int]]:
    """ltx_core.tiling.split_by_size_pinned: equal `size` tiles with the first
    and last origins pinned to the extent, the count grown until the realized
    overlap reaches `min_overlap` (capped so three tiles never share a cell),
    interior origins spread evenly (remainder to the leading gaps)."""
    if dim <= size:
        return [(0, dim, 0, 0)]
    limit = max(2, 1 + (dim - size) // ((size + 1) // 2))
    count = 2
    while count < limit and (count * size - dim) / (count - 1) < min_overlap:
        count += 1
    base, remainder = divmod(dim - size, count - 1)
    origins = [0]
    for index in range(count - 1):
        origins.append(origins[-1] + base + (1 if index < remainder else 0))
    out = []
    for index, start in enumerate(origins):
        left = 0 if index == 0 else origins[index - 1] + size - start
        right = 0 if index == count - 1 else start + size - origins[index + 1]
        out.append((start, start + size, left, right))
    return out


def spatial_tiles_by_size(
    h: int, w: int, tile_h: int, tile_w: int, overlap_h: int, overlap_w: int
) -> list[SpatialTile]:
    """Upstream FixedSizeSpatialTiling (the IC-LoRA --tile stages): pinned
    equal-size tiles per axis with trapezoid blends, in latent cells."""
    tiles = []
    for h0, h1, hl, hr in split_by_size_pinned(h, tile_h, overlap_h):
        mh = trapezoid_mask(h1 - h0, hl, hr)
        for w0, w1, wl, wr in split_by_size_pinned(w, tile_w, overlap_w):
            mw = trapezoid_mask(w1 - w0, wl, wr)
            tiles.append(SpatialTile(h0, h1, w0, w1, mh[:, None] * mw[None, :]))
    return tiles


def spatial_tiles(h: int, w: int, num_tiles: int, overlap: int) -> list[SpatialTile]:
    """num_tiles x num_tiles spatial tiles over an (h, w) latent grid with
    separable trapezoid blend weights; every frame of the window is in every
    tile (upstream TileCountConfig frames=1)."""
    tiles = []
    for h0, h1, hl, hr in split_by_count(h, num_tiles, overlap):
        mh = trapezoid_mask(h1 - h0, hl, hr)
        for w0, w1, wl, wr in split_by_count(w, num_tiles, overlap):
            mw = trapezoid_mask(w1 - w0, wl, wr)
            tiles.append(SpatialTile(h0, h1, w0, w1, mh[:, None] * mw[None, :]))
    return tiles


# ---- res_2s (the HQ pipeline; upstream utils/res2s.py, samplers.py, ----------
# ---- diffusion_steps.Res2sDiffusionStep) --------------------------------------
RES2S_SUBSTEP_SEED_OFFSET = 10000
RES2S_TERMINAL_SIGMA = 0.0011


def _phi(j: int, z: float) -> float:
    """phi_j(z) = (e^z - sum_{k<j} z^k / k!) / z^j, the exponential-integrator
    functions; 1 / j! at z = 0."""
    import math

    if abs(z) < 1e-10:
        return 1.0 / math.factorial(j)
    remainder = sum(z**k / math.factorial(k) for k in range(j))
    return (math.exp(z) - remainder) / (z**j)


def res2s_coefficients(h: float, c2: float = 0.5) -> tuple[float, float, float]:
    """(a21, b1, b2) for the two-stage exponential Runge-Kutta step of size h in
    log-sigma: a21 = c2 phi_1(-h c2), b2 = phi_2(-h) / c2, b1 = phi_1(-h) - b2."""
    a21 = c2 * _phi(1, -h * c2)
    b2 = _phi(2, -h) / c2
    return a21, _phi(1, -h) - b2, b2


def _res2s_noise(shape: tuple[int, ...], key: mx.array) -> mx.array:
    """upstream _get_new_noise: standard normal, standardized globally, then
    per batch row over (tokens, channels)."""
    n = mx.random.normal(shape, key=key)
    n = (n - mx.mean(n)) / mx.std(n)
    return (n - mx.mean(n, axis=(1, 2), keepdims=True)) / mx.std(
        n, axis=(1, 2), keepdims=True
    )


def _res2s_inject(
    state: LatentState,
    sample: mx.array,
    denoised: mx.array,
    sigma: float,
    sigma_next: float,
    eta: float,
    key: mx.array,
) -> mx.array:
    """Res2sDiffusionStep.step with sigma_up = eta sigma_next (legacy mode:
    the result is blended with the clean latent on the denoise mask). The
    noise is drawn whether or not it is used, as upstream does."""
    noise = _res2s_noise(sample.shape, key)
    sigma_up = min(sigma_next * eta, sigma_next * 0.9999)
    if sigma_up == 0 or sigma_next == 0:
        x = denoised
    else:
        residual = max(sigma_next**2 - sigma_up**2, 0.0) ** 0.5
        alpha_ratio = (1.0 - sigma_next) + residual
        sigma_down = residual / alpha_ratio
        eps = (sample - denoised) / (sigma - sigma_next)
        x = alpha_ratio * ((sample - sigma * eps) + sigma_down * eps) + sigma_up * noise
    return blend(x, state)


def res2s_loop(
    denoise: Denoiser,
    video: LatentState,
    audio: LatentState,
    sigmas: list[float],
    noise_seed: int,
    eta: float = 0.5,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[mx.array, mx.array]:
    """Upstream res2s_audio_video_denoising_loop: per step an x0 at sigma, a
    midpoint at sqrt(sigma sigma_next) reached with a21 and SDE-noised from
    the substep stream (eta 0.5), an anchor refinement when the step is small
    (h < 0.5 and sigma > 0.03), an x0 at the midpoint, the RK combination
    b1 / b2, SDE noise from the step stream (eta); a terminal 0 is replaced by
    0.0011 and the loop ends with that x0. Two draws per modality per step
    (video first), each stream from its own seed."""
    import math

    if sigmas[-1] == 0:
        sigmas = [*sigmas[:-1], RES2S_TERMINAL_SIGMA, 0.0]
        terminal = True
    else:
        terminal = False
    n_steps = len(sigmas) - 2 if terminal else len(sigmas) - 1
    hs = [-math.log(sigmas[i + 1] / sigmas[i]) for i in range(n_steps)]
    vx, ax = video.latent, audio.latent
    step_key = mx.random.key(noise_seed)
    sub_key = mx.random.key(noise_seed + RES2S_SUBSTEP_SEED_OFFSET)

    for i in range(n_steps):
        s, s_next = sigmas[i], sigmas[i + 1]
        h = hs[i]
        a21, b1, b2 = res2s_coefficients(h)
        sub = math.sqrt(s * s_next)
        v0, a0 = denoise(video, audio, vx, ax, s, step=i)
        v0, a0 = blend(v0, video), blend(a0, audio)
        ev, ea = v0 - vx, a0 - ax
        vm, am = vx + h * a21 * ev, ax + h * a21 * ea
        sub_key, k1, k2 = mx.random.split(sub_key, 3)
        vm = _res2s_inject(video, vx, vm, s, sub, 0.5, k1)
        am = _res2s_inject(audio, ax, am, s, sub, 0.5, k2)
        va, aa = vx, ax  # the anchors
        if bongmath and h < 0.5 and s > 0.03:
            for _ in range(bongmath_max_iter):
                va = vm - h * a21 * ev
                ev = v0 - va
                aa = am - h * a21 * ea
                ea = a0 - aa
            mx.eval(va, ev, aa, ea)
        # upstream evaluates the midpoint with step_index 0 (a skipping guider
        # never skips it)
        v2, a2 = denoise(video, audio, vm, am, sub, step=0)
        v2, a2 = blend(v2, video), blend(a2, audio)
        vn = va + h * (b1 * ev + b2 * (v2 - va))
        an = aa + h * (b1 * ea + b2 * (a2 - aa))
        step_key, k1, k2 = mx.random.split(step_key, 3)
        vn = _res2s_inject(video, va, vn, s, s_next, eta, k1)
        an = _res2s_inject(audio, aa, an, s, s_next, eta, k2)
        vx, ax = vn, an
        mx.async_eval(vx, ax)
        if on_step:
            on_step(i, s)
    if terminal:
        v0, a0 = denoise(video, audio, vx, ax, sigmas[n_steps], step=n_steps)
        vx, ax = blend(v0, video), blend(a0, audio)
        mx.async_eval(vx, ax)
        if on_step:
            on_step(n_steps, sigmas[n_steps])
    return vx, ax
