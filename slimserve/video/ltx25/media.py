# SPDX-License-Identifier: Apache-2.0
"""Source video and audio for the editing pipelines (retake, IC-LoRA, A2Vid,
DubIt): what Lightricks' `ltx_pipelines.utils.media_io.decode` reads through
PyAV, read here through the `ffmpeg` binary (as the muxer and the still's
H.264 round trip already do), with the same conversions:

- video frames: `frame.to_rgb().to_ndarray()` is swscale's yuv -> rgb24 at its
  default (bilinear) flags, so `-sws_flags bilinear -pix_fmt rgb24`; frames are
  selected by presentation time (`decode_video_from_file`: pts >= start, < start
  + max_duration), here with accurate input seeking and `-t`;
- audio: `_audio_frame_to_float` divides integer PCM by its full scale and
  keeps the stream's own channel count and rate, so `-f f32le` at the native
  rate and layout; `decode_audio_from_file` trims to [start, start +
  max_duration) by sample count;
- resampling to the audio VAE's 16 kHz is torchaudio's `resample`
  (`AudioProcessor.resample_audio`): the Hann-windowed sinc kernel with
  lowpass_filter_width 6 and rolloff 0.99, one filter per output phase, the
  input zero-padded by the kernel half-width (`resample_sinc` below).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def _ffmpeg_binary() -> str:
    from slimserve.video.ltx25.mux import _ffmpeg as binary

    return binary()


def _run(args: list[str]) -> bytes:
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(args[0])} exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')[-2000:]}"
        )
    return proc.stdout


@dataclass(frozen=True)
class VideoInfo:
    """Upstream get_videostream_metadata: the stream's frame count (counted by
    decoding when the container does not say), size and average rate."""

    frames: int
    height: int
    width: int
    fps: float
    has_audio: bool
    audio_rate: int | None = None
    audio_channels: int | None = None


def probe(path: str | os.PathLike, fps: float | None = None) -> VideoInfo:
    """Upstream get_videostream_metadata; an EXR frame folder has no container
    rate, so `fps` is required for one (`--frame-rate`)."""
    from slimserve.video.ltx25 import hdr

    if hdr.is_exr_dir(path):
        if fps is None:
            raise ValueError(f"{path} is an EXR frame folder; its frame rate is needed")
        files = hdr.exr_paths(path)
        h, w = hdr.read_exr(files[0]).shape[:2]
        return VideoInfo(len(files), h, w, float(fps), has_audio=False)
    ffprobe = _ffmpeg_binary().replace("ffmpeg", "ffprobe")
    out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-count_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = json.loads(out)["streams"]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: no video stream")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    if frames <= 0:
        raise ValueError(f"{path}: could not count the video frames")
    return VideoInfo(
        frames=frames,
        height=int(video["height"]),
        width=int(video["width"]),
        fps=float(Fraction(video["avg_frame_rate"])),
        has_audio=audio is not None,
        audio_rate=int(audio["sample_rate"]) if audio else None,
        audio_channels=int(audio["channels"]) if audio else None,
    )


def probe_audio(path: str | os.PathLike) -> VideoInfo:
    """A file that may be audio-only (A2Vid's --audio-path): the audio
    stream's rate and layout, the video fields zero when there is none."""
    ffprobe = _ffmpeg_binary().replace("ffmpeg", "ffprobe")
    out = _run([ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)])
    streams = json.loads(out)["streams"]
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError(f"{path}: no audio stream")
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    return VideoInfo(
        frames=int(video.get("nb_frames") or 0) if video else 0,
        height=int(video["height"]) if video else 0,
        width=int(video["width"]) if video else 0,
        fps=float(Fraction(video["avg_frame_rate"]))
        if video and video.get("avg_frame_rate", "0/0") != "0/0"
        else 0.0,
        has_audio=True,
        audio_rate=int(audio["sample_rate"]),
        audio_channels=int(audio["channels"]),
    )


def read_frames(
    path: str | os.PathLike,
    info: VideoInfo,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> np.ndarray:
    """Decoded frames as uint8 (F, H, W, 3), the frames whose presentation
    time lies in [start_time, start_time + max_duration). An EXR folder gives
    float32 scene-linear frames by index instead (upstream
    load_exr_as_hdr_conditioning: frame_start round(start * fps), frame_cap
    round(duration * fps))."""
    from slimserve.video.ltx25 import hdr

    if hdr.is_exr_dir(path):
        files = hdr.exr_paths(path)[round(start_time * info.fps) :]
        if max_duration is not None:
            files = files[: max(1, round(max_duration * info.fps))]
        return np.stack([hdr.read_exr(f) for f in files])
    args = [_ffmpeg_binary(), "-loglevel", "error"]
    if start_time > 0:
        args += ["-accurate_seek", "-ss", f"{start_time:.6f}"]
    args += ["-i", str(path)]
    if max_duration is not None:
        args += ["-t", f"{max_duration:.6f}"]
    # passthrough: the decoded frames themselves, as PyAV yields them; the
    # default constant-rate mode duplicates frames when the container's average
    # rate is below its nominal one (a 23.6 fps clip came back with 246 frames
    # for 241)
    args += [
        "-an",
        "-fps_mode",
        "passthrough",
        "-sws_flags",
        "bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = _run(args)
    per = info.height * info.width * 3
    if len(raw) % per:
        raise RuntimeError(f"{path}: decoded {len(raw)} bytes, not whole frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, info.height, info.width, 3)


def read_frames_by_index(
    path: str | os.PathLike,
    info: VideoInfo,
    start_frame: int = 0,
    count: int | None = None,
) -> np.ndarray:
    """Upstream decode_video_by_frame(starting_frame, frame_cap): frames
    selected by decode index rather than by presentation time, so a window
    of `count` frames is exactly `count` long whatever the stream's timestamp
    jitter (the chunked IC-LoRA / Dub-It reference windows need their frame
    count on the 8k + 1 grid)."""
    from slimserve.video.ltx25 import hdr

    if start_frame < 0 or (count is not None and count < 0):
        raise ValueError(f"start_frame {start_frame} / count {count} must be >= 0")
    if hdr.is_exr_dir(path):
        files = hdr.exr_paths(path)[start_frame:]
        if count is not None:
            files = files[:count]
        return np.stack([hdr.read_exr(f) for f in files])
    if count is None:
        select = f"gte(n\\,{start_frame})"
    else:
        select = f"between(n\\,{start_frame}\\,{start_frame + count - 1})"
    args = [
        _ffmpeg_binary(),
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-an",
        "-vf",
        f"select={select}",
        "-fps_mode",
        "passthrough",
        "-sws_flags",
        "bilinear",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    if count is not None:
        args[-1:-1] = ["-frames:v", str(count)]
    raw = _run(args)
    per = info.height * info.width * 3
    if len(raw) % per:
        raise RuntimeError(f"{path}: decoded {len(raw)} bytes, not whole frames")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, info.height, info.width, 3)
    return frames if count is None else frames[:count]


def read_audio(
    path: str | os.PathLike,
    info: VideoInfo,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> tuple[np.ndarray, int] | None:
    """The audio stream as float32 (channels, samples) in [-1, 1] at its own
    rate, trimmed to [start_time, start_time + max_duration) by sample count
    as upstream; None without an audio stream."""
    if not info.has_audio:
        return None
    rate, channels = int(info.audio_rate), int(info.audio_channels)
    args = [_ffmpeg_binary(), "-loglevel", "error"]
    if start_time > 0:
        args += ["-accurate_seek", "-ss", f"{start_time:.6f}"]
    args += ["-i", str(path), "-vn", "-f", "f32le", "-"]
    raw = _run(args)
    samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T
    if max_duration is not None:
        samples = samples[:, : round(max_duration * rate)]
    if samples.shape[1] == 0:
        return None
    return np.ascontiguousarray(samples), rate


def resample_sinc(
    waveform: np.ndarray,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
) -> np.ndarray:
    """torchaudio.functional.resample (sinc_interp_hann) on (C, T) float32."""
    if orig_freq == new_freq:
        return np.asarray(waveform, dtype=np.float32)
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig, new = int(orig_freq) // gcd, int(new_freq) // gcd
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None] / new + idx
    t = np.clip(t * base, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    kernel = np.where(t == 0, 1.0, np.sin(t) / np.where(t == 0, 1.0, t))
    kernel = (kernel * window * (base / orig)).astype(np.float32)  # (new, taps)
    x = np.asarray(waveform, dtype=np.float32)
    length = x.shape[-1]
    padded = np.pad(x, [(0, 0), (width, width + orig)])
    taps = kernel.shape[1]
    # out[c, j*new + p] = sum_k kernel[p, k] * padded[c, j*orig + k]
    windows = np.lib.stride_tricks.sliding_window_view(padded, taps, axis=1)[
        :, ::orig
    ]  # (C, J, taps)
    out = np.einsum("cjk,pk->cjp", windows, kernel).reshape(x.shape[0], -1)
    target = math.ceil(new * length / orig)
    return np.ascontiguousarray(out[:, :target])
