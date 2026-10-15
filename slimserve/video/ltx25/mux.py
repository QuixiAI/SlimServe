# SPDX-License-Identifier: Apache-2.0
"""mp4 writer: raw RGB frames on ffmpeg's stdin, H.264 + AAC.

Encode settings follow the reference pipeline: libx264, yuv420p, CRF 18,
AAC from 16-bit PCM; both streams written in full (no `-shortest`: the audio
is a few milliseconds off the video length and truncating would drop frames).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH (brew install ffmpeg)")
    return path


def write_wav(path: str | os.PathLike, waveform: np.ndarray, sample_rate: int) -> None:
    """float waveform (channels, samples) in [-1, 1] -> 16-bit PCM WAV."""
    wav = np.asarray(waveform, dtype=np.float32)
    wav = wav[None] if wav.ndim == 1 else wav
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype(np.int16).T
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.ascontiguousarray(pcm).tobytes())


def write_mp4(
    path: str | os.PathLike,
    frames: np.ndarray | Iterable[np.ndarray],
    fps: float,
    waveform: np.ndarray | None = None,
    sample_rate: int = 48000,
    crf: int = 18,
    size: tuple[int, int] | None = None,
) -> None:
    """Write `frames` (uint8 (F, H, W, 3), or an iterable of (H, W, 3) /
    (n, H, W, 3) chunks so a tiled decode can stream; then pass size=(H, W))
    and an optional (channels, samples) float waveform to `path`."""
    if isinstance(frames, np.ndarray):
        size = frames.shape[1:3]
        frames = (frames,)
    if size is None:
        raise ValueError("size=(H, W) is required when frames is an iterable")
    height, width = size
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    wav_path = None
    if waveform is not None:
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        write_wav(wav_path, waveform, sample_rate)
        cmd += ["-i", wav_path, "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), str(path)]
    try:
        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
            try:
                for chunk in frames:
                    chunk = np.ascontiguousarray(chunk, dtype=np.uint8)
                    if chunk.shape[-3:] != (height, width, 3):
                        raise ValueError(
                            f"frame shape {chunk.shape}, "
                            f"expected (..., {height}, {width}, 3)"
                        )
                    proc.stdin.write(memoryview(chunk).cast("B"))
            except BrokenPipeError:
                pass  # ffmpeg died; its stderr below says why
            finally:
                with contextlib.suppress(BrokenPipeError):
                    proc.stdin.close()
                proc.wait()
            if proc.returncode != 0:
                err.seek(0)
                raise RuntimeError(
                    f"ffmpeg exited {proc.returncode}: "
                    f"{err.read().decode(errors='replace')[-2000:]}"
                )
    finally:
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)
