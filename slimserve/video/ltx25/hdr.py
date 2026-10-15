# SPDX-License-Identifier: Apache-2.0
"""HDR in and out (upstream's `--hdr {SRGB_LINEAR, ACESCG, ACESCCT}`):
`ltx_core/hdr.py` (the ACEScct working space), `ltx_core/color/primaries.py`
(Rec.709 / ACEScg / Rec.2020 matrices, colour-science's CAT02-adapted
values dumped as constants), `ltx_core/color/hlg.py` (the HLG master),
`ltx_core/color/yuv.py` (BT.2020 10-bit planes) and
`ltx_pipelines/utils/media_io/exr.py` (EXR frames; OpenEXR here in place of
OpenImageIO) in numpy.

The model works in ACEScct [0, 1] codes mapped to the VAE's [-1, 1]: an EXR
input (scene-linear in Rec.709 or ACEScg primaries, or ACEScct codes) is
compressed into that space before the encoder; a decoded HDR clip is
decompressed back to scene-linear and written twice: an EXR frame folder in
the request's colour space, and an HLG master (Rec.2020, BT.2100 HLG OETF,
HEVC Main10 through ffmpeg's libx265 with upstream's x265 parameters).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

COLOR_SPACES = ("srgb_linear", "acescg", "acescct")

# ACEScct (ACES TB-2014-004)
_A_LIN, _B_LIN = 10.5402377416545, 0.0729055341958355
_X_BRK, _Y_BRK = 0.0078125, 0.155251141552511
_LOG_M, _LOG_B = 17.52, 9.72

# colour.matrix_RGB_to_RGB with the CAT02 adaptation, as upstream builds them
SRGB_TO_ACESCG = np.array(
    [
        [0.6130974292755127, 0.33952316641807556, 0.04737945273518562],
        [0.07019372284412384, 0.9163538813591003, 0.013452397659420967],
        [0.0206155925989151, 0.10956976562738419, 0.8698146343231201],
    ],
    dtype=np.float32,
)
ACESCG_TO_SRGB = np.array(
    [
        [1.7050509452819824, -0.6217921376228333, -0.08325887471437454],
        [-0.13025641441345215, 1.1408047676086426, -0.010548318736255169],
        [-0.02400335669517517, -0.1289689689874649, 1.1529723405838013],
    ],
    dtype=np.float32,
)
REC709_TO_2020 = np.array(
    [
        [0.6274039149284363, 0.3292830288410187, 0.04331306740641594],
        [0.06909728795289993, 0.9195404052734375, 0.011362315155565739],
        [0.016391439363360405, 0.08801330626010895, 0.8955952525138855],
    ],
    dtype=np.float32,
)
ACESCG_TO_2020 = np.array(
    [
        [1.025824785232544, -0.020053191110491753, -0.005771556869149208],
        [-0.0022343695163726807, 1.0045864582061768, -0.002352132461965084],
        [-0.0050133513286709785, -0.025290071964263916, 1.0303034782409668],
    ],
    dtype=np.float32,
)
# EXR chromaticities (R, G, B, W x,y) per primaries
CHROMATICITIES = {
    "rec709": (0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329),
    "ap1": (0.713, 0.293, 0.165, 0.83, 0.128, 0.044, 0.32168, 0.33767),
}
# colour.matrix_YCbCr(WEIGHTS_YCBCR["ITU-R BT.2020"]) inverted: full-range RGB -> Y'CbCr
RGB_TO_YCBCR_2020 = np.array(
    [
        [0.26269999146461487, 0.6779999732971191, 0.059300001710653305],
        [-0.13963006436824799, -0.3603699505329132, 0.5],
        [0.5, -0.45978569984436035, -0.04021429643034935],
    ],
    dtype=np.float32,
)
# ARIB STD-B67 / BT.2100 HLG OETF (colour-science CONSTANTS_ARIBSTDB67)
_HLG_A, _HLG_B, _HLG_C = 0.17883277, 0.28466892, 0.55991073
HLG_WHITE_SIGNAL = 0.75
HLG_WHITE_LINEAR = 0.26496256042100724  # colour oetf_inverse_BT2100_HLG(0.75)


def source_primaries(color_space: str) -> str:
    return "ap1" if color_space in ("acescg", "acescct") else "rec709"


def _apply(mat: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """A 3x3 primaries matrix on (..., 3) pixels."""
    return np.asarray(rgb, dtype=np.float32) @ mat.T


def acescct_compress(linear_acescg: np.ndarray) -> np.ndarray:
    """linear ACEScg [0, inf) -> ACEScct [0, 1] (upstream _compress)."""
    x = np.clip(np.asarray(linear_acescg, dtype=np.float32), 0.0, None)
    log_part = (np.log2(np.clip(x, 1e-12, None)) + _LOG_B) / _LOG_M
    lin_part = _A_LIN * x + _B_LIN
    out = np.clip(np.where(x > _X_BRK, log_part, lin_part), 0.0, 1.0)
    return out.astype(np.float32)


def acescct_decompress(ct: np.ndarray) -> np.ndarray:
    """ACEScct [0, 1] -> linear ACEScg (upstream _decompress)."""
    ct = np.clip(np.asarray(ct, dtype=np.float32), 0.0, 1.0)
    from_log = np.power(2.0, ct * _LOG_M - _LOG_B)
    from_lin = (ct - _B_LIN) / _A_LIN
    return np.where(ct > _Y_BRK, from_log, from_lin).astype(np.float32)


def srgb_eotf_to_linear(srgb: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0, 1] -> display-linear (IEC 61966-2-1)."""
    x = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(
        np.float32
    )


def to_working_space(rgb: np.ndarray, color_space: str) -> np.ndarray:
    """HDR float RGB (..., 3) of `color_space` -> ACEScct [0, 1] codes (upstream
    to_working_space / to_acescct_working_space)."""
    if color_space == "acescct":
        return np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    linear = rgb if color_space == "acescg" else _apply(SRGB_TO_ACESCG, rgb)
    return acescct_compress(np.clip(linear, 0.0, None))


def to_linear(codes: np.ndarray, primaries: str = "rec709") -> np.ndarray:
    """ACEScct codes (..., 3) -> scene-linear in `primaries` (upstream
    to_hdr_linear / decode_hdr_video)."""
    linear = acescct_decompress(codes)
    if primaries == "ap1":
        return np.clip(linear, 0.0, None)
    return np.clip(_apply(ACESCG_TO_SRGB, linear), 0.0, None)


def srgb_video_to_working_space(rgb_01: np.ndarray, gamma_encoded: bool) -> np.ndarray:
    """Rec.709 float RGB -> ACEScct (upstream srgb_to_acescct): display-encoded
    video (an mp4) goes through the sRGB EOTF first."""
    x = np.asarray(rgb_01, dtype=np.float32)
    if gamma_encoded:
        x = srgb_eotf_to_linear(x)
    return to_working_space(x, "srgb_linear")


# ---- EXR frames -----------------------------------------------------------------
def is_exr_dir(path: str | os.PathLike) -> bool:
    p = Path(path)
    return p.is_dir() and any(p.glob("*.exr"))


def exr_paths(path: str | os.PathLike) -> list[Path]:
    """Upstream _exr_paths_for_conditioning: one .exr or a sorted folder."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".exr":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.exr"))
        if not files:
            raise RuntimeError(f"no EXR frames in {path}")
        return files
    raise ValueError(f"an EXR path is a .exr file or a folder of them: {path!r}")


def read_exr(path: str | os.PathLike) -> np.ndarray:
    """A scene-linear EXR frame as float32 (H, W, 3) RGB (extra channels
    dropped, one channel broadcast), values unmodified."""
    import OpenEXR

    channels = OpenEXR.File(str(path)).channels()
    if "RGB" in channels or "RGBA" in channels:
        px = channels["RGB" if "RGB" in channels else "RGBA"].pixels
        arr = np.asarray(px, dtype=np.float32)[..., :3]
    else:
        names = [n for n in ("R", "G", "B") if n in channels]
        if not names:
            first = next(iter(channels.values())).pixels
            arr = np.repeat(np.asarray(first, dtype=np.float32)[..., None], 3, axis=-1)
        else:
            planes = [np.asarray(channels[n].pixels, dtype=np.float32) for n in names]
            while len(planes) < 3:
                planes.append(planes[-1])
            arr = np.stack(planes, axis=-1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return np.ascontiguousarray(arr)


def write_exr(
    path: str | os.PathLike, rgb: np.ndarray, primaries: str, color_space_tag: str
) -> None:
    """Upstream save_exr_tensor: half floats, ZIP, the primaries' chromaticities
    and a colorSpace tag ("sRGB" / "ACEScg" for linear, "ACEScct" for log)."""
    import OpenEXR

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "colorSpace": color_space_tag,
        "chromaticities": tuple(CHROMATICITIES[primaries]),
    }
    OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb, dtype=np.float16)}).write(
        str(path)
    )


EXR_TAGS = {"acescct": "ACEScct", "acescg": "ACEScg", "srgb_linear": "sRGB"}


def exr_tag(color_space: str) -> str:
    return EXR_TAGS[color_space]


# ---- the HLG master ---------------------------------------------------------------
def hlg_oetf(x: np.ndarray) -> np.ndarray:
    return np.clip(
        np.where(
            x <= 1.0 / 12.0,
            np.sqrt(np.clip(3.0 * x, 0.0, None)),
            _HLG_A * np.log(np.clip(12.0 * x - _HLG_B, 1e-12, None)) + _HLG_C,
        ),
        0.0,
        1.0,
    ).astype(np.float32)


def linear_to_hlg_planes(
    frames_linear_709: np.ndarray, white_signal: float = HLG_WHITE_SIGNAL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upstream HlgGpuConverter: (F, H, W, 3) scene-linear Rec.709 ->
    Rec.2020 -> diffuse white mapped to the signal's `white_signal` with
    highlights rolled toward 1 -> HLG OETF -> BT.2020 Y'CbCr, limited range,
    4:2:0 (2x2 mean), 10-bit planes (uint16)."""
    white_x = HLG_WHITE_LINEAR
    roll_k = white_x / (1.0 - white_x)
    lin = np.clip(np.nan_to_num(_apply(REC709_TO_2020, frames_linear_709)), 0.0, None)
    x = np.where(
        lin <= 1.0,
        lin * white_x,
        1.0 - (1.0 - white_x) * np.exp(-roll_k * (lin - 1.0)),
    )
    hlg = hlg_oetf(x)
    ycc = hlg @ RGB_TO_YCBCR_2020.T  # (F, H, W, 3)
    y = ycc[..., 0] * (219 * 4) + 16 * 4
    f, h, w = ycc.shape[:3]
    uv = ycc[..., 1:3].reshape(f, h // 2, 2, w // 2, 2, 2).mean(axis=(2, 4))
    uv = uv * (224 * 4) + 128 * 4
    to10 = lambda p: np.clip(np.rint(p), 0, 1023).astype(np.uint16)  # noqa: E731
    return to10(y), to10(uv[..., 0]), to10(uv[..., 1])


def x265_params(width: int, height: int, threads: int) -> str:
    base = (
        "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=limited:"
        f"repeat-headers=1:info=0:pools={threads}"
    )
    if width <= 32 and height <= 32:
        return f"{base}:frame-threads=1:bframes=0:lookahead=0"
    return f"{base}:frame-threads=4"


def write_hlg_mp4(
    path: str | os.PathLike,
    frames_linear_709: np.ndarray,
    fps: float,
    waveform: np.ndarray | None = None,
    sample_rate: int | None = None,
    crf: int = 12,
    preset: str = "ultrafast",
) -> None:
    """The HLG master (upstream encode_linear_hdr_frames_to_hlg_mp4): HEVC
    Main10 yuv420p10le tagged BT.2020 / arib-std-b67 / bt2020nc limited,
    `hvc1`, faststart, AAC audio when given."""
    from slimserve.video.ltx25.mux import _ffmpeg, write_wav

    f, h, w = frames_linear_709.shape[:3]
    y, u, v = linear_to_hlg_planes(frames_linear_709)
    raw = b"".join(
        np.concatenate([y[i].reshape(-1), u[i].reshape(-1), v[i].reshape(-1)])
        .astype("<u2")
        .tobytes()
        for i in range(f)
    )
    threads = max(1, min(os.cpu_count() or 8, 16))
    args = [
        _ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p10le",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
    ]
    wav_path = None
    if waveform is not None and sample_rate is not None:
        wav_path = Path(path).with_suffix(".hlg_audio.wav")
        write_wav(str(wav_path), waveform, sample_rate)
        args += ["-i", str(wav_path)]
    args += [
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-x265-params",
        x265_params(w, h, threads),
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "arib-std-b67",
        "-colorspace",
        "bt2020nc",
        "-color_range",
        "tv",
        "-tag:v",
        "hvc1",
    ]
    if wav_path is not None:
        args += ["-c:a", "aac", "-b:a", "192k"]
    args += ["-movflags", "+faststart", str(path)]
    try:
        proc = subprocess.run(args, input=raw, capture_output=True)
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg (HLG) exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')[-2000:]}"
        )


def write_hdr_outputs(
    path: str | os.PathLike,
    codes: np.ndarray,
    fps: float,
    color_space: str,
    waveform: np.ndarray | None = None,
    sample_rate: int | None = None,
) -> Path:
    """Upstream _encode_hdr_video_outputs: `codes` are the decoded ACEScct
    [0, 1] frames (F, H, W, 3). The HLG master at `path` is always built from
    Rec.709 scene-linear; the EXR folder `<stem>_<color_space>_exr/` holds
    log codes for acescct, else scene-linear in the colour space's primaries.
    Returns the EXR folder."""
    path = Path(path)
    exr_dir = path.parent / f"{path.stem}_{color_space}_exr"
    exr_dir.mkdir(parents=True, exist_ok=True)
    linear_709 = to_linear(codes, "rec709")
    if color_space == "acescct":
        exr = np.asarray(codes, dtype=np.float32)
    elif color_space == "acescg":
        exr = to_linear(codes, "ap1")
    else:
        exr = linear_709
    primaries, tag = source_primaries(color_space), exr_tag(color_space)
    for i, frame in enumerate(exr):
        write_exr(exr_dir / f"frame_{i:05d}.exr", frame, primaries, tag)
    write_hlg_mp4(path, linear_709, fps, waveform, sample_rate)
    return exr_dir
