# SPDX-License-Identifier: Apache-2.0
"""Still-image conditioning input (image-to-video first frame).

Follows Lightricks' `ltx_pipelines.utils.media_io`: decode to oriented sRGB
uint8 (decode.py decode_image), re-compress the still as one H.264 frame at the
checkpoint's CRF (decode.py preprocess / encode_single_frame: libx264, preset
veryfast, yuv420p; an LTX-2.5 checkpoint uses CRF 18, constants.py
LTX_2_4_IMAGE_CRF) so the conditioning carries the compression statistics the
model was trained on, resize preserving aspect ratio to fill the target with
bilinear interpolation (align_corners=False, no antialias), center crop
(resize.py resize_and_center_crop), and map to [-1, 1] (range_map.py
normalize_images: x / 127.5 - 1).
"""

from __future__ import annotations

import io
import math
import os
import subprocess

import mlx.core as mx
import numpy as np

DEFAULT_IMAGE_CRF = 18  # upstream LTX_2_4_PARAMS.default_image_crf (2.4+ checkpoints)

_ORIENTATION_TO_ROTATION = {3: 180, 6: 270, 8: 90}


def decode_image(source: str | bytes | os.PathLike) -> np.ndarray:
    """Path or encoded bytes -> uint8 (H, W, 3) sRGB, EXIF orientation applied."""
    from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError

    orientation_key = next(k for k, v in ExifTags.TAGS.items() if v == "Orientation")
    handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source
    try:
        with Image.open(handle) as src:
            image = src
            orientation = image.getexif().get(orientation_key)
            if orientation in _ORIENTATION_TO_ROTATION:
                image = image.rotate(_ORIENTATION_TO_ROTATION[orientation], expand=True)
            icc = image.info.get("icc_profile")
            if image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode == "LA":
                image = image.convert("L")
            if icc:
                try:
                    image = ImageCms.profileToProfile(
                        image,
                        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                        ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")),
                        outputMode="RGB",
                    )
                except (ImageCms.PyCMSError, OSError, ValueError):
                    image = image.convert("RGB")
            else:
                image = image.convert("RGB")
            return np.array(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"cannot decode conditioning image: {exc}") from exc


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    from slimserve.video.ltx25.mux import _ffmpeg as binary

    proc = subprocess.run(
        [binary(), "-y", "-loglevel", "error", *args],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg exited {proc.returncode}: "
            f"{proc.stderr.decode(errors='replace')[-2000:]}"
        )
    return proc.stdout


def recompress(image: np.ndarray, crf: int) -> np.ndarray:
    """One libx264 round trip at `crf` (preset veryfast, yuv420p), as upstream
    `preprocess`: odd edges are trimmed to even sizes first (codec requirement);
    crf 0 and sub-2-pixel images pass through."""
    if crf == 0 or min(image.shape[0], image.shape[1]) < 2:
        return image
    h, w = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    image = np.ascontiguousarray(image[:h, :w])
    raw = [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        "1",
        "-i",
        "-",
    ]
    # PyAV's reformat() defaults to bilinear chroma scaling on both legs.
    encoded = _ffmpeg(
        raw
        + [
            "-frames:v",
            "1",
            "-sws_flags",
            "bilinear",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-f",
            "h264",
            "-",
        ],
        image.tobytes(),
    )
    decoded = _ffmpeg(
        [
            "-f",
            "h264",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-sws_flags",
            "bilinear",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        encoded,
    )
    if len(decoded) != h * w * 3:
        raise RuntimeError(
            f"ffmpeg returned {len(decoded)} bytes for a {w}x{h} rgb24 frame"
        )
    return np.frombuffer(decoded, dtype=np.uint8).reshape(h, w, 3)


def resize_plan(src_h: int, src_w: int, height: int, width: int):
    """Upstream resize_and_center_crop geometry: (new_h, new_w, top, left)."""
    scale = max(height / src_h, width / src_w)
    new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
    return new_h, new_w, (new_h - height) // 2, (new_w - width) // 2


def _bilinear_weights(
    n_in: int, n_out: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """torch bilinear, align_corners=False, no antialias: source coordinate
    (i + 0.5) * in / out - 0.5 clamped at 0; taps floor and floor + 1 (clamped)."""
    src = (np.arange(n_out, dtype=np.float64) + 0.5) * (n_in / n_out) - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_in - 1)
    i1 = np.minimum(i0 + 1, n_in - 1)
    lam = (src - i0).astype(np.float32)
    return i0, i1, lam


def resize_bilinear(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """(H, W, C) float32 -> (new_h, new_w, C) float32, separable bilinear."""
    x = image.astype(np.float32, copy=False)
    r0, r1, lr = _bilinear_weights(x.shape[0], new_h)
    x = x[r0] * (1.0 - lr)[:, None, None] + x[r1] * lr[:, None, None]
    c0, c1, lc = _bilinear_weights(x.shape[1], new_w)
    return x[:, c0] * (1.0 - lc)[None, :, None] + x[:, c1] * lc[None, :, None]


def _uint8_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray, int]:
    """torch's int16 fixed-point bilinear taps for a uint8 axis
    (UpSampleKernel.cpp _compute_indices_min_size_weights and
    _compute_index_ranges_int16_weights, antialias=False): per output index the
    first source index and two weights (the second folded into the first at the
    last pixel), weights rounded to `precision` fractional bits, where precision
    is the largest keeping twice the biggest weight below 2^15."""
    i0, _i1, lam = _bilinear_weights(n_in, n_out)
    lam = lam.astype(np.float64)
    w = np.stack([1.0 - lam, lam], axis=1)
    last = i0 == n_in - 1  # only one source pixel: both taps land on it
    w[last] = [1.0, 0.0]
    wt_max = w.max()
    precision = 0
    while precision < 22:
        if int(0.5 + wt_max * (1 << (precision + 1))) >= 1 << 15:
            break
        precision += 1
    q = np.floor(0.5 + w * (1 << precision)).astype(np.int64)  # v >= 0 here
    return i0, q, precision


def _uint8_pass(x: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    i0, q, precision = _uint8_taps(x.shape[axis], n_out)
    i1 = np.minimum(i0 + 1, x.shape[axis] - 1)
    a = np.moveaxis(x, axis, 0).astype(np.int64)
    shape = (-1,) + (1,) * (a.ndim - 1)
    acc = (
        (1 << (precision - 1))
        + a[i0] * q[:, 0].reshape(shape)
        + a[i1] * q[:, 1].reshape(shape)
    )
    out = np.clip(acc >> precision, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def resize_bilinear_uint8(image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """(H, W, C) uint8 -> (new_h, new_w, C) uint8 exactly as torch interpolates
    a uint8 tensor (bilinear, align_corners=False, no antialias): a horizontal
    pass then a vertical pass, each in int16 fixed point rounding to uint8
    (UpSampleKernelAVXAntialias.h / UpSampleKernelNEONAntialias.h, the generic
    separable kernel has the same arithmetic). Differs from float bilinear
    rounded in about a tenth of the pixels by one level."""
    x = np.asarray(image, dtype=np.uint8)
    if x.shape[1] != new_w:
        x = _uint8_pass(x, new_w, 1)
    if x.shape[0] != new_h:
        x = _uint8_pass(x, new_h, 0)
    return x


def resize_and_center_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """(H, W, C) -> (height, width, C) float32: scale to fill, then center crop."""
    new_h, new_w, top, left = resize_plan(image.shape[0], image.shape[1], height, width)
    resized = resize_bilinear(image, new_h, new_w)
    return resized[top : top + height, left : left + width]


def resize_and_reflect_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Upstream resize.py resize_and_reflect_pad (the HDR reference frames):
    keep the aspect, scale to fit inside (no interpolation when the target
    already holds the source), pad bottom / right by reflection (replicate
    when the pad would exceed the content)."""
    src_h, src_w = image.shape[:2]
    if height >= src_h and width >= src_w:
        x = np.asarray(image, dtype=np.float32)
    else:
        scale = min(height / src_h, width / src_w)
        x = resize_bilinear(image, round(src_h * scale), round(src_w * scale))
    pad_b, pad_r = height - x.shape[0], width - x.shape[1]
    if pad_b > 0 or pad_r > 0:
        mode = "reflect" if pad_b < x.shape[0] and pad_r < x.shape[1] else "edge"
        x = np.pad(x, [(0, pad_b), (0, pad_r), (0, 0)], mode=mode)
    return x


class HDRFrame:
    """A scene-linear (or ACEScct-coded) float still from an EXR, with the
    colour space it was declared in; resized per stage and compressed into
    the model's ACEScct working space there (as upstream resizes the linear
    pixels before compressing)."""

    def __init__(self, linear: np.ndarray, color_space: str):
        self.linear, self.color_space = linear, color_space
        self.shape = linear.shape


def hdr_conditioning_frame(frame: HDRFrame, height: int, width: int) -> mx.array:
    """EXR still -> (1, 3, 1, height, width) fp32 in [-1, 1] (upstream
    load_exr_image_as_hdr_conditioning: center crop, to_working_space,
    to_vae_range)."""
    from slimserve.video.ltx25 import hdr

    px = resize_and_center_crop(frame.linear, height, width)
    codes = hdr.to_working_space(px, frame.color_space) * 2.0 - 1.0
    return mx.array(np.ascontiguousarray(codes.transpose(2, 0, 1)))[None, :, None]


def prepare_image(
    source: str | bytes | os.PathLike,
    crf: int = DEFAULT_IMAGE_CRF,
    color_space: str | None = None,
):
    """Decode and CRF round-trip once; the result is resized per stage. An
    `.exr` path (needs `color_space`) is read scene-linear and skips the
    round trip (upstream load_image_and_preprocess)."""
    if isinstance(source, (str, os.PathLike)) and str(source).lower().endswith(".exr"):
        if color_space is None:
            raise ValueError("an EXR still needs the hdr colour space")
        from slimserve.video.ltx25 import hdr

        return HDRFrame(hdr.read_exr(source), color_space)
    return recompress(decode_image(source), crf)


def conditioning_frame(image, height: int, width: int) -> mx.array:
    """Prepared uint8 (H, W, 3) -> (1, 3, 1, height, width) fp32 in [-1, 1]
    (an HDRFrame through the ACEScct working space)."""
    if isinstance(image, HDRFrame):
        return hdr_conditioning_frame(image, height, width)
    frame = resize_and_center_crop(image, height, width) / 127.5 - 1.0
    return mx.array(np.ascontiguousarray(frame.transpose(2, 0, 1)))[None, :, None]


def load_conditioning_image(
    source: str | bytes | os.PathLike,
    height: int,
    width: int,
    crf: int = DEFAULT_IMAGE_CRF,
) -> mx.array:
    """Path or image bytes -> (1, 3, 1, height, width) fp32 in [-1, 1], ready
    for `VideoVAE.encode`."""
    return conditioning_frame(prepare_image(source, crf), height, width)
