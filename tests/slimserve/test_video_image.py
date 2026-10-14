# SPDX-License-Identifier: Apache-2.0
"""Image-to-video conditioning: the still's preprocessing (upstream
resize_and_center_crop / preprocess semantics), the request field, and the
latent-frame conditioning on a tiny synthetic state. CPU only; no weights."""

import base64
import io
import shutil

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")  # the stills reach MLX latents

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import image as image_mod  # noqa: E402
from slimserve.video.ltx25 import sampling  # noqa: E402

CFG = {
    "pipeline": "distilled",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def _png(width=8, height=6, color=(200, 30, 60)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


# ---- resize / crop rule -----------------------------------------------------
def test_resize_plan_scales_to_fill_and_center_crops():
    # 100 wide x 60 high into 64x64: the height is the tight side, scale 64/60.
    new_h, new_w, top, left = image_mod.resize_plan(60, 100, 64, 64)
    assert (new_h, new_w) == (64, 107)  # ceil(60 * 1.0667), ceil(100 * 1.0667)
    assert (top, left) == (0, 21)  # (107 - 64) // 2
    # Portrait source into a landscape target: the width is the tight side.
    assert image_mod.resize_plan(200, 100, 64, 128) == (256, 128, 96, 0)
    # Already the target size: no scaling, no crop.
    assert image_mod.resize_plan(64, 64, 64, 64) == (64, 64, 0, 0)


def test_bilinear_taps_follow_torch_align_corners_false():
    # Upsample 2 -> 4: source coordinates (i + 0.5) / 2 - 0.5 = -0.25, 0.25,
    # 0.75, 1.25 -> clamped 0, then taps (0, 1) at 0.25, (0, 1) at 0.75, and
    # the last has both taps clamped to index 1 (so its weight is moot).
    i0, i1, lam = image_mod._bilinear_weights(2, 4)
    assert i0.tolist() == [0, 0, 0, 1]
    assert i1.tolist() == [1, 1, 1, 1]
    assert np.allclose(lam, [0.0, 0.25, 0.75, 0.25])
    ramp = np.array([[[0.0], [4.0]]], dtype=np.float32)  # (1, 2, 1)
    out = image_mod.resize_bilinear(ramp, 1, 4)
    assert np.allclose(out[0, :, 0], [0.0, 1.0, 3.0, 4.0])
    # Downsample 4 -> 2 without antialias: coordinates 0.5 and 2.5, so each
    # output is the mean of two neighbours, never of all four.
    i0, i1, lam = image_mod._bilinear_weights(4, 2)
    assert i0.tolist() == [0, 2] and i1.tolist() == [1, 3]
    assert np.allclose(lam, [0.5, 0.5])


def test_resize_and_center_crop_is_identity_at_target_size_and_crops_centre():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, size=(16, 24, 3)).astype(np.uint8)
    same = image_mod.resize_and_center_crop(src, 16, 24)
    assert same.shape == (16, 24, 3) and np.array_equal(same, src.astype(np.float32))
    # Same height, narrower target: no scaling, a centred horizontal crop.
    crop = image_mod.resize_and_center_crop(src, 16, 16)
    assert np.array_equal(crop, src[:, 4:20].astype(np.float32))


def test_conditioning_frame_shape_and_range():
    img = np.full((6, 8, 3), 255, dtype=np.uint8)
    img[..., 1] = 0
    frame = image_mod.conditioning_frame(img, 64, 64)
    assert frame.shape == (1, 3, 1, 64, 64)
    assert frame.dtype.size == 4
    vals = np.array(frame)
    assert np.allclose(vals[0, 0], 1.0) and np.allclose(vals[0, 1], -1.0)


def test_decode_image_accepts_bytes_and_paths(tmp_path):
    data = _png(8, 6)
    assert image_mod.decode_image(data).shape == (6, 8, 3)
    path = tmp_path / "still.png"
    path.write_bytes(data)
    assert image_mod.decode_image(str(path)).shape == (6, 8, 3)
    with pytest.raises(ValueError):
        image_mod.decode_image(b"not an image")


def test_crf_zero_and_tiny_images_skip_the_round_trip():
    img = np.zeros((9, 7, 3), dtype=np.uint8)
    assert image_mod.recompress(img, 0) is img
    assert image_mod.recompress(np.zeros((1, 7, 3), dtype=np.uint8), 18).shape == (
        1,
        7,
        3,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_crf_round_trip_keeps_the_picture_and_trims_to_even_sizes():
    y, x = np.mgrid[0:33, 0:47]
    img = np.stack([x * 5, y * 7, (x + y) * 3], axis=-1).astype(np.uint8)
    out = image_mod.recompress(img, 18)
    assert out.shape == (32, 46, 3)  # odd edges trimmed, as upstream
    err = np.abs(out.astype(np.float32) - img[:32, :46].astype(np.float32))
    assert err.mean() < 6.0  # H.264 at CRF 18 is near-lossless on a gradient
    assert not np.array_equal(out, img[:32, :46])  # ... but not a no-op


# ---- request field ----------------------------------------------------------
def test_request_image_accepts_base64_and_data_urls():
    data = _png()
    b64 = base64.b64encode(data).decode()
    params = server.normalize_request({"prompt": "x", "image": b64}, CFG)
    assert params["image"] == data and params["image_strength"] == 1.0
    params = server.normalize_request(
        {"prompt": "x", "image": "data:image/png;base64," + b64, "image_strength": 0.6},
        CFG,
    )
    assert params["image"] == data and params["image_strength"] == 0.6
    assert "image" not in server.normalize_request({"prompt": "x"}, CFG)


def test_request_image_passes_raw_bytes_through_for_the_cli():
    data = _png()
    assert (
        server.normalize_request({"prompt": "x", "image": data}, CFG)["image"] == data
    )


@pytest.mark.parametrize(
    "body",
    [
        {"prompt": "x", "image": "@@not base64@@"},
        {"prompt": "x", "image": base64.b64encode(b"not an image").decode()},
        {"prompt": "x", "image": "data:image/png," + base64.b64encode(_png()).decode()},
        {"prompt": "x", "image": ""},
        {"prompt": "x", "image": 7},
        {
            "prompt": "x",
            "image": base64.b64encode(_png()).decode(),
            "image_strength": 1.5,
        },
        {
            "prompt": "x",
            "image": base64.b64encode(_png()).decode(),
            "image_strength": "a",
        },
        {"prompt": "x", "image_strength": 0.5},
    ],
)
def test_request_image_errors_are_bad_requests(body):
    with pytest.raises(server.BadRequest):
        server.normalize_request(body, CFG)


def test_job_status_does_not_leak_the_image_bytes():
    params = server.normalize_request({"prompt": "x", "image": _png()}, CFG)
    public = server.Job(id="video_1", params=params).public("LTX-2.5")
    assert isinstance(public["image"], str) and "bytes" in public["image"]
    import json

    json.dumps(public)


# ---- latent-frame conditioning ----------------------------------------------
def test_condition_latent_frame_pins_the_first_frame():
    import mlx.core as mx

    f, h, w, c = 3, 2, 2, 4
    per = h * w
    state = sampling.noised_state(
        (1, f * per, c),
        sampling.video_positions(f, h, w, 24.0),
        0,
        tokens_per_frame=per,
    )
    before = np.array(state.latent)
    latent = mx.arange(c * per, dtype=mx.float32).reshape(1, c, 1, h, w)
    tokens = np.array(sampling.patchify(latent))

    out = sampling.condition_latent_frame(state, latent)
    assert out.latent.shape == state.latent.shape
    assert not out.uniform and state.uniform
    mask = np.array(out.denoise_mask)[0, :, 0]
    assert np.array_equal(mask[:per], np.zeros(per))
    assert np.array_equal(mask[per:], np.ones((f - 1) * per))
    clean = np.array(out.clean)
    assert np.array_equal(clean[:, :per], tokens)
    assert np.array_equal(clean[:, per:], np.zeros((1, (f - 1) * per, c)))
    after = np.array(out.latent)
    assert np.array_equal(after[:, :per], tokens)  # the trajectory starts clean
    assert np.array_equal(after[:, per:], before[:, per:])  # the rest is untouched
    assert out.keyframes_mask is state.keyframes_mask
    assert out.positions is state.positions

    # Partial strength: mask 1 - s, latent lerped towards the clean frame.
    half = sampling.condition_latent_frame(state, latent, strength=0.25)
    assert np.allclose(np.array(half.denoise_mask)[0, :per, 0], 0.75)
    expect = 0.75 * before[:, :per] + 0.25 * tokens
    assert np.allclose(np.array(half.latent)[:, :per], expect, atol=1e-6)

    # latent_idx places the frame later in the clip.
    later = sampling.condition_latent_frame(state, latent, latent_idx=2)
    assert np.array_equal(np.array(later.clean)[:, 2 * per :], tokens)
    assert np.array_equal(
        np.array(later.denoise_mask)[0, : 2 * per, 0], np.ones(2 * per)
    )
    with pytest.raises(ValueError):
        sampling.condition_latent_frame(state, latent, latent_idx=3)


def test_conditioned_tokens_survive_the_samplers():
    """blend() keeps the pinned frame fixed through every step."""
    import mlx.core as mx

    f, h, w, c = 2, 2, 2, 4
    per = h * w
    video = sampling.noised_state(
        (1, f * per, c),
        sampling.video_positions(f, h, w, 24.0),
        1,
        tokens_per_frame=per,
    )
    latent = mx.ones((1, c, 1, h, w)) * 3.0
    video = sampling.condition_latent_frame(video, latent)
    audio = sampling.noised_state((1, 5, c), sampling.audio_positions(5), 2)

    def denoise(video, audio, vx, ax, sigma, step=None):
        return mx.zeros_like(vx), mx.zeros_like(ax)

    vx, _ = sampling.euler_ancestral_loop(
        denoise, video, audio, sampling.DISTILLED_SIGMAS, noise_seed=3
    )
    assert np.allclose(np.array(vx)[:, :per], 3.0)
    assert np.allclose(np.array(vx)[:, per:], 0.0)
    vx, _ = sampling.euler_loop(denoise, video, audio, [1.0, 0.5, 0.0])
    assert np.allclose(np.array(vx)[:, :per], 3.0)
