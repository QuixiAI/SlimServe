# SPDX-License-Identifier: Apache-2.0
"""IC-LoRA conditioning pieces against values printed by upstream
(ltx_core.tiling.split_by_size_pinned, conditioning.mask_utils
build_attention_mask, iclora_utils.downsample_mask_video_to_latent on
Lightricks/LTX-2 9ec55f9), plus the reference token geometry and the request
parsing."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import sampling  # noqa: E402


@pytest.mark.parametrize(
    "dim,size,overlap,expect",
    [
        (48, 32, 16, [(0, 32, 0, 16), (16, 48, 16, 0)]),
        (33, 32, 16, [(0, 32, 0, 31), (1, 33, 31, 0)]),
        (24, 32, 16, [(0, 24, 0, 0)]),
        (
            100,
            32,
            16,
            [
                (0, 32, 0, 15),
                (17, 49, 15, 15),
                (34, 66, 15, 15),
                (51, 83, 15, 15),
                (68, 100, 15, 0),
            ],
        ),
        (40, 16, 8, [(0, 16, 0, 8), (8, 24, 8, 8), (16, 32, 8, 8), (24, 40, 8, 0)]),
        (96, 32, 0, [(0, 32, 0, 0), (32, 64, 0, 0), (64, 96, 0, 0)]),
    ],
)
def test_pinned_split_matches_upstream(dim, size, overlap, expect):
    assert sampling.split_by_size_pinned(dim, size, overlap) == expect


def test_attention_bias_is_upstreams_block_mask_in_log_space():
    # build_attention_mask(None, noisy 3, new 2, existing 4, cross [0.5, 1])
    bias = sampling.attention_bias(None, 4, 2, 3, np.array([0.5, 1.0]))
    w = np.exp(np.array(bias[0], dtype=np.float32))
    expect = np.array(
        [
            [1, 1, 1, 1, 0.5, 1],
            [1, 1, 1, 1, 0.5, 1],
            [1, 1, 1, 1, 0.5, 1],
            [1, 1, 1, 1, 0, 0],
            [0.5, 0.5, 0.5, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ]
    )
    assert np.allclose(w, expect, atol=1e-3)
    # a second block of 1 token at cross 0.25 over the existing mask
    bias2 = sampling.attention_bias(bias, 6, 1, 3, 0.25)
    w2 = np.exp(np.array(bias2[0], dtype=np.float32))
    expect2 = np.array(
        [
            [1, 1, 1, 1, 0.5, 1, 0.25],
            [1, 1, 1, 1, 0.5, 1, 0.25],
            [1, 1, 1, 1, 0.5, 1, 0.25],
            [1, 1, 1, 1, 0, 0, 0],
            [0.5, 0.5, 0.5, 0, 1, 1, 0],
            [1, 1, 1, 0, 1, 1, 0],
            [0.25, 0.25, 0.25, 0, 0, 0, 1],
        ]
    )
    assert np.allclose(w2, expect2, atol=1e-3)
    assert bias.dtype == mx.float16 and np.isneginf(np.array(bias[0, 3, 4]))


def test_mask_video_downsampling_matches_upstream():
    mask = np.arange(9 * 64 * 96, dtype=np.float32).reshape(9, 64, 96) / (9 * 64 * 96)
    out = sampling.mask_video_to_tokens(mask, 2, 2, 3)
    expect = [
        0.02719, 0.027769, 0.028347, 0.082746, 0.083324, 0.083903,
        0.52719, 0.527769, 0.528347, 0.582746, 0.583324, 0.583903,
    ]  # fmt: skip
    assert np.allclose(out, expect, atol=2e-6)


def test_reference_tokens_scale_space_and_time_as_upstream():
    h, w, fps = 2, 2, 24.0
    state = sampling.noised_state(
        (1, 3 * h * w, 4),
        sampling.video_positions(3, h, w, fps),
        0,
        tokens_per_frame=h * w,
    )
    n = state.latent.shape[1]
    ref = mx.ones((1, 2 * 1 * 1, 4))
    out = sampling.append_reference(
        state,
        ref,
        sampling.video_positions(2, 1, 1, fps),
        downscale=2,
        strength=1.0,
        temporal_scale=4,
        fps=fps,
        attention=0.5,
        num_noisy=n,
    )
    pos = np.array(out.positions[0, n:])
    # spatial midpoints 16 -> 32 (x2); times (0.5 / 24, 5 / 24) spread by 4
    # and shifted back by 3 / 24: (max(0, 2/24 - 3/24), 20/24 - 3/24)
    assert np.allclose(pos[:, 1:], 32.0)
    assert np.allclose(pos[:, 0], [0.0, 17 / 24])
    assert np.allclose(np.array(out.denoise_mask[0, n:, 0]), 0.0)
    assert out.attention_mask.shape == (1, n + 2, n + 2)
    assert np.allclose(
        np.exp(np.array(out.attention_mask[0, :n, n:], dtype=np.float32)),
        0.5,
        atol=1e-3,
    )
    plain = sampling.append_reference(
        state, ref, sampling.video_positions(2, 1, 1, fps), 1
    )
    assert plain.attention_mask is None


def test_pinned_tiles_cover_the_grid_with_complementary_blends():
    tiles = sampling.spatial_tiles_by_size(48, 100, 32, 32, 16, 16)
    assert len(tiles) == 2 * 5
    total = np.zeros((48, 100), dtype=np.float32)
    for t in tiles:
        total[t.h0 : t.h1, t.w0 : t.w1] += t.weights
    assert np.allclose(total, 1.0, atol=1e-5)


CFG = {
    "pipeline": "ic_lora",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def test_ic_lora_requests_need_references_and_adapters(tmp_path):
    ref = tmp_path / "ref.mp4"
    ref.write_bytes(b"x")
    lora = tmp_path / "a.safetensors"
    lora.write_bytes(b"x")
    with pytest.raises(server.BadRequest, match="video_conditioning"):
        server.normalize_request({"prompt": "x"}, CFG)
    with pytest.raises(server.BadRequest, match="loras"):
        server.normalize_request(
            {"prompt": "x", "video_conditioning": [{"path": str(ref)}]}, CFG
        )
    params = server.normalize_request(
        {
            "prompt": "x",
            "video_conditioning": [{"path": str(ref), "strength": 0.8}],
            "loras": [{"path": str(lora), "strength": 0.9}],
            "attention_strength": 0.5,
            "attention_mask": str(ref),
            "tile": True,
            "tile_height": 512,
            "stage_2_ic_lora": True,
        },
        CFG,
    )
    assert params["video_conditioning"] == [(str(ref), 0.8)]
    assert params["loras"] == [(str(lora), 0.9)]
    assert params["attention_strength"] == 0.5 and params["attention_mask"] == str(ref)
    assert (
        params["tile"] and params["tile_height"] == 512 and "tile_width" not in params
    )
    assert params["stage_2_ic_lora"] and "skip_stage_2" not in params
    with pytest.raises(server.BadRequest, match="multiple of 32"):
        server.normalize_request(
            {
                "prompt": "x",
                "video_conditioning": [{"path": str(ref)}],
                "loras": [{"path": str(lora)}],
                "tile_height": 100,
            },
            CFG,
        )
    with pytest.raises(server.BadRequest, match="ic_lora pipeline only"):
        server.normalize_request(
            {"prompt": "x", "tile": True}, {**CFG, "pipeline": "distilled"}
        )
