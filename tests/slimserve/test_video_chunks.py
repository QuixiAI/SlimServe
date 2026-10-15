# SPDX-License-Identifier: Apache-2.0
"""Chunked long clips: the window layout and keyframe planning against values
printed by ltx_pipelines.chunks (layout.py, planner.py) on Lightricks/LTX-2
9ec55f9, the seam blends, and the request parsing."""

import json
from pathlib import Path

import numpy as np
import pytest

from slimserve.video import server
from slimserve.video.ltx25 import chunks as ch

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "ltx25_chunk_layout.json").read_text()
)["cases"]


def _case(key):
    frames, size, carry, blend = key.split(",")
    return int(frames), int(size), int(carry), None if blend == "None" else int(blend)


@pytest.mark.parametrize("key", list(FIXTURE))
def test_layouts_and_keyframe_plans_match_upstream(key):
    frames, size, carry, blend = _case(key)
    plan = ch.layouts(frames, ch.ChunkConfig(size, carry, blend))
    got = [
        [w.pixel_frames, w.start_pixel_frame, w.prev_carry, w.next_carry, w.blend]
        for w in plan
    ]
    assert got == FIXTURE[key]["layouts"]
    assert ch.plan_keyframes(2, frames, plan) == FIXTURE[key]["kf2"]
    if frames > 150:
        assert ch.plan_keyframes([150, 10], frames, plan) == [10, 150]


def test_layout_ownership_and_carry_arithmetic():
    plan = ch.layouts(241, ch.ChunkConfig())
    second = plan[1]
    assert second.owned == (97, 169) and second.local_frame(96) is None
    assert second.local_frame(97) == 25 and second.local_frame(168) == 96
    assert second.latent_frames == 13
    assert ch.carry_audio_frames(25, 24.0) == 26  # round(25 / 24 * 25)
    with pytest.raises(ValueError, match="at least 17"):
        ch.chunk_lengths(241, 97, 9)
    with pytest.raises(ValueError, match="grid"):
        ch.chunk_lengths(241, 96, 25)


def test_audio_window_slices_and_pads():
    full = np.arange(1 * 120 * 3, dtype=np.float32).reshape(1, 120, 3) + 1.0
    plan = ch.layouts(241, ch.ChunkConfig())
    piece = ch.audio_window(full, plan[1], 24.0, 101)  # start round(72 / 24 * 25) = 75
    assert piece.shape == (1, 101, 3)
    assert np.array_equal(piece[0, :10], full[0, 75:85])
    assert (
        np.all(piece[0, 60 - 75 + 0 :] == 0.0) and piece[0, 45 - 1 - 45 + 44].sum() != 0
    )


def test_seam_blends_are_upstreams():
    prev = np.full((4, 2, 2, 3), 100, dtype=np.uint8)
    over = np.full((6, 2, 2, 3), 200, dtype=np.uint8)
    out = ch.crossfade_video(prev, over)
    assert out.shape == prev.shape
    assert out[:, 0, 0, 0].tolist() == [120, 140, 160, 180]  # weights 1/5 .. 4/5
    left = np.ones((2, 10), dtype=np.float32)
    right = np.zeros((2, 10), dtype=np.float32)
    a, b = ch.crossfade_audio(left, right, 4)
    assert a.shape == (2, 10) and b.shape == (2, 6)
    assert np.allclose(a[0, -4:], np.cos(np.linspace(0, 1, 4) * np.pi / 2), atol=1e-6)


def test_chunk_requests():
    cfg = {
        "pipeline": "distilled",
        "width": 1536,
        "height": 1024,
        "num_frames": 121,
        "fps": 24.0,
        "max_video_tokens": 24576,
    }
    params = server.normalize_request(
        {"prompt": "x", "chunked": True, "seconds": 10}, cfg
    )
    assert params["chunk"] == ch.ChunkConfig(97, 25, 25) and params["num_frames"] == 241
    params = server.normalize_request(
        {
            "prompt": "x",
            "chunk_pixel_frames": 121,
            "chunk_blend_frames": 8,
            "num_frames": 505,
            "size": "768x512",
        },
        cfg,
    )
    assert params["chunk"] == ch.ChunkConfig(121, 25, 8)
    with pytest.raises(server.BadRequest, match="larger than this profile's envelope"):
        server.normalize_request({"prompt": "x", "chunk_pixel_frames": 129}, cfg)
    with pytest.raises(server.BadRequest, match="at least 17"):
        server.normalize_request({"prompt": "x", "chunk_carry_frames": 9}, cfg)
    with pytest.raises(server.BadRequest, match="applies to"):
        server.normalize_request(
            {"prompt": "x", "chunked": True}, {**cfg, "pipeline": "dfr"}
        )
    with pytest.raises(server.BadRequest, match="validated up to"):
        server.normalize_request({"prompt": "x", "num_frames": 241}, cfg)  # not chunked
