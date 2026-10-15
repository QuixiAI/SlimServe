# SPDX-License-Identifier: Apache-2.0
"""Stills at any pixel frame (upstream's repeatable --image PATH FRAME_IDX
STRENGTH [CRF]): the conditioning geometry against values printed by
ltx_core.conditioning.VideoConditionByKeyframeIndex and
ltx_pipelines.dfr_helpers.ops.rebase_image_conditionings (Lightricks/LTX-2
9ec55f9), the request parsing, and the CLI's --image groups."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import sampling  # noqa: E402
from slimserve.video.ltx25.sampling import Still  # noqa: E402

H, W, FPS = 2, 3, 24.0


def _state(f=3, c=4):
    return sampling.noised_state(
        (1, f * H * W, c),
        sampling.video_positions(f, H, W, FPS),
        0,
        tokens_per_frame=H * W,
    )


def _plane(value, c=4):
    return mx.full((1, c, 1, H, W), value)


def test_keyframe_stills_are_appended_with_upstreams_single_frame_positions():
    """VideoConditionByKeyframeIndex at frames 0 / 5 / 16 (24 fps): temporal
    midpoints 0.0208 / 0.2292 / 0.6875, the spatial grid 16 + 32 i, denoise mask
    1 - strength, clean = the plane, the noisy latent zeros at strength 1."""
    state = _state()
    n = state.latent.shape[1]
    out = sampling.condition_stills(
        state,
        [(_plane(1.0), 5, 0.7), (_plane(2.0), 16, 1.0)],
        FPS,
        1.0,
        0,
    )
    assert out.latent.shape[1] == n + 2 * H * W
    pos = np.array(out.positions[0, n:])
    assert np.allclose(pos[: H * W, 0], (5 + 0.5) / 24)  # 0.22916
    assert np.allclose(pos[H * W :, 0], (16 + 0.5) / 24)  # 0.6875
    assert pos[:, 1].tolist() == [16.0, 16.0, 16.0, 48.0, 48.0, 48.0] * 2
    assert pos[:, 2].tolist() == [16.0, 48.0, 80.0, 16.0, 48.0, 80.0] * 2
    mask = np.array(out.denoise_mask[0, n:, 0])
    assert np.allclose(mask[: H * W], 0.3) and np.allclose(mask[H * W :], 0.0)
    assert np.allclose(np.array(out.clean[0, n : n + H * W]), 1.0)
    assert np.allclose(np.array(out.clean[0, n + H * W :]), 2.0)
    # strength 1: the appended latent is the clean plane; 0.7: 0.7 plane + noise
    assert np.allclose(np.array(out.latent[0, n + H * W :]), 2.0)
    assert not np.allclose(np.array(out.latent[0, n : n + H * W]), 0.7)
    assert np.allclose(np.array(out.keyframes_mask[0, n:, 0]), 0.0)  # unmarked
    assert np.allclose(np.array(out.keyframes_mask[0, : H * W, 0]), 1.0)


def test_a_frame_zero_still_replaces_latent_frame_zero_unless_append_all():
    state = _state()
    n = state.latent.shape[1]
    out = sampling.condition_stills(state, [(_plane(3.0), 0, 1.0)], FPS, 1.0, 0)
    assert out.latent.shape[1] == n
    assert np.allclose(np.array(out.clean[0, : H * W]), 3.0)
    assert np.allclose(np.array(out.denoise_mask[0, : H * W, 0]), 0.0)
    assert np.allclose(np.array(out.latent[0, : H * W]), 3.0)
    # the keyframe interpolation pipeline appends frame 0 too, at [0, 1) / fps
    kf = sampling.condition_stills(
        state, [(_plane(3.0), 0, 1.0)], FPS, 1.0, 0, append_all=True
    )
    assert kf.latent.shape[1] == n + H * W
    assert np.allclose(np.array(kf.positions[0, n:, 0]), 0.5 / 24)  # 0.020833
    assert np.allclose(np.array(kf.denoise_mask[0, : H * W, 0]), 1.0)


def test_rebase_matches_upstreams_rebase_image_conditionings():
    imgs = [("a", 0, 1.0), ("b", 12, 0.8), ("c", 48, 1.0), ("d", 120, 0.5)]
    key = lambda items: [(f, s) for _, f, s in items]  # noqa: E731
    assert key(sampling.rebase_stills(imgs, 2)) == [
        (0, 1.0),
        (24, 0.8),
        (96, 1.0),
        (240, 0.5),
    ]
    assert key(sampling.rebase_stills(imgs, 2, 24, 96)) == [(0, 0.8), (72, 1.0)]
    assert key(sampling.rebase_stills(imgs, 4, 48, 192)) == [(0, 0.8), (144, 1.0)]
    # the epilogue's extra filter: nothing before the window's resume pixel
    assert key(sampling.rebase_stills(imgs, 2, 24, 96, resume=25)) == [(72, 1.0)]


CFG = {
    "pipeline": "distilled",
    "width": 768,
    "height": 512,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def _png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_images_requests_become_stills_in_order():
    import base64

    png = _png()
    b64 = base64.b64encode(png).decode()
    params = server.normalize_request(
        {
            "prompt": "x",
            "seconds": 2,
            "images": [
                {"image": b64, "frame": 24, "strength": 0.6},
                {"image": f"data:image/png;base64,{b64}", "frame": 48, "crf": 0},
            ],
        },
        CFG,
    )
    stills = params["images"]
    assert [(s.frame, s.strength, s.crf) for s in stills] == [
        (24, 0.6, None),
        (48, 1.0, 0),
    ]
    assert stills[0].image == png and isinstance(stills[0], Still)
    assert "image" not in params
    with pytest.raises(server.BadRequest, match="outside the clip"):
        server.normalize_request(
            {"prompt": "x", "seconds": 2, "images": [{"image": b64, "frame": 49}]},
            CFG,
        )
    with pytest.raises(server.BadRequest, match="frame"):
        server.normalize_request(
            {"prompt": "x", "images": [{"image": b64, "frame": -1}]}, CFG
        )
    with pytest.raises(server.BadRequest, match="crf"):
        server.normalize_request(
            {"prompt": "x", "images": [{"image": b64, "crf": 99}]}, CFG
        )
    with pytest.raises(server.BadRequest, match="strength"):
        server.normalize_request(
            {"prompt": "x", "images": [{"image": b64, "strength": 2}]}, CFG
        )
    with pytest.raises(server.BadRequest, match="non-empty list"):
        server.normalize_request({"prompt": "x", "images": []}, CFG)


def test_the_keyframe_pipeline_requires_stills_and_takes_the_shorthand():
    cfg = {**CFG, "pipeline": "keyframes"}
    with pytest.raises(server.BadRequest, match="needs `images`"):
        server.normalize_request({"prompt": "x"}, cfg)
    png = _png()
    params = server.normalize_request(
        {"prompt": "x", "image": png, "image_strength": 0.5}, cfg
    )
    assert "image" not in params and "image_strength" not in params
    assert [(s.frame, s.strength) for s in params["images"]] == [(0, 0.5)]
    # a negative prompt is a guided-pipeline option: keyframes and one_stage
    assert "negative_prompt" in server.normalize_request(
        {"prompt": "x", "negative_prompt": "blur"}, {**cfg, "pipeline": "one_stage"}
    )


def test_cli_image_groups_follow_upstreams_path_frame_strength_crf(tmp_path):
    from types import SimpleNamespace

    from slimserve.video import cli

    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(_png())
    b.write_bytes(_png())
    body = {}
    cli._add_images(body, SimpleNamespace(image=[[str(a)]], image_strength=0.8))
    assert body["image"] == a.read_bytes() and body["image_strength"] == 0.8
    body = {}
    cli._add_images(
        body,
        SimpleNamespace(
            image=[[str(a), "0", "0.9"], [str(b), "48"]], image_strength=None
        ),
    )
    assert "image" not in body
    assert [(s["frame"], s["strength"], s["crf"]) for s in body["images"]] == [
        (0, 0.9, None),
        (48, 1.0, None),
    ]
    body = {}
    cli._add_images(
        body, SimpleNamespace(image=[[str(a), "24", "1", "0"]], image_strength=None)
    )
    assert body["images"][0]["crf"] == 0 and body["images"][0]["frame"] == 24
    with pytest.raises(ValueError, match="not both"):
        cli._add_images(
            {}, SimpleNamespace(image=[[str(a), "0", "0.9"]], image_strength=0.5)
        )
    with pytest.raises(ValueError, match="needs --image"):
        cli._add_images({}, SimpleNamespace(image=None, image_strength=0.5))
    with pytest.raises(ValueError, match="cannot read"):
        cli._add_images(
            {}, SimpleNamespace(image=[[str(tmp_path / "no.png")]], image_strength=None)
        )


def test_generated_keyframe_frames_match_upstreams_spacing():
    """evenly_spaced_keyframe_positions on Lightricks/LTX-2 9ec55f9."""
    from slimserve.video.ltx25.pipeline import LTX25Engine

    f = LTX25Engine._slot_frames
    assert f(1, 49) == [24] and f(2, 49) == [16, 32]
    assert f(3, 121) == [30, 60, 90] and f(5, 121) == [20, 40, 60, 80, 100]
    assert f(4, 33) == [6, 13, 19, 26]
    assert f(0, 49) == [] and f([40, 8, 8], 49) == [8, 40]
    with pytest.raises(ValueError):
        f(2, 3)
    with pytest.raises(ValueError):
        f([49], 49)
    params = server.normalize_request(
        {
            "prompt": "x",
            "seconds": 2,
            "generated_keyframes": 2,
            "decode_with_keyframes": True,
        },
        CFG,
    )
    assert params["generated_keyframes"] == 2 and params["decode_with_keyframes"]
    with pytest.raises(server.BadRequest, match="needs generated_keyframes"):
        server.normalize_request({"prompt": "x", "decode_with_keyframes": True}, CFG)
    with pytest.raises(server.BadRequest, match="applies to"):
        server.normalize_request(
            {"prompt": "x", "generated_keyframes": 2}, {**CFG, "pipeline": "dfr"}
        )
