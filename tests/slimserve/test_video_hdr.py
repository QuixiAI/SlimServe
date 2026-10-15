# SPDX-License-Identifier: Apache-2.0
"""HDR: the ACEScct working space, primaries, HLG planes and the sRGB EOTF
against values computed by ltx_core.hdr / color on Lightricks/LTX-2 9ec55f9
(tests/slimserve/data/ltx25_hdr_ref.npz), the EXR round trip, and the
request parsing."""

from pathlib import Path

import numpy as np
import pytest

from slimserve.video import server
from slimserve.video.ltx25 import hdr

REF = np.load(Path(__file__).parent / "data" / "ltx25_hdr_ref.npz")


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_working_space_and_back_match_upstream():
    assert _rel(hdr.to_working_space(REF["hdr"], "srgb_linear"), REF["ws_709"]) < 1e-6
    assert _rel(hdr.to_working_space(REF["hdr"], "acescg"), REF["ws_ap1"]) < 1e-6
    assert _rel(hdr.to_linear(REF["ws_709"], "rec709"), REF["lin_back"]) < 1e-6
    assert _rel(hdr.to_linear(REF["ws_709"], "ap1"), REF["lin_ap1"]) < 1e-6
    codes = hdr.to_working_space(REF["ws_709"], "acescct")
    assert np.array_equal(codes, np.clip(REF["ws_709"], 0, 1))
    ramp = np.linspace(0, 1, 50, dtype=np.float32)
    assert _rel(hdr.srgb_eotf_to_linear(ramp), REF["eotf"]) < 1e-6


def test_hlg_planes_match_upstream_bit_for_bit():
    y, u, v = hdr.linear_to_hlg_planes(REF["hdr"])
    assert y.dtype == np.uint16 and y.shape == (2, 8, 12) and u.shape == (2, 4, 6)
    assert np.array_equal(y.astype(np.int32), REF["y"])
    assert np.array_equal(u.astype(np.int32), REF["u"])
    assert np.array_equal(v.astype(np.int32), REF["v"])
    assert y.min() >= 64 and y.max() <= 940  # limited range, 10 bit


def test_exr_round_trip_with_tags(tmp_path):
    pytest.importorskip("OpenEXR")
    rgb = (np.random.default_rng(0).random((6, 8, 3)) * 10).astype(np.float32)
    hdr.write_exr(tmp_path / "f.exr", rgb, "ap1", "ACEScg")
    back = hdr.read_exr(tmp_path / "f.exr")
    assert back.shape == (6, 8, 3) and back.dtype == np.float32
    assert np.abs(back - rgb).max() < 0.01  # half floats
    assert hdr.is_exr_dir(tmp_path) and len(hdr.exr_paths(tmp_path)) == 1
    assert not hdr.is_exr_dir(tmp_path / "nope")


def test_hdr_requests(tmp_path):
    pytest.importorskip("OpenEXR")
    cfg = {
        "pipeline": "distilled",
        "width": 1536,
        "height": 1024,
        "num_frames": 121,
        "fps": 24.0,
        "max_video_tokens": 24576,
    }
    params = server.normalize_request(
        {"prompt": "x", "hdr": "ACEScg", "seconds": 2}, cfg
    )
    assert params["hdr"] == "acescg"
    with pytest.raises(server.BadRequest, match="hdr must be one of"):
        server.normalize_request({"prompt": "x", "hdr": "rec2020"}, cfg)
    still = tmp_path / "a.exr"
    hdr.write_exr(still, np.ones((8, 8, 3), np.float32), "rec709", "sRGB")
    with pytest.raises(server.BadRequest, match="EXR inputs need"):
        server.normalize_request(
            {"prompt": "x", "images": [{"path": str(still), "frame": 8}]}, cfg
        )
    params = server.normalize_request(
        {
            "prompt": "x",
            "hdr": "srgb_linear",
            "images": [{"path": str(still), "frame": 8}],
        },
        cfg,
    )
    assert params["images"][0].image == str(still) and params["hdr"] == "srgb_linear"
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(9):
        hdr.write_exr(
            folder / f"f{i:03d}.exr",
            np.full((64, 96, 3), 0.5, np.float32),
            "ap1",
            "ACEScg",
        )
    retake = {**cfg, "pipeline": "retake"}
    with pytest.raises(server.BadRequest, match="frame rate"):
        server.normalize_request(
            {
                "prompt": "x",
                "video_path": str(folder),
                "start_time": 0,
                "end_time": 0.2,
                "hdr": "acescg",
            },
            retake,
        )
    params = server.normalize_request(
        {
            "prompt": "x",
            "video_path": str(folder),
            "start_time": 0,
            "end_time": 0.2,
            "hdr": "acescg",
            "fps": 24,
        },
        retake,
    )
    assert params["source"] == {"width": 96, "height": 64, "num_frames": 9, "fps": 24.0}
    assert params["fps"] == 24.0 and params["hdr"] == "acescg"


def test_hdr_ic_lora_requests(tmp_path):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    clip = tmp_path / "c.mp4"
    src = "testsrc=size=100x60:rate=24:duration=0.7"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            src,
            "-frames:v",
            "17",
        ]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
        check=True,
    )
    lora = tmp_path / "hdr.safetensors"
    lora.write_bytes(b"x")
    emb = tmp_path / "scene.safetensors"
    emb.write_bytes(b"x")
    cfg = {
        "pipeline": "hdr_ic_lora",
        "width": 1536,
        "height": 1024,
        "num_frames": 121,
        "fps": 24.0,
        "max_video_tokens": 24576,
    }
    params = server.normalize_request(
        {
            "video_path": str(clip),
            "loras": [{"path": str(lora)}],
            "text_embeddings": str(emb),
            "high_quality": True,
        },
        cfg,
    )
    assert "prompt" not in params and params["input_colorspace"] == "srgb_gamma"
    assert params["exr_colorspace"] == "acescg" and params["high_quality"] is True
    assert params["source"] == {
        "width": 100,
        "height": 60,
        "num_frames": 17,
        "fps": 24.0,
    }
    with pytest.raises(server.BadRequest, match="exactly one adapter"):
        server.normalize_request(
            {"video_path": str(clip), "text_embeddings": str(emb)}, cfg
        )
    with pytest.raises(server.BadRequest, match="text_embeddings"):
        server.normalize_request(
            {"video_path": str(clip), "loras": [{"path": str(lora)}]}, cfg
        )
    with pytest.raises(server.BadRequest, match="input_colorspace"):
        server.normalize_request(
            {
                "video_path": str(clip),
                "loras": [{"path": str(lora)}],
                "text_embeddings": str(emb),
                "input_colorspace": "rec2020",
            },
            cfg,
        )
    with pytest.raises(server.BadRequest, match="reference tokens"):
        server.normalize_request(
            {
                "video_path": str(clip),
                "loras": [{"path": str(lora)}],
                "text_embeddings": str(emb),
            },
            {**cfg, "max_video_tokens": 10},
        )
