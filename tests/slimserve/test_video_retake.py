# SPDX-License-Identifier: Apache-2.0
"""The retake pipeline's pieces that need no weights: the temporal region mask
(upstream TemporalRegionMask), torchaudio's sinc resampler, the source clip
probe / decode through ffmpeg, and the request parsing. The encoders are
gated against upstream by perf/ltx25_harness/n12_encoders_parity.py."""

import shutil
import subprocess

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import media, sampling  # noqa: E402


def test_time_bounds_follow_the_causal_grids():
    s, e = sampling.video_time_bounds(3, 24.0)
    assert np.allclose(s, [0.0, 1 / 24, 9 / 24]) and np.allclose(
        e, [1 / 24, 9 / 24, 17 / 24]
    )
    s, e = sampling.audio_time_bounds(3)
    assert np.allclose(s, [0.0, 0.01, 0.05]) and np.allclose(e, [0.01, 0.05, 0.09])


def test_region_mask_marks_overlapping_spans_and_keeps_the_source_outside():
    h, w = 2, 2
    f = 5  # spans 0-1, 1-9, 9-17, 17-25, 25-33 frames at 24 fps
    source = mx.full((1, f * h * w, 4), 7.0)
    state = sampling.noised_state(
        (1, f * h * w, 4),
        sampling.video_positions(f, h, w, 24.0),
        0,
        initial=source,
        tokens_per_frame=h * w,
    )
    out = sampling.region_mask(state, 0.5, 1.0, 24.0, h * w)
    mask = np.array(out.denoise_mask[0, :, 0]).reshape(f, h * w)[:, 0]
    # 0.5 s = frame 12 (span 9-17 f -> 0.375-0.708 s), 1.0 s = frame 24 (17-25)
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]
    latent = np.array(out.latent[0]).reshape(f, h * w, 4)
    assert np.allclose(latent[[0, 1, 4]], 7.0)  # the source outside
    assert not np.allclose(latent[2], 7.0)  # noise inside
    audio = sampling.noised_state((1, 40, 4), sampling.audio_positions(40), 1)
    am = np.array(
        sampling.region_mask(audio, 0.5, 1.0, 24.0, None).denoise_mask[0, :, 0]
    )
    s, e = sampling.audio_time_bounds(40)
    assert am.tolist() == ((e > 0.5) & (s < 1.0)).astype(float).tolist()
    assert am[:12].sum() == 0 and am[13:25].sum() == 12  # 0.5 s ~ token 12.5


def test_sinc_resample_is_torchaudios_kernel():
    t = np.arange(48000) / 48000.0
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)[None]
    out = media.resample_sinc(sine, 48000, 16000)
    assert out.shape == (1, 16000)  # ceil(new * len / orig)
    t16 = np.arange(16000) / 16000.0
    expect = np.sin(2 * np.pi * 440.0 * t16)
    err = np.abs(out[0, 100:-100] - expect[100:-100]).max()
    assert err < 2e-3  # the 6-zero-crossing Hann sinc at rolloff 0.99
    assert media.resample_sinc(sine, 16000, 16000) is not None
    up = media.resample_sinc(sine[:, :1600], 16000, 48000)
    assert up.shape == (1, 4800)


def _clip(path, frames=17, size=(96, 64), audio=True):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg not on PATH")
    args = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size[0]}x{size[1]}:rate=24:duration={frames / 24:.4f}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / 24:.4f}"]
    args += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        args += ["-c:a", "aac"]
    args += [str(path)]
    subprocess.run(args, check=True)
    return str(path)


def test_probe_and_decode_through_ffmpeg(tmp_path):
    clip = _clip(tmp_path / "c.mp4")
    info = media.probe(clip)
    assert (info.frames, info.width, info.height, info.fps) == (17, 96, 64, 24.0)
    assert info.has_audio and info.audio_rate == 44100 and info.audio_channels == 1
    frames = media.read_frames(clip, info)
    assert frames.shape == (17, 64, 96, 3) and frames.dtype == np.uint8
    head = media.read_frames(clip, info, 0.0, 9 / 24)
    assert head.shape[0] == 9 and np.array_equal(head, frames[:9])
    audio = media.read_audio(clip, info)
    assert audio is not None
    wav, rate = audio
    assert rate == 44100 and wav.shape[0] == 1 and wav.dtype == np.float32
    assert 0.05 < np.abs(wav).max() <= 1.0  # lavfi sine is -20 dB-ish after AAC
    silent = _clip(tmp_path / "s.mp4", audio=False)
    info = media.probe(silent)
    assert not info.has_audio and media.read_audio(silent, info) is None


CFG = {
    "pipeline": "retake",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def test_retake_requests_take_the_source_clips_geometry(tmp_path):
    clip = _clip(tmp_path / "c.mp4")
    params = server.normalize_request(
        {"prompt": "x", "video_path": clip, "start_time": 0.2, "end_time": 0.5},
        CFG,
    )
    assert params["video_path"] == clip
    assert (params["start_time"], params["end_time"]) == (0.2, 0.5)
    assert params["source"] == {
        "width": 96,
        "height": 64,
        "num_frames": 17,
        "fps": 24.0,
    }
    assert "width" not in params and "num_frames" not in params
    with pytest.raises(server.BadRequest, match="does not apply"):
        server.normalize_request(
            {
                "prompt": "x",
                "video_path": clip,
                "start_time": 0,
                "end_time": 1,
                "size": "768x512",
            },
            CFG,
        )
    with pytest.raises(server.BadRequest, match="start_time < end_time"):
        server.normalize_request(
            {"prompt": "x", "video_path": clip, "start_time": 0.5, "end_time": 0.2}, CFG
        )
    with pytest.raises(server.BadRequest, match="inside the clip"):
        server.normalize_request(
            {"prompt": "x", "video_path": clip, "start_time": 5.0, "end_time": 6.0}, CFG
        )
    with pytest.raises(server.BadRequest, match="8k \\+ 1"):
        bad = _clip(tmp_path / "b.mp4", frames=16)
        server.normalize_request(
            {"prompt": "x", "video_path": bad, "start_time": 0, "end_time": 0.5}, CFG
        )
    with pytest.raises(server.BadRequest, match="multiples of 32"):
        odd = _clip(tmp_path / "o.mp4", size=(100, 64))
        server.normalize_request(
            {"prompt": "x", "video_path": odd, "start_time": 0, "end_time": 0.5}, CFG
        )
    with pytest.raises(server.BadRequest, match="needs `video_path`"):
        server.normalize_request({"prompt": "x", "start_time": 0, "end_time": 1}, CFG)
    with pytest.raises(server.BadRequest, match="validated up to"):
        server.normalize_request(
            {"prompt": "x", "video_path": clip, "start_time": 0, "end_time": 0.5},
            {**CFG, "max_video_tokens": 5},
        )
    flags = server.normalize_request(
        {
            "prompt": "x",
            "video_path": clip,
            "start_time": 0,
            "end_time": 0.5,
            "regenerate_audio": False,
        },
        CFG,
    )
    assert flags["regenerate_audio"] is False and "regenerate_video" not in flags


def test_frames_by_index_are_exact_windows_and_passthrough_keeps_the_count(tmp_path):
    # a clip with a one-frame timestamp gap (the fox clip had four, 246
    # frames decoded for 241): ffmpeg's default constant-rate output pads
    # them with duplicates and a pts window can come back a frame short; the
    # readers take the decoded frames as they are, the index reader exactly
    clip = tmp_path / "v.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=24:duration=2",
            "-vf",
            "setpts=PTS+gte(N\\,20)/24/TB",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "41",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    info = media.probe(clip)
    assert info.frames == 41 and 23.0 < info.fps < 24.0
    full = media.read_frames(clip, info)
    assert full.shape[0] == 41
    window = media.read_frames_by_index(clip, info, 8, 17)
    assert window.shape[0] == 17 and np.array_equal(window, full[8:25])
    assert media.read_frames_by_index(clip, info, 32).shape[0] == 9
    assert media.read_frames_by_index(clip, info, 32, 17).shape[0] == 9  # short
    with pytest.raises(ValueError):
        media.read_frames_by_index(clip, info, -1)


def test_a2vid_and_dubit_requests(tmp_path):
    clip = _clip(tmp_path / "c.mp4")
    a2v = {**CFG, "pipeline": "a2vid", "width": 768, "height": 512}
    params = server.normalize_request(
        {"prompt": "x", "audio_path": clip, "audio_start_time": 0.1}, a2v
    )
    assert params["audio_path"] == clip and params["audio_start_time"] == 0.1
    assert params["num_frames"] is None  # the audio decides
    assert params["max_num_frames"] == 505  # the envelope at 768x512
    with pytest.raises(server.BadRequest, match="exclusive"):
        server.normalize_request(
            {"prompt": "x", "audio_path": clip, "audio_max_duration": 2, "seconds": 2},
            a2v,
        )
    with pytest.raises(server.BadRequest, match="a2vid pipeline only"):
        server.normalize_request(
            {"prompt": "x", "audio_path": clip}, {**CFG, "pipeline": "dev"}
        )
    lora = tmp_path / "d.safetensors"
    lora.write_bytes(b"x")
    dub = {**CFG, "pipeline": "dubit"}
    params = server.normalize_request(
        {
            "prompt": "x",
            "reference_video": clip,
            "loras": [{"path": str(lora)}],
            "size": "768x512",
            "reference_strength": 0.9,
        },
        dub,
    )
    assert params["reference_video"] == clip and params["reference_strength"] == 0.9
    assert (params["width"], params["height"]) == (768, 512)
    assert params["source"]["num_frames"] == 17 and params["loras"] == [
        (str(lora), 1.0)
    ]
    with pytest.raises(server.BadRequest, match="exactly one adapter"):
        server.normalize_request({"prompt": "x", "reference_video": clip}, dub)
    with pytest.raises(server.BadRequest, match="does not apply to Dub-It"):
        server.normalize_request(
            {
                "prompt": "x",
                "reference_video": clip,
                "loras": [{"path": str(lora)}],
                "seconds": 2,
            },
            dub,
        )
    silent = _clip(tmp_path / "s.mp4", audio=False)
    with pytest.raises(server.BadRequest, match="no audio stream"):
        server.normalize_request(
            {"prompt": "x", "reference_video": silent, "loras": [{"path": str(lora)}]},
            dub,
        )
    # chunked Dub-It: the window layout rides along, the clip's own length
    params = server.normalize_request(
        {
            "prompt": "x",
            "reference_video": clip,
            "loras": [{"path": str(lora)}],
            "chunked": True,
            "chunk_carry_frames": 17,
        },
        dub,
    )
    assert params["chunk"].carry_frames == 17 and params["source"]["num_frames"] == 17


def test_audio_reference_tokens_sit_before_the_timeline():
    state = sampling.noised_state((1, 10, 4), sampling.audio_positions(10), 0)
    ref = mx.ones((1, 5, 4))
    out = sampling.append_audio_reference(state, ref)
    pos = np.array(out.positions[0, 10:, 0])
    s, e = sampling.audio_time_bounds(5)
    assert np.allclose(pos, (s + e) / 2 - e.max() - 0.04)
    assert pos.max() < 0 and np.allclose(np.array(out.denoise_mask[0, 10:, 0]), 0.0)
    assert np.allclose(np.array(out.clean[0, 10:]), 1.0)
