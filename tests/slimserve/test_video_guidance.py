# SPDX-License-Identifier: Apache-2.0
"""Per-request guidance (upstream MultiModalGuiderParams, incl. skip_step) and
user LoRAs (upstream --lora): the request parsing, the skip bookkeeping in the
guided denoiser, and the adapter loader. The skipped-modality forward itself
is checked against upstream by perf/ltx25_harness/n8_dit_parity.py skip-ref /
skip-ours (ledger section 32)."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import pipeline as pl  # noqa: E402
from slimserve.video.ltx25 import sampling  # noqa: E402

CFG = {
    "pipeline": "dev",
    "width": 768,
    "height": 512,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def test_skip_step_follows_upstreams_modulo_rule():
    g = sampling.Guidance(skip_step=0)
    assert not any(g.skips(i) for i in range(6))
    g = sampling.Guidance(skip_step=1)  # every other step
    assert [g.skips(i) for i in range(6)] == [False, True, False, True, False, True]
    g = sampling.Guidance(skip_step=2)
    assert [g.skips(i) for i in range(6)] == [False, True, True, False, True, True]
    assert not g.skips(None)  # a loop that gives no index never skips


class _FakeDiT:
    """Returns a per-call constant and records the run flags it was given."""

    cfg = SimpleNamespace(num_layers=48)

    def __init__(self):
        self.calls = []
        self.value = 0.0

    def __call__(self, vx, ax, t, *args, run_video=True, run_audio=True, **kw):
        self.calls.append((run_video, run_audio))
        v = mx.full(vx.shape, self.value) if run_video else None
        a = mx.full(ax.shape, self.value + 100.0) if run_audio else None
        return v, a


def _states(n=4, t=3):
    video = sampling.noised_state((1, n, 2), mx.zeros((1, n, 3)), 0, tokens_per_frame=n)
    audio = sampling.noised_state((1, t, 2), mx.zeros((1, t, 1)), 1)
    return video, audio


def test_guided_denoiser_reuses_the_last_x0_on_a_skipped_step():
    dit = _FakeDiT()
    cond = (mx.zeros((1, 2, 4)), mx.zeros((1, 2, 4)))
    den = pl.GuidedDenoiser(
        dit,
        cond,
        cond,
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0, rescale=0.0, skip_step=1),
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0, rescale=0.0),
    )
    video, audio = _states()
    vx, ax = video.latent, audio.latent
    dit.value = 1.0
    v0, a0 = den(video, audio, vx, ax, 0.9, step=0)
    assert dit.calls[-1] == (True, True)
    # velocity 1 at sigma 0.9: x0 = x - 0.9
    assert np.allclose(np.array(v0), np.array(vx) - 0.9)
    assert np.allclose(np.array(a0), np.array(ax) - 0.9 * 101.0)
    dit.value = 5.0
    v1, a1 = den(video, audio, vx, ax, 0.5, step=1)  # video skipped
    assert dit.calls[-1] == (False, True)
    assert np.allclose(np.array(v1), np.array(v0))  # the last video x0
    assert np.allclose(np.array(a1), np.array(ax) - 0.5 * 105.0)
    v2, _ = den(video, audio, vx, ax, 0.3, step=2)
    assert dit.calls[-1] == (True, True)
    assert np.allclose(np.array(v2), np.array(vx) - 0.3 * 5.0)
    # both skipped: no forward at all
    both = pl.GuidedDenoiser(
        dit,
        cond,
        cond,
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0, rescale=0.0, skip_step=1),
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0, rescale=0.0, skip_step=1),
    )
    with pytest.raises(ValueError, match="first step"):
        both(video, audio, vx, ax, 0.9, step=1)
    both(video, audio, vx, ax, 0.9, step=0)
    n = len(dit.calls)
    out = both(video, audio, vx, ax, 0.5, step=1)
    assert len(dit.calls) == n and out is both.last


def test_guidance_requests_override_the_pipelines_defaults():
    params = server.normalize_request(
        {
            "prompt": "x",
            "steps": 12,
            "guidance": {
                "video": {"cfg": 4.0, "stg_blocks": [20, 28], "skip_step": 1},
                "audio": {"modality": 1.0, "rescale": 0.5},
            },
        },
        CFG,
    )
    assert params["steps"] == 12
    v, a = params["video_guidance"], params["audio_guidance"]
    assert v == sampling.Guidance(cfg=4.0, stg_blocks=(20, 28), skip_step=1)
    assert a == sampling.Guidance(cfg=7.0, modality=1.0, rescale=0.5)
    hq = server.normalize_request(
        {"prompt": "x", "guidance": {"video": {"cfg": 2.0}}}, {**CFG, "pipeline": "hq"}
    )
    assert hq["video_guidance"] == pl.HQ_VIDEO_GUIDANCE.__class__(
        cfg=2.0, stg=0.0, modality=3.0, rescale=0.45
    )
    for bad, msg in [
        ({"guidance": {"video": {"cfg": "x"}}}, "number"),
        ({"guidance": {"video": {"stg_blocks": [48]}}}, "stg_blocks"),
        ({"guidance": {"video": {"skip_step": -1}}}, "skip_step"),
        ({"guidance": {"video": {"rescale": 2}}}, "rescale"),
        ({"guidance": {"video": {"wat": 1}}}, "unknown fields"),
        ({"guidance": {"text": {}}}, "unknown modalities"),
        ({"steps": 0}, "steps"),
    ]:
        with pytest.raises(server.BadRequest, match=msg):
            server.normalize_request({"prompt": "x", **bad}, CFG)
    with pytest.raises(server.BadRequest, match="guided"):
        server.normalize_request(
            {"prompt": "x", "steps": 5}, {**CFG, "pipeline": "distilled"}
        )
    with pytest.raises(server.BadRequest, match="hq"):
        server.normalize_request({"prompt": "x", "lora_strengths": [0.1, 0.2]}, CFG)
    hq = server.normalize_request(
        {"prompt": "x", "lora_strengths": [0.1, 0.2]}, {**CFG, "pipeline": "hq"}
    )
    assert (hq["lora_stage_1"], hq["lora_stage_2"]) == (0.1, 0.2)


def _lora_file(path, name="transformer_blocks.0.attn1.to_q", rank=2, d=4):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((rank, d)).astype(np.float32)
    b = rng.standard_normal((d, rank)).astype(np.float32)
    mx.save_safetensors(
        str(path),
        {
            f"diffusion_model.{name}.lora_A.weight": mx.array(a),
            f"diffusion_model.{name}.lora_B.weight": mx.array(b),
        },
    )
    return a, b


def test_user_lora_loads_from_a_path_and_attaches_by_strength(tmp_path):
    from slimserve.video.ltx25.lora import Lora

    path = tmp_path / "style.safetensors"
    a, b = _lora_file(path)
    lora = Lora.from_path(path).load()
    assert list(lora.pairs) == ["transformer_blocks.0.attn1.to_q"]
    dit = SimpleNamespace(
        w={"transformer_blocks.0.attn1.to_q.weight": mx.zeros((4, 4))},
        split_k={},
        lora={},
    )
    lora.attach(dit, 0.6)
    (pa, pb, strength) = dit.lora["transformer_blocks.0.attn1.to_q"][0]
    assert strength == 0.6 and pa.shape == (2, 4) and pb.shape == (4, 2)
    lora.detach(dit)
    assert dit.lora == {}
    with pytest.raises(FileNotFoundError):
        Lora.from_path(tmp_path / "missing.safetensors")
    # an adapter whose targets are not in the model is refused at attach
    other = tmp_path / "other.safetensors"
    _lora_file(other, name="transformer_blocks.9.ff.net.2")
    with pytest.raises(KeyError, match="not in the DiT"):
        Lora.from_path(other).load().attach(dit)


def test_lora_requests_and_cli_groups(tmp_path):
    from slimserve.video import cli

    path = tmp_path / "s.safetensors"
    _lora_file(path)
    params = server.normalize_request(
        {"prompt": "x", "loras": [{"path": str(path), "strength": 0.7}]}, CFG
    )
    assert params["loras"] == [(str(path), 0.7)]
    with pytest.raises(server.BadRequest, match="safetensors"):
        server.normalize_request({"prompt": "x", "loras": [{"path": "nope.bin"}]}, CFG)
    body = {}
    cli._add_loras(body, SimpleNamespace(lora=[[str(path)], [str(path), "0.3"]]))
    assert [item["strength"] for item in body["loras"]] == [1.0, 0.3]
    body = {}
    cli._add_guidance(
        body,
        SimpleNamespace(
            num_inference_steps=20,
            video_cfg_guidance_scale=2.5,
            video_stg_blocks=[28],
            a2v_guidance_scale=None,
            audio_skip_step=1,
            distilled_lora_strength_stage_2=0.4,
        ),
    )
    assert body["steps"] == 20
    assert body["guidance"] == {
        "video": {"cfg": 2.5, "stg_blocks": [28]},
        "audio": {"skip_step": 1},
    }
    assert body["lora_strengths"] == [pl.HQ_LORA_STAGE_1, 0.4]


def test_the_engine_attaches_user_loras_around_a_request():
    """server._generate wraps the pipeline call in engine.user_loras(specs)."""
    import inspect

    src = inspect.getsource(server.VideoService._generate)
    assert "with engine.user_loras(loras):" in src
