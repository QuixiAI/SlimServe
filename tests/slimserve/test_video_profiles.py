# SPDX-License-Identifier: Apache-2.0
"""The LTX-2.5 video profiles: registry shape, request envelope, and the
one-at-a-time serving queue. No weights and no GPU are needed here; the
measured gates live in perf/ltx25_harness/."""

import base64
import os
import threading
import time

import pytest

from slimserve import registry
from slimserve.registry import files_for, resolve
from slimserve.video import server

VIDEO_IDS = (
    "ltx25-distilled",
    "ltx25-dev",
    "ltx25-dfr",
    "ltx25-hq",
    "ltx25-keyframes",
    "ltx25-one-stage",
    "ltx25-retake",
    "ltx25-ic-lora",
    "ltx25-a2vid",
    "ltx25-dubit",
    "ltx25-t2a",
    "ltx25-alpha",
    "ltx25-hdr-ic-lora",
)
FAST_IDS = (
    "ltx25-distilled-fast",
    "ltx25-dev-fast",
    "ltx25-dfr-fast",
    "ltx25-keyframes-fast",
    "ltx25-one-stage-fast",
    "ltx25-retake-fast",
    "ltx25-ic-lora-fast",
    "ltx25-a2vid-fast",
    "ltx25-dubit-fast",
    "ltx25-alpha-fast",
    "ltx25-hdr-ic-lora-fast",
)
GIB = 1 << 30


def _plan(profile_id):
    return resolve(profile_id, "metal", 1, None, 128 * GIB)


def test_video_profiles_resolve_on_metal_only():
    for profile_id in VIDEO_IDS:
        entry = registry.describe(profile_id)
        assert entry["platforms"] == ["metal"]
        plan = _plan(profile_id)
        assert plan.engine["pipeline"] == profile_id.split("-", 1)[1].replace("-", "_")
        assert not registry.is_language_model(plan.source)


def test_fast_profiles_are_the_exact_profiles_plus_a_fast_block():
    pytest.importorskip("mlx.core")  # the pipeline module needs MLX
    from slimserve.video.ltx25.pipeline import Fast

    for fast_id in FAST_IDS:
        exact_id = fast_id[: -len("-fast")]
        fast, exact = _plan(fast_id).engine, _plan(exact_id).engine
        assert fast["pipeline"] == exact["pipeline"]
        assert fast["decoder"] == "conv" and "fast" not in exact
        settings = server.fast_settings(fast)
        assert isinstance(settings, Fast) and settings.active
        assert {k: v for k, v in fast.items() if k not in ("fast", "decoder")} == {
            k: v for k, v in exact.items() if k != "decoder"
        }
        assert registry.describe(fast_id)["platforms"] == ["metal"]
    assert server.fast_settings(_plan("ltx25-dfr-fast").engine).attention_tiles == (
        2,
        2,
    )
    dev = server.fast_settings(_plan("ltx25-dev-fast").engine)
    assert dev.steps == 20 and dev.step_cache == 0.1
    assert dev.guidance is None  # all four guidance passes stay (ledger section 24)


def test_video_profiles_carry_no_chat_serving_defaults():
    for profile_id in VIDEO_IDS:
        engine = _plan(profile_id).engine
        for key in (
            "enable_prefix_caching",
            "enable_auto_tool_choice",
            "default_chat_template_kwargs",
        ):
            assert key not in engine


def test_video_profiles_are_gated_on_the_validated_memory_size():
    with pytest.raises(registry.ProfileError):
        resolve("ltx25-distilled", "metal", 1, None, 64 * GIB)


def test_each_pipeline_fetches_its_own_transformer_and_adapter():
    paths = {
        pid: {entry["path"] for entry in files_for(_plan(pid))} for pid in VIDEO_IDS
    }
    dev = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    distilled = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    assert distilled in paths["ltx25-distilled"] and dev not in paths["ltx25-distilled"]
    assert dev in paths["ltx25-dev"] and distilled not in paths["ltx25-dev"]
    assert "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" in paths["ltx25-dev"]
    assert paths["ltx25-hq"] == paths["ltx25-dev"]  # the same transformer and adapter
    assert paths["ltx25-keyframes"] == paths["ltx25-dev"]
    assert paths["ltx25-one-stage"] == paths["ltx25-dev"]
    detail = "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    assert detail in paths["ltx25-dfr"] and detail not in paths["ltx25-distilled"]
    for pid in VIDEO_IDS:
        assert (
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" in paths[pid]
        )
        assert "vae/ltx-2.5-video-vae-conv-bf16.safetensors" in paths[pid]


def test_the_detailing_adapter_downloads_from_its_own_repository():
    entry = next(e for e in files_for(_plan("ltx25-dfr")) if "ic-lora" in e["path"])
    assert "LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler" in entry["url"]
    assert entry["url"].endswith(
        "/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    )


def test_the_prompt_enhancer_is_fetched_only_when_asked_for():
    """The enhancer is google/gemma-4-E2B-it (10.2 GB): an optional file group,
    pinned to a revision and kept under the LTX root where checkpoints.path_of
    finds it, fetched only by a run that asks for enhancement."""
    from dataclasses import replace

    from slimserve.video import cli as video_cli

    enhancer = "prompt_enhancer/gemma-4-E2B-it/model.safetensors"
    for pid in VIDEO_IDS + FAST_IDS:
        plan = _plan(pid)
        assert not [e for e in files_for(plan) if e["role"] in ("shared", "optional")]
        args = video_cli.parser().parse_args([])
        assert video_cli.prepare(plan, args).optional == ()
        args = video_cli.parser().parse_args(["--enhance-prompt"])
        wanted = files_for(video_cli.prepare(plan, args))
        optional = [e for e in wanted if e["role"] == "optional"]
        paths = {e["path"] for e in optional}
        assert enhancer in paths
        assert "prompt_enhancer/gemma-4-E2B-it/tokenizer.json" in paths
        assert "prompt_enhancer/gemma-4-E2B-it/config.json" in paths
        for e in optional:
            assert e["url"].startswith(
                "https://huggingface.co/google/gemma-4-E2B-it/resolve/"
            )
            assert "/resolve/main/" not in e["url"]
            assert e["url"].endswith("/" + e["path"].rsplit("/", 1)[1])
            assert e["sha256"]
            assert e["local_dir"] == "ltx-2.5/official"
        serve = video_cli.prepare(
            plan, video_cli.parser().parse_args(["--prompt-enhancer"])
        )
        assert serve.optional == ("prompt_enhancer",)
    with pytest.raises(registry.ProfileError):
        files_for(replace(_plan("ltx25-distilled"), optional=("nope",)))


def test_the_enhancer_path_matches_the_checkpoint_registry():
    pytest.importorskip("mlx.core")  # checkpoints imports MLX
    from slimserve.video.ltx25 import checkpoints

    paths = {
        e["path"]
        for e in registry._registry()["sources"]["ltx25"]["optional"]["prompt_enhancer"]
    }
    assert checkpoints.FILES["enhancer"] in paths


def test_a_server_without_the_enhancer_refuses_enhance_requests(tmp_path):
    engine = _FakeEngine()
    service = server.VideoService(
        {**CFG, "prompt_enhancer": False},
        "LTX-2.5",
        tmp_path,
        engine_factory=lambda: engine,
    )
    assert service.ready.wait(5)
    with pytest.raises(server.BadRequest, match="prompt enhancer"):
        service.submit({"prompt": "x", "enhance_prompt": True})
    assert service.submit({"prompt": "x"}).done.wait(10)
    service.stop()


CFG = {
    "pipeline": "distilled",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def test_request_without_a_length_is_auto_duration_within_the_envelope():
    params = server.normalize_request({"prompt": "a fox"}, CFG)
    assert (params["width"], params["height"], params["seed"]) == (1536, 1024, 42)
    assert params["num_frames"] is None  # the duration head decides
    assert params["max_num_frames"] == 121  # 24,576 tokens at 1536x1024
    small = server.normalize_request({"prompt": "a fox", "size": "768x512"}, CFG)
    assert small["max_num_frames"] == 505
    explicit = server.normalize_request({"prompt": "a fox", "seconds": 5}, CFG)
    assert explicit["num_frames"] == 121 and "max_num_frames" not in explicit


def test_request_accepts_size_and_seconds():
    params = server.normalize_request(
        {"prompt": "a fox", "size": "768x512", "seconds": 2}, CFG
    )
    assert (params["width"], params["height"], params["num_frames"]) == (768, 512, 49)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": " "},
        {"prompt": "x", "size": "770x512"},
        {"prompt": "x", "num_frames": 120},
        {"prompt": "x", "size": "3072x2048"},  # outside the validated memory envelope
        {"prompt": "x", "num_frames": 241},
        {"prompt": "x", "negative_prompt": "blurry"},  # distilled has no CFG
    ],
)
def test_request_outside_the_envelope_is_refused(body):
    with pytest.raises(server.BadRequest):
        server.normalize_request(body, CFG)


def test_temporal_upscalings_is_a_dfr_option():
    dfr = dict(CFG, pipeline="dfr")
    assert "temporal_upscalings" not in server.normalize_request({"prompt": "x"}, dfr)
    req = {"prompt": "x", "temporal_upscalings": 2}
    assert server.normalize_request(req, dfr)["temporal_upscalings"] == 2
    assert "temporal_upscalings" not in server.normalize_request(
        {"prompt": "x", "temporal_upscalings": 0}, dfr
    )
    with pytest.raises(server.BadRequest):
        server.normalize_request({"prompt": "x", "temporal_upscalings": 3}, dfr)
    with pytest.raises(server.BadRequest):
        server.normalize_request(req, CFG)  # distilled


def test_spatial_upscalings_is_a_dfr_option_on_128_multiples():
    dfr = dict(CFG, pipeline="dfr")
    assert "spatial_upscalings" not in server.normalize_request({"prompt": "x"}, dfr)
    assert "spatial_upscalings" not in server.normalize_request(
        {"prompt": "x", "spatial_upscalings": 1}, dfr
    )
    req = {"prompt": "x", "spatial_upscalings": 2}
    assert server.normalize_request(req, dfr)["spatial_upscalings"] == 2
    with pytest.raises(server.BadRequest):
        server.normalize_request(dict(req, size="1472x1024"), dfr)
    with pytest.raises(server.BadRequest):
        server.normalize_request({"prompt": "x", "spatial_upscalings": 3}, dfr)
    with pytest.raises(server.BadRequest):
        server.normalize_request(req, CFG)


def test_decoder_defaults_to_diffusion_and_accepts_conv():
    assert server.normalize_request({"prompt": "x"}, CFG)["decoder"] == "diffusion"
    req = {"prompt": "x", "decoder": "conv"}
    assert server.normalize_request(req, CFG)["decoder"] == "conv"
    with pytest.raises(server.BadRequest):
        server.normalize_request({"prompt": "x", "decoder": "fast"}, CFG)


def test_negative_prompt_reaches_the_dev_pipeline():
    params = server.normalize_request(
        {"prompt": "x", "negative_prompt": "blurry"}, {**CFG, "pipeline": "dev"}
    )
    assert params["negative_prompt"] == "blurry"


def test_enhance_prompt_is_opt_in_and_boolean():
    assert "enhance_prompt" not in server.normalize_request({"prompt": "x"}, CFG)
    assert "enhance_prompt" not in server.normalize_request(
        {"prompt": "x", "enhance_prompt": False}, CFG
    )
    assert server.normalize_request({"prompt": "x", "enhance_prompt": True}, CFG)[
        "enhance_prompt"
    ]
    with pytest.raises(server.BadRequest):
        server.normalize_request({"prompt": "x", "enhance_prompt": "yes"}, CFG)


def test_enhanced_prompt_is_what_the_pipeline_sees_and_is_reported(tmp_path):
    engine = _FakeEngine()
    service = _service(tmp_path, engine)
    job = service.submit({"prompt": "a cat", "enhance_prompt": True})
    assert job.done.wait(10) and job.status == "completed", job.error
    assert engine.prompts == ["ENHANCED a cat"]
    assert job.params["enhanced_prompt"] == "ENHANCED a cat"
    assert list(job.timings["spans_s"]) == ["enhance", "stage1"]
    assert engine.enhancer_loaded is False  # dropped before the pipeline runs
    assert engine.enhance_images == [None]
    plain = service.submit({"prompt": "a dog"})
    assert plain.done.wait(10)
    assert engine.prompts[-1] == "a dog" and "enhanced_prompt" not in plain.params
    # an image request is enhanced with the still (upstream enhance_i2v)
    png = _tiny_png()
    i2v = service.submit(
        {
            "prompt": "a cat",
            "enhance_prompt": True,
            "image": base64.b64encode(png).decode(),
        }
    )
    assert i2v.done.wait(10) and i2v.status == "completed", i2v.error
    assert engine.enhance_images[-1] == png
    service.stop()


def _tiny_png() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeEngine:
    """Records overlap: the real engine must never run two generations at once."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.prompts = []

    def enhance(self, prompt, image=None):
        self.enhancer_loaded = True
        self.enhance_images = getattr(self, "enhance_images", []) + [image]
        return f"ENHANCED {prompt}"

    def user_loras(self, specs):
        from contextlib import nullcontext

        self.loras = getattr(self, "loras", []) + [specs]
        return nullcontext()

    def unload_enhancer(self):
        self.enhancer_loaded = False

    def distilled(self, prompt, on_step=None, **params):
        self.prompts.append(prompt)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls += 1
        if "fail" in prompt:
            self.active -= 1
            raise RuntimeError("boom")
        on_step("stage1", 0, 1.0)
        time.sleep(0.02)
        self.active -= 1

        class _Timings:
            spans = {"stage1": 0.02}
            memory = {"stage1": (GIB, GIB, 0)}

        class _Result:
            timings = _Timings()
            num_frames = params.get("num_frames") or 49
            predicted_seconds = None if params.get("num_frames") else 2.0

        return _Result()

    def render(self, result, path, seed=42, decoder=None, hdr=None):
        self.decoders = getattr(self, "decoders", []) + [decoder]
        path.write_bytes(b"mp4")
        return path


def _service(tmp_path, engine):
    service = server.VideoService(
        CFG, "LTX-2.5", tmp_path, engine_factory=lambda: engine
    )
    assert service.ready.wait(5)
    return service


def test_concurrent_requests_run_one_at_a_time_and_all_complete(tmp_path):
    engine = _FakeEngine()
    service = _service(tmp_path, engine)
    jobs, errors = [], []

    def client(i):
        try:
            jobs.append(service.submit({"prompt": f"clip {i}", "size": "768x512"}))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for job in jobs:
        assert job.done.wait(10)
        assert job.status == "completed" and job.path.read_bytes() == b"mp4"
    assert engine.calls == 8 and engine.max_active == 1
    assert service.health()["completed"] == 8
    service.stop()


def test_a_failed_generation_does_not_stop_the_queue(tmp_path):
    engine = _FakeEngine()
    service = _service(tmp_path, engine)
    bad = service.submit({"prompt": "fail please"})
    good = service.submit({"prompt": "fine"})
    assert bad.done.wait(10) and good.done.wait(10)
    assert bad.status == "failed" and "boom" in bad.error
    assert good.status == "completed"
    service.stop()


def test_the_queue_is_bounded(tmp_path):
    gate = threading.Event()

    class _Blocked(_FakeEngine):
        def distilled(self, prompt, on_step=None, **params):
            gate.wait(10)
            return super().distilled(prompt, on_step=on_step, **params)

    service = _service(tmp_path, _Blocked())
    first = service.submit({"prompt": "x"})
    deadline = time.time() + 5
    while first.status != "in_progress" and time.time() < deadline:
        time.sleep(0.01)  # the worker must hold it before the queue is filled
    assert first.status == "in_progress"
    accepted = [first] + [
        service.submit({"prompt": "x"}) for _ in range(server.MAX_QUEUED)
    ]
    with pytest.raises(OverflowError):
        for _ in range(3):
            service.submit({"prompt": "x"})
    gate.set()
    for job in accepted:
        assert job.done.wait(20)
    service.stop()


def test_an_engine_that_fails_to_load_is_reported_not_hidden(tmp_path):
    def broken():
        raise FileNotFoundError("weights missing")

    service = server.VideoService(CFG, "LTX-2.5", tmp_path, engine_factory=broken)
    assert service.ready.wait(5)
    health = service.health()
    assert health["status"] == "error" and "weights missing" in health["error"]


def test_dfr_canvas_and_dev_schedule():
    mx = pytest.importorskip("mlx.core")  # noqa: F841 - the sampling module needs MLX
    from slimserve.video.ltx25 import sampling

    assert sampling.dfr_canvas(121) == (121, 24, [24, 48, 72, 96, 120])
    assert sampling.dfr_canvas(97) == (97, 32, [32, 64, 96])
    assert sampling.dfr_canvas(49)[0] == 49
    sigmas = sampling.ltx2_schedule(30, 1536)
    assert len(sigmas) == 31 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert abs(sigmas[-2] - 0.1) < 1e-9
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_samplers_reach_the_denoisers_answer():
    mx = pytest.importorskip("mlx.core")
    from slimserve.video.ltx25 import sampling

    target_v, target_a = mx.full((1, 6, 4), 2.0), mx.full((1, 3, 4), -1.0)
    video = sampling.noised_state(
        (1, 6, 4), mx.zeros((1, 6, 3)), seed=1, tokens_per_frame=2
    )
    audio = sampling.noised_state((1, 3, 4), mx.zeros((1, 3, 1)), seed=2)
    perfect = lambda vs, au, vx, ax, sigma, step=None: (target_v, target_a)  # noqa: E731
    for out in (
        sampling.euler_loop(perfect, video, audio, sampling.STAGE_2_DISTILLED_SIGMAS),
        sampling.euler_ancestral_loop(
            perfect, video, audio, sampling.DISTILLED_SIGMAS, noise_seed=3
        ),
    ):
        assert float(mx.abs(out[0] - target_v).max()) < 1e-5
        assert float(mx.abs(out[1] - target_a).max()) < 1e-5


def test_adapter_and_chunk_arguments_reach_the_right_place(tmp_path):
    """IC-LoRA / Dub-It / alpha get their adapters as a pipeline argument (they
    attach them per stage), every other pipeline through engine.user_loras;
    a chunked request also gets the decoder (it decodes per window)."""

    class _Engine(_FakeEngine):
        def __init__(self):
            super().__init__()
            self.kwargs = []

        def _run(self, prompt, on_step=None, **params):
            self.kwargs.append(params)
            return self.distilled(prompt, on_step=on_step, **params)

        ic_lora = dubit = alpha = dev = _run

    lora = tmp_path / "a.safetensors"
    lora.write_bytes(b"x")
    ref = tmp_path / "r.mp4"
    ref.write_bytes(b"x")
    engine = _Engine()
    service = server.VideoService(
        {**CFG, "pipeline": "ic_lora"},
        "LTX-2.5",
        tmp_path,
        engine_factory=lambda: engine,
        input_root=tmp_path,
    )
    assert service.ready.wait(5)
    job = service.submit(
        {
            "prompt": "x",
            "seconds": 2,
            "video_conditioning": [{"path": str(ref)}],
            "loras": [{"path": str(lora), "strength": 0.5}],
        }
    )
    assert job.done.wait(10) and job.status == "completed", job.error
    assert engine.kwargs[-1]["loras"] == [(os.path.realpath(lora), 0.5)]
    assert engine.loras[-1] is None  # not through the request-wide context
    service.stop()
    engine = _Engine()
    service = server.VideoService(
        {**CFG, "pipeline": "dev"},
        "LTX-2.5",
        tmp_path,
        engine_factory=lambda: engine,
        input_root=tmp_path,
    )
    assert service.ready.wait(5)
    job = service.submit(
        {"prompt": "x", "seconds": 10, "chunked": True, "loras": [{"path": str(lora)}]}
    )
    assert job.done.wait(10) and job.status == "completed", job.error
    assert "loras" not in engine.kwargs[-1]
    assert engine.loras[-1] == [(os.path.realpath(lora), 1.0)]
    assert engine.kwargs[-1]["decoder"] == "diffusion"
    assert engine.kwargs[-1]["chunk"].chunk_pixel_frames == 97
    service.stop()
