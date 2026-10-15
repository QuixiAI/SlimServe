# SPDX-License-Identifier: Apache-2.0
"""HTTP serving for the video profiles.

One GPU, one generation at a time: requests queue and a single worker thread
owns the engine (and every MLX call), with the weights resident between
requests. The API is job-shaped because a clip takes minutes:

    POST   /v1/videos                 {"prompt": ..., "size": "1536x1024",
                                       "steps": 30,
                                       "guidance": {"video": {"cfg": 3.0, "stg": 1.0,
                                                    "stg_blocks": [28], "rescale": 0.7,
                                                    "modality": 3.0, "skip_step": 0},
                                                    "audio": {...}},
                                       "loras": [{"path": "x.safetensors",
                                                  "strength": 1.0}],
                                       "lora_strengths": [0.25, 0.5],
                                       "seconds": 5, "seed": 42,
                                       "enhance_prompt": false,
                                       "image": <base64 or data: URL>,
                                       "image_strength": 1.0,
                                       "images": [{"image": ..., "frame": 48,
                                                   "strength": 1.0, "crf": 18}]}
    POST   /v1/videos  (retake)       {"prompt": ..., "video_path": "/clips/a.mp4"
                                       or "video": <base64 mp4>, "start_time": 1.0,
                                       "end_time": 2.5, "regenerate_audio": true}
    GET    /v1/videos/<id>            status, progress, timings
    GET    /v1/videos/<id>/content    the mp4 (H.264 + AAC)
    DELETE /v1/videos/<id>
    GET    /v1/models, /health

Fields that name files on the server (`video_path`, `audio_path`,
`reference_video`, `video_conditioning[].path`, `attention_mask`,
`loras[].path`, `images[].path`, `text_embeddings`) are refused unless the
operator started the server with an input root (`--input-root DIR` or
SLIMSERVE_VIDEO_INPUT_ROOT); with one, a path resolves inside that directory
(relative paths against it, symlinks followed) and anything outside it is
refused. Base64 uploads (`video`, `audio`, `image`, `images[].image`) work
either way. Bodies above the configured cap are answered with 413, chunked
bodies without a Content-Length with 411, malformed fields with 400, and
requests beyond the normalisation slots (SLIMSERVE_VIDEO_MAX_PREPARING) or a
full queue with 429. The input root must be operator-owned: a path is
checked on arrival and opened when the job runs (see docs/ltx25_metal.md).

`"wait": true` in the POST body holds the request open and returns the
finished job. `enhance_prompt` (default false, as upstream's --enhance-prompt)
has Gemma-4 E2B-it rewrite the request into the model's caption style first
(looking at the still when there is one); the rewritten text is reported as
the job's `enhanced_prompt`. `image`
(image-to-video) is an encoded still (PNG, JPEG, ...)
as base64, optionally wrapped in a `data:image/...;base64,` URL; it becomes
the clip's first frame, pinned at `image_strength` (0-1, default 1.0). `images`
is upstream's repeatable `--image PATH FRAME_IDX STRENGTH [CRF]`: stills at any
pixel frame (frame 0 replaces the first latent frame, other frames ride along
as keyframe tokens); the keyframe interpolation profile requires it. On the
guided pipelines `steps` and `guidance` (per modality: upstream's
MultiModalGuiderParams cfg / stg / stg_blocks / rescale / modality (a2v on
video, v2a on audio) / skip_step) override the profile's defaults; `loras`
attaches user adapters (upstream --lora PATH [STRENGTH]; files on the server)
to every stage; `lora_strengths` is the hq pipeline's two distilled-LoRA
strengths. Requests outside the profile's validated envelope are refused: the
envelope is what was measured to fit this machine's memory.
"""

from __future__ import annotations

import base64
import binascii
import contextvars
import io
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_QUEUED = 16
# Requests being normalised at once (decoding base64 sources, writing them to
# disk, probing them with ffprobe); more than this are answered 429 before
# their body is touched. SLIMSERVE_VIDEO_MAX_PREPARING overrides it.
DEFAULT_MAX_PREPARING = 4
KEEP_FINISHED = 32
# The largest POST body the server reads (base64 sources and stills travel in
# it). SLIMSERVE_VIDEO_MAX_BODY_MB or --max-body-mb override it.
DEFAULT_MAX_BODY_BYTES = 256 << 20
# pipelines on the dev transformer (CFG/STG guidance, a negative prompt)
GUIDED_PIPELINES = ("dev", "hq", "keyframes", "one_stage", "a2vid", "t2a", "alpha")
# pipelines that edit a source clip: its size, length and rate are the output's
SOURCE_PIPELINES = ("retake", "dubit", "hdr_ic_lora")


class BadRequest(ValueError):
    pass


class QueueFull(OverflowError):
    """Answered 429: the queue or the normalisation slots are full."""


class ServiceStopped(RuntimeError):
    """Answered 503: the server is shutting down."""


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: str = "queued"  # queued | in_progress | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    path: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)
    # base64 sources written for this job; removed once it has run
    uploads: list[Path] = field(default_factory=list)

    def public(self, model: str) -> dict[str, Any]:
        out = {
            "id": self.id,
            "object": "video",
            "model": model,
            "status": self.status,
            "created_at": int(self.created_at),
            "progress": self.progress,
            **self.params,
        }
        if isinstance(out.get("image"), (bytes, bytearray)):
            out["image"] = f"<{len(out['image'])} bytes>"
        for key in ("video_guidance", "audio_guidance"):
            if key in out:
                out[key] = asdict(out[key])
        for key in ("loras", "video_conditioning"):
            if out.get(key):
                out[key] = [{"path": p, "strength": s} for p, s in out[key]]
        if out.get("images"):
            out["images"] = [
                {
                    "image": f"<{len(s.image)} bytes>",
                    "frame": s.frame,
                    "strength": s.strength,
                    **({"crf": s.crf} if s.crf is not None else {}),
                }
                for s in out["images"]
            ]
        if self.completed_at:
            out["completed_at"] = int(self.completed_at)
            out["seconds_elapsed"] = round(
                self.completed_at - (self.started_at or self.created_at), 2
            )
        if self.error:
            out["error"] = {"message": self.error}
        if self.timings:
            out["timings"] = self.timings
        return out


# Set while VideoService.submit normalises a request: the uploads written for
# it, so they can be removed when the request is refused or the job ends.
_UPLOADS: contextvars.ContextVar[list[Path] | None] = contextvars.ContextVar(
    "slimserve_video_uploads", default=None
)
# cfg key holding the server's input root: absent = the local CLI (any path the
# user can read), None = serving with path fields disabled, str = the root.
INPUT_ROOT = "input_root"


def server_path(value: Any, name: str, cfg: dict[str, Any], dirs: bool = False) -> str:
    """A request field that names a file (or, with `dirs`, an EXR frame folder)
    on the server.

    From the local CLI (no input root key in cfg) it is the user's own path.
    When serving, it must resolve inside the operator's input root; nothing
    outside the root is checked for existence or echoed back."""
    if not isinstance(value, str) or not value:
        raise BadRequest(f"{name} must be a path string")
    if INPUT_ROOT not in cfg:
        path = Path(value).expanduser()
        if not (path.is_file() or (dirs and path.is_dir())):
            raise BadRequest(f"{name} {path} is not a file")
        return str(path)
    root = cfg[INPUT_ROOT]
    if not root:
        raise BadRequest(
            f"{name}: this server does not accept server-side paths (the operator "
            "has not set an input root); send the file as base64 where the "
            "request allows it"
        )
    real = os.path.realpath(os.path.join(root, value))
    if os.path.commonpath([real, root]) != root:
        raise BadRequest(f"{name}: the path is outside the server's input root")
    if not (os.path.isfile(real) or (dirs and os.path.isdir(real))):
        raise BadRequest(f"{name} {value} is not a file under the input root")
    return real


def resolve_input_root(value: str | os.PathLike | None) -> str | None:
    """The operator's input root as a real directory path, or None (disabled)."""
    if value is None or str(value) == "":
        return None
    root = os.path.realpath(os.path.expanduser(str(value)))
    if not os.path.isdir(root):
        raise ValueError(f"input root {value} is not a directory")
    return root


def max_body_bytes(value: float | None = None) -> int:
    """The POST body cap in bytes: the argument (MiB), else
    SLIMSERVE_VIDEO_MAX_BODY_MB, else DEFAULT_MAX_BODY_BYTES."""
    if value is None:
        env = os.environ.get("SLIMSERVE_VIDEO_MAX_BODY_MB")
        if not env:
            return DEFAULT_MAX_BODY_BYTES
        value = float(env)
    if value <= 0:
        raise ValueError("the body size cap must be positive")
    return int(value * (1 << 20))


def normalize_request(body: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a request against the profile's envelope; returns engine parameters."""
    prompt = body.get("prompt")
    if cfg["pipeline"] == "hdr_ic_lora":
        return normalize_hdr_ic_lora_request(body, cfg)
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("`prompt` is required")
    if cfg["pipeline"] in SOURCE_PIPELINES:
        return normalize_source_request(body, cfg, prompt)
    if cfg["pipeline"] == "t2a":
        return normalize_t2a_request(body, cfg, prompt)
    width, height = cfg["width"], cfg["height"]
    if size := body.get("size"):
        try:
            width, height = (int(x) for x in str(size).lower().split("x"))
        except ValueError as exc:
            raise BadRequest("`size` must look like 1536x1024") from exc
    width, height = int(body.get("width", width)), int(body.get("height", height))
    fps = float(body.get("fps", cfg["fps"]))
    if width % 64 or height % 64 or width < 256 or height < 256:
        raise BadRequest("width and height must be multiples of 64, at least 256")
    if not 1.0 <= fps <= 60.0:
        raise BadRequest("fps must be between 1 and 60")
    per_frame = (height // 32) * (width // 32)
    if per_frame > (cfg["height"] // 32) * (cfg["width"] // 32):
        raise BadRequest(
            f"{width}x{height} frames are larger than this profile's validated "
            f"{cfg['width']}x{cfg['height']}"
        )
    # the longest 8k + 1 clip at this size inside the validated token envelope
    max_frames = (cfg["max_video_tokens"] // per_frame - 1) * 8 + 1
    chunk = chunk_config(body, cfg["pipeline"])
    if chunk is not None:
        # chunked: the window is what must fit; the clip may be long (1024
        # frames, the model's maximum)
        if chunk.chunk_pixel_frames > max_frames:
            raise BadRequest(
                f"chunk_pixel_frames {chunk.chunk_pixel_frames} is larger than this "
                f"profile's envelope at {width}x{height} ({max_frames} frames)"
            )
        max_frames = 1024
    if "num_frames" in body:
        frames = int(body["num_frames"])
    elif "seconds" in body:
        frames = int(round(float(body["seconds"]) * fps / 8)) * 8 + 1
    else:
        # no length asked for: the duration head predicts one from the prompt
        # (upstream's auto duration), capped at the envelope
        frames = None
    if frames is not None:
        if frames < 9 or (frames - 1) % 8:
            raise BadRequest("num_frames must be 8k + 1 (9, 17, ..., 121)")
        if frames > max_frames:
            tokens = ((frames - 1) // 8 + 1) * per_frame
            raise BadRequest(
                f"{width}x{height}x{frames} is {tokens} latent tokens; "
                "this profile is validated up to "
                f"{cfg['max_video_tokens']} "
                f"({cfg['width']}x{cfg['height']}x{cfg['num_frames']})"
            )
    params: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": frames,
        "fps": fps,
        "seed": int(body.get("seed", 42)),
    }
    if frames is None:
        params["max_num_frames"] = max_frames
    if chunk is not None:
        params["chunk"] = chunk
    decoder = str(body.get("decoder", cfg.get("decoder", "diffusion")))
    if decoder not in ("diffusion", "conv"):
        raise BadRequest(
            "decoder must be 'diffusion' (default, sharper) or 'conv' (faster)"
        )
    params["decoder"] = decoder
    if "negative_prompt" in body:
        if cfg["pipeline"] not in GUIDED_PIPELINES:
            raise BadRequest(
                "negative_prompt applies to the guided pipelines only "
                f"({', '.join(GUIDED_PIPELINES)}; the distilled flows have no CFG)"
            )
        params["negative_prompt"] = str(body["negative_prompt"])
    if "temporal_upscalings" in body:
        if cfg["pipeline"] != "dfr":
            raise BadRequest("temporal_upscalings applies to the DFR pipeline only")
        rounds = body["temporal_upscalings"]
        if rounds not in (0, 1, 2):
            raise BadRequest("temporal_upscalings must be 0, 1 or 2")
        if rounds:
            params["temporal_upscalings"] = int(rounds)
    if "spatial_upscalings" in body:
        if cfg["pipeline"] != "dfr":
            raise BadRequest("spatial_upscalings applies to the DFR pipeline only")
        stages = body["spatial_upscalings"]
        if stages not in (1, 2):
            raise BadRequest("spatial_upscalings must be 1 or 2")
        if stages == 2:
            if width % 128 or height % 128:
                raise BadRequest(
                    "spatial_upscalings 2 needs width and height as multiples of 128"
                )
            params["spatial_upscalings"] = 2
    if "enhance_prompt" in body:
        if not isinstance(body["enhance_prompt"], bool):
            raise BadRequest("enhance_prompt must be true or false")
        if body["enhance_prompt"]:
            params["enhance_prompt"] = True
    if body.get("image") is not None:
        params["image"] = decode_image_field(body["image"])
        params["image_strength"] = _strength(body.get("image_strength", 1.0))
    elif "image_strength" in body:
        raise BadRequest("image_strength needs an image")
    if body.get("images") is not None:
        params["images"] = [
            still_field(item, frames, cfg["pipeline"], cfg)
            for item in _list(body["images"])
        ]
    guided = cfg["pipeline"] in GUIDED_PIPELINES
    if "steps" in body:
        if not guided:
            raise BadRequest("steps applies to the guided pipelines only")
        steps = body["steps"]
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 200
        ):
            raise BadRequest("steps must be an integer in [1, 200]")
        params["steps"] = steps
    if "guidance" in body:
        if not guided:
            raise BadRequest("guidance applies to the guided pipelines only")
        params.update(guidance_fields(body["guidance"], cfg["pipeline"]))
    if "lora_strengths" in body:
        if cfg["pipeline"] != "hq":
            raise BadRequest("lora_strengths applies to the hq pipeline only")
        pair = body["lora_strengths"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise BadRequest("lora_strengths is [stage 1, stage 2]")
        params["lora_stage_1"], params["lora_stage_2"] = (
            _strength(pair[0], "lora_strengths", 2.0),
            _strength(pair[1], "lora_strengths", 2.0),
        )
    if body.get("loras") is not None:
        params["loras"] = [
            lora_field(item, cfg) for item in _list(body["loras"], "loras")
        ]
    params.update(keyframe_fields(body, cfg["pipeline"]))
    if cfg["pipeline"] == "ic_lora":
        params.update(ic_lora_fields(body, params, cfg))
    elif cfg["pipeline"] == "alpha":
        extra = [k for k in IC_LORA_KEYS[1:] if k in body]
        if extra:
            raise BadRequest(f"{', '.join(extra)}: not an alpha-gen option")
        params.update(ic_lora_fields(body, params, cfg))
    elif any(key in body for key in IC_LORA_KEYS):
        raise BadRequest(
            f"{', '.join(k for k in IC_LORA_KEYS if k in body)}: IC-LoRA options "
            "apply to the ic_lora pipeline only"
        )
    if cfg["pipeline"] == "a2vid":
        params.update(a2vid_fields(body, params, cfg))
    elif any(key in body for key in A2VID_KEYS):
        raise BadRequest(
            f"{', '.join(k for k in A2VID_KEYS if k in body)}: audio-to-video "
            "options apply to the a2vid pipeline only"
        )
    hdr_field(body, params)
    if cfg["pipeline"] == "keyframes":
        if "image" in params:  # the shorthand is a frame-0 keyframe here
            from slimserve.video.ltx25.sampling import Still

            first = Still(params.pop("image"), 0, params.pop("image_strength"))
            params["images"] = [first, *params.get("images", [])]
        if not params.get("images"):
            raise BadRequest(
                "the keyframe interpolation pipeline needs `images` "
                '([{"image": ..., "frame": N, "strength": 1.0}, ...])'
            )
    return params


def normalize_t2a_request(
    body: dict[str, Any], cfg: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Text to audio: `seconds` / `num_frames` (at `fps`; the duration head
    decides otherwise), `seed`, `negative_prompt`, `steps`, `guidance.audio`;
    no size, stills or decoder. The job's content is a WAV."""
    for key in ("size", "width", "height", "image", "images", "decoder"):
        if key in body:
            raise BadRequest(f"{key} does not apply to the t2a pipeline (audio only)")
    fps = float(body.get("fps", cfg["fps"]))
    if not 1.0 <= fps <= 60.0:
        raise BadRequest("fps must be between 1 and 60")
    if "num_frames" in body:
        frames = int(body["num_frames"])
    elif "seconds" in body:
        frames = int(round(float(body["seconds"]) * fps / 8)) * 8 + 1
    else:
        frames = None
    max_frames = cfg["num_frames"]
    if frames is not None and (frames < 9 or (frames - 1) % 8 or frames > max_frames):
        raise BadRequest(f"num_frames must be 8k + 1 up to {max_frames}")
    params: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": frames,
        "fps": fps,
        "seed": int(body.get("seed", 42)),
    }
    if frames is None:
        params["max_num_frames"] = max_frames
    if "negative_prompt" in body:
        params["negative_prompt"] = str(body["negative_prompt"])
    if "steps" in body:
        steps = body["steps"]
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 200
        ):
            raise BadRequest("steps must be an integer in [1, 200]")
        params["steps"] = steps
    if "guidance" in body:
        fields = guidance_fields(body["guidance"], cfg["pipeline"])
        if "video_guidance" in fields:
            raise BadRequest("t2a has no video modality: give guidance.audio only")
        params.update(fields)
    if body.get("enhance_prompt") is True:
        params["enhance_prompt"] = True
    if body.get("loras") is not None:
        params["loras"] = [
            lora_field(item, cfg) for item in _list(body["loras"], "loras")
        ]
    return params


def normalize_source_request(
    body: dict[str, Any], cfg: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """A retake request: `video_path` (a file on the server) or `video`
    (base64 mp4, written next to the outputs), `start_time` / `end_time` in
    seconds, `regenerate_video` / `regenerate_audio` (default true). The
    source's size, frame count and rate are the output's; it must fit the
    profile's token envelope. No size / seconds / fps / images here. A Dub-It
    request (`reference_video` / `video`, `loras`, `reference_strength`)
    takes a size (the reference sets the length and rate)."""
    if cfg["pipeline"] == "dubit":
        return normalize_dubit_request(body, cfg, prompt)
    for key in (
        "size",
        "width",
        "height",
        "seconds",
        "num_frames",
        "image",
        "images",
    ):
        if key in body:
            raise BadRequest(
                f"{key} does not apply to the retake pipeline (the source clip sets it)"
            )
    if body.get("video_path") is not None:
        # a clip, or an EXR frame folder
        video_path = server_path(body["video_path"], "video_path", cfg, dirs=True)
    elif body.get("video") is not None:
        video_path = str(_store_video(body["video"]))
    else:
        raise BadRequest(
            "the retake pipeline needs `video_path` (a file on the server) "
            "or `video` (base64)"
        )
    from slimserve.video.ltx25 import media

    fps_hint = float(body["fps"]) if body.get("fps") is not None else None
    try:
        info = media.probe(video_path, fps_hint)
    except (RuntimeError, ValueError) as exc:
        raise BadRequest(f"cannot read the source clip: {exc}") from exc
    if (info.frames - 1) % 8:
        raise BadRequest(
            f"the source has {info.frames} frames; retake needs 8k + 1 "
            f"(trim it to {(info.frames - 1) // 8 * 8 + 1})"
        )
    if info.width % 32 or info.height % 32:
        raise BadRequest(
            f"the source is {info.width}x{info.height}; sides must be multiples of 32"
        )
    tokens = ((info.frames - 1) // 8 + 1) * (info.height // 32) * (info.width // 32)
    if tokens > cfg["max_video_tokens"] or (info.height // 32) * (info.width // 32) > (
        cfg["height"] // 32
    ) * (cfg["width"] // 32):
        raise BadRequest(
            f"{info.width}x{info.height}x{info.frames} is {tokens} latent tokens; "
            f"this profile is validated up to {cfg['max_video_tokens']} "
            f"({cfg['width']}x{cfg['height']}x{cfg['num_frames']})"
        )
    try:
        start, end = float(body["start_time"]), float(body["end_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BadRequest("start_time and end_time (seconds) are required") from exc
    duration = info.frames / info.fps
    if not 0.0 <= start < end or start >= duration:
        raise BadRequest(
            "the region must satisfy 0 <= start_time < end_time inside the "
            f"clip's {duration:.3f} s"
        )
    params: dict[str, Any] = {
        "prompt": prompt,
        "video_path": video_path,
        "start_time": start,
        "end_time": end,
        "seed": int(body.get("seed", 42)),
        **({"fps": fps_hint} if fps_hint is not None else {}),
        "source": {
            "width": info.width,
            "height": info.height,
            "num_frames": info.frames,
            "fps": info.fps,
        },
    }
    for key in ("regenerate_video", "regenerate_audio"):
        if key in body:
            if not isinstance(body[key], bool):
                raise BadRequest(f"{key} must be true or false")
            params[key] = body[key]
    decoder = str(body.get("decoder", cfg.get("decoder", "diffusion")))
    if decoder not in ("diffusion", "conv"):
        raise BadRequest(
            "decoder must be 'diffusion' (default, sharper) or 'conv' (faster)"
        )
    params["decoder"] = decoder
    if "enhance_prompt" in body:
        if not isinstance(body["enhance_prompt"], bool):
            raise BadRequest("enhance_prompt must be true or false")
        if body["enhance_prompt"]:
            params["enhance_prompt"] = True
    if body.get("loras") is not None:
        params["loras"] = [
            lora_field(item, cfg) for item in _list(body["loras"], "loras")
        ]
    hdr_field(body, params)
    return params


def normalize_hdr_ic_lora_request(
    body: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """SDR to HDR: `video_path` (an mp4, or an EXR frame folder with `fps`),
    `input_colorspace` (srgb_gamma default for display video; srgb; acescg /
    acescct for EXR), `exr_colorspace` (the EXR sidecar's space, default
    acescg), exactly one adapter in `loras` (the SDR-To-HDR IC-LoRA),
    `text_embeddings` (its scene embedding file), `high_quality`,
    `keyframes` (default true), `keyframe_strength`, `conditioning_strength`.
    No prompt is used (the embeddings stand in for it)."""
    from slimserve.video.ltx25 import hdr as hdr_mod
    from slimserve.video.ltx25 import media

    for key in ("size", "width", "height", "seconds", "num_frames", "image", "images"):
        if key in body:
            raise BadRequest(
                f"{key} does not apply to the HDR IC-LoRA (the source sets it)"
            )
    if body.get("video_path") is not None:
        video_path = server_path(body["video_path"], "video_path", cfg, dirs=True)
    elif body.get("video") is not None:
        video_path = str(_store_video(body["video"]))
    else:
        raise BadRequest("the HDR IC-LoRA needs `video_path` (or `video` as base64)")
    fps_hint = float(body["fps"]) if body.get("fps") is not None else None
    try:
        info = media.probe(video_path, fps_hint)
    except (RuntimeError, ValueError) as exc:
        raise BadRequest(f"cannot read the source: {exc}") from exc
    if (info.frames - 1) % 8:
        raise BadRequest(
            f"the source has {info.frames} frames; it must be 8k + 1 "
            f"(trim it to {(info.frames - 1) // 8 * 8 + 1})"
        )
    gen_w, gen_h = -(-info.width // 32) * 32, -(-info.height // 32) * 32
    frames = 2 * info.frames - 1 if body.get("high_quality") else info.frames
    tokens = ((frames - 1) // 8 + 1) * (gen_h // 32) * (gen_w // 32)
    if 2 * tokens > cfg["max_video_tokens"]:  # the reference doubles the sequence
        raise BadRequest(
            f"{info.width}x{info.height}x{info.frames} is {tokens} latent tokens plus "
            f"as many reference tokens; this profile is validated up to "
            f"{cfg['max_video_tokens']}"
        )
    params: dict[str, Any] = {
        "video_path": video_path,
        "seed": int(body.get("seed", 42)),
        "source": {
            "width": info.width,
            "height": info.height,
            "num_frames": info.frames,
            "fps": info.fps,
        },
    }
    if fps_hint is not None:
        params["fps"] = fps_hint
    spaces = ("srgb_gamma", "srgb", "acescg", "acescct")
    ics = str(body.get("input_colorspace", "srgb_gamma")).lower()
    if ics not in spaces:
        raise BadRequest(f"input_colorspace must be one of {', '.join(spaces)}")
    params["input_colorspace"] = ics
    ecs = str(body.get("exr_colorspace", "acescg")).lower()
    if ecs not in hdr_mod.COLOR_SPACES:
        raise BadRequest(
            f"exr_colorspace must be one of {', '.join(hdr_mod.COLOR_SPACES)}"
        )
    params["exr_colorspace"] = ecs
    loras = body.get("loras")
    if not loras or not isinstance(loras, list) or len(loras) != 1:
        raise BadRequest("the HDR IC-LoRA needs exactly one adapter in `loras`")
    params["loras"] = [lora_field(loras[0], cfg)]
    emb = body.get("text_embeddings")
    if not isinstance(emb, str):
        raise BadRequest(
            "`text_embeddings` must name the adapter's scene embedding .safetensors"
        )
    params["text_embeddings"] = server_path(emb, "text_embeddings", cfg)
    for key in ("high_quality", "keyframes"):
        if key in body:
            if not isinstance(body[key], bool):
                raise BadRequest(f"{key} must be true or false")
            params[key] = body[key]
    for key in ("keyframe_strength", "conditioning_strength"):
        if key in body:
            params[key] = _strength(body[key], key)
    decoder = str(body.get("decoder", cfg.get("decoder", "diffusion")))
    if decoder not in ("diffusion", "conv"):
        raise BadRequest(
            "decoder must be 'diffusion' (default, sharper) or 'conv' (faster)"
        )
    params["decoder"] = decoder
    return params


def normalize_dubit_request(
    body: dict[str, Any], cfg: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Dub-It: `reference_video` (a file on the server) or `video` (base64),
    exactly one adapter in `loras` (the Dub-It IC-LoRA), `reference_strength`
    (0-1), a size (multiples of 64), stills; the reference's frame count
    (snapped to 8k + 1) and rate are the output's and must fit the envelope."""
    for key in ("seconds", "num_frames"):
        if key in body:
            raise BadRequest(
                f"{key} does not apply to Dub-It (the reference clip sets it)"
            )
    if body.get("reference_video") is not None:
        ref = server_path(body["reference_video"], "reference_video", cfg, dirs=True)
    elif body.get("video") is not None:
        ref = str(_store_video(body["video"]))
    else:
        raise BadRequest(
            "Dub-It needs `reference_video` (a file on the server) or `video`"
        )
    from slimserve.video.ltx25 import media

    fps_hint = float(body["fps"]) if body.get("fps") is not None else None
    try:
        info = media.probe(ref, fps_hint)
    except (RuntimeError, ValueError) as exc:
        raise BadRequest(f"cannot read the reference clip: {exc}") from exc
    if not info.has_audio:
        raise BadRequest("the reference clip has no audio stream")
    frames = max(1, (info.frames - 1) // 8 * 8 + 1)
    width, height = cfg["width"], cfg["height"]
    if size := body.get("size"):
        try:
            width, height = (int(x) for x in str(size).lower().split("x"))
        except ValueError as exc:
            raise BadRequest("`size` must look like 1536x1024") from exc
    width, height = int(body.get("width", width)), int(body.get("height", height))
    if width % 64 or height % 64 or width < 256 or height < 256:
        raise BadRequest("width and height must be multiples of 64, at least 256")
    per_frame = (height // 32) * (width // 32)
    tokens = ((frames - 1) // 8 + 1) * per_frame
    if tokens > cfg["max_video_tokens"] or per_frame > (cfg["height"] // 32) * (
        cfg["width"] // 32
    ):
        raise BadRequest(
            f"{width}x{height}x{frames} is {tokens} latent tokens; this profile is "
            f"validated up to {cfg['max_video_tokens']} "
            f"({cfg['width']}x{cfg['height']}x{cfg['num_frames']})"
        )
    params: dict[str, Any] = {
        "prompt": prompt,
        "reference_video": ref,
        "width": width,
        "height": height,
        "seed": int(body.get("seed", 42)),
        "source": {
            "width": info.width,
            "height": info.height,
            "num_frames": frames,
            "fps": info.fps,
        },
    }
    if "reference_strength" in body:
        params["reference_strength"] = _strength(
            body["reference_strength"], "reference_strength"
        )
    loras = body.get("loras")
    if not loras or not isinstance(loras, list) or len(loras) != 1:
        raise BadRequest(
            "Dub-It needs exactly one adapter in `loras` (the Dub-It IC-LoRA)"
        )
    params["loras"] = [lora_field(loras[0], cfg)]
    decoder = str(body.get("decoder", cfg.get("decoder", "diffusion")))
    if decoder not in ("diffusion", "conv"):
        raise BadRequest(
            "decoder must be 'diffusion' (default, sharper) or 'conv' (faster)"
        )
    params["decoder"] = decoder
    if body.get("enhance_prompt") is True:
        params["enhance_prompt"] = True
    if body.get("image") is not None:
        params["image"] = decode_image_field(body["image"])
        params["image_strength"] = _strength(body.get("image_strength", 1.0))
    if body.get("images") is not None:
        params["images"] = [
            still_field(item, frames, "dubit", cfg) for item in _list(body["images"])
        ]
    params.update(keyframe_fields(body, "dubit"))
    chunk = chunk_config(body, "dubit")
    if chunk is not None:
        params["chunk"] = chunk
    if fps_hint is not None:
        params["fps_hint"] = fps_hint
    hdr_field(body, params)
    return params


def _store_video(value: Any) -> Path:
    return _store_blob(value, "video", "source")


def _store_blob(value: Any, name: str, stem: str) -> Path:
    """`video` / `audio`: base64 (optionally a data: URL) -> a file under the
    output directory (the engine reads sources from disk; ffmpeg probes the
    container)."""
    data = _b64(value, name)
    target = output_dir() / f"{stem}_{uuid.uuid4().hex[:16]}.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    uploads = _UPLOADS.get()
    if uploads is not None:  # owned by the request being submitted
        uploads.append(target)
    target.write_bytes(data)
    return target


def _remove(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _b64(value: Any, name: str = "video") -> bytes:
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            header, sep, text = text.partition(",")
            if not sep or ";base64" not in header:
                raise BadRequest(f"{name} data: URL must be base64 encoded")
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest(f"{name} must be base64 (optionally a data: URL)") from exc
    else:
        raise BadRequest(f"{name} must be a base64 string")
    if not data:
        raise BadRequest(f"{name} is empty")
    return data


def hdr_field(body: dict[str, Any], params: dict[str, Any]) -> None:
    """`hdr` (upstream --hdr {srgb_linear, acescg, acescct}): the colour space
    of EXR inputs and of the HDR outputs (an EXR frame folder and an HLG
    master in place of the H.264 mp4). Required when any input is an EXR
    still or frame folder."""
    from slimserve.video.ltx25 import hdr as hdr_mod

    value = body.get("hdr")
    if value is not None:
        value = str(value).lower()
        if value not in hdr_mod.COLOR_SPACES:
            raise BadRequest(f"hdr must be one of {', '.join(hdr_mod.COLOR_SPACES)}")
        params["hdr"] = value
    exr = [
        str(s.image)
        for s in params.get("images", [])
        if isinstance(s.image, str) and s.image.lower().endswith(".exr")
    ]
    for key in ("video_path", "reference_video", "attention_mask"):
        p = params.get(key)
        if p and hdr_mod.is_exr_dir(p):
            exr.append(p)
    for p, _ in params.get("video_conditioning", []):
        if hdr_mod.is_exr_dir(p):
            exr.append(p)
    if exr and "hdr" not in params:
        raise BadRequest("EXR inputs need `hdr` (srgb_linear, acescg or acescct)")


def _list(value: Any, name: str = "images") -> list:
    if not isinstance(value, list) or not value:
        raise BadRequest(f"{name} must be a non-empty list")
    return value


def _strength(value: Any, name: str = "image strength", top: float = 1.0) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise BadRequest(f"{name} must be a number in [0, {top:g}]") from exc
    if not 0.0 <= strength <= top:
        raise BadRequest(f"{name} must be a number in [0, {top:g}]")
    return strength


GUIDANCE_KEYS = ("cfg", "stg", "stg_blocks", "rescale", "modality", "skip_step")


def guidance_fields(value: Any, pipeline: str) -> dict[str, Any]:
    """`guidance: {"video": {...}, "audio": {...}}` with upstream's
    MultiModalGuiderParams fields per modality (cfg_scale, stg_scale,
    stg_blocks, rescale_scale, modality_scale (a2v on video, v2a on audio),
    skip_step), over the pipeline's defaults -> video_guidance / audio_guidance."""
    from dataclasses import replace as dc_replace

    from slimserve.video.ltx25 import pipeline as pl

    if not isinstance(value, dict) or not value:
        raise BadRequest('guidance is {"video": {...}, "audio": {...}}')
    unknown = set(value) - {"video", "audio"}
    if unknown:
        raise BadRequest(f"guidance has unknown modalities {sorted(unknown)}")
    defaults = {
        "video": pl.HQ_VIDEO_GUIDANCE
        if pipeline == "hq"
        else pl.DEFAULT_VIDEO_GUIDANCE,
        "audio": pl.HQ_AUDIO_GUIDANCE
        if pipeline == "hq"
        else pl.DEFAULT_AUDIO_GUIDANCE,
    }
    out = {}
    for modality, fields in value.items():
        if not isinstance(fields, dict):
            raise BadRequest(f"guidance.{modality} must be an object")
        bad = set(fields) - set(GUIDANCE_KEYS)
        if bad:
            raise BadRequest(
                f"guidance.{modality} has unknown fields {sorted(bad)}; "
                f"known: {', '.join(GUIDANCE_KEYS)}"
            )
        clean = {}
        for key, raw in fields.items():
            if key == "stg_blocks":
                if not isinstance(raw, list) or not all(
                    isinstance(b, int) and not isinstance(b, bool) and 0 <= b < 48
                    for b in raw
                ):
                    raise BadRequest("stg_blocks is a list of block indices in [0, 48)")
                clean[key] = tuple(raw)
            elif key == "skip_step":
                if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
                    raise BadRequest("skip_step must be a non-negative integer")
                clean[key] = raw
            else:
                try:
                    clean[key] = float(raw)
                except (TypeError, ValueError) as exc:
                    raise BadRequest(
                        f"guidance.{modality}.{key} must be a number"
                    ) from exc
                if key == "rescale" and not 0.0 <= clean[key] <= 1.0:
                    raise BadRequest("rescale must be in [0, 1]")
                if key != "rescale" and clean[key] < 0.0:
                    raise BadRequest(f"{key} must be non-negative")
        out[f"{modality}_guidance"] = dc_replace(defaults[modality], **clean)
    return out


A2VID_KEYS = ("audio_path", "audio", "audio_start_time", "audio_max_duration")


def a2vid_fields(
    body: dict[str, Any], params: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """The A2Vid request: `audio_path` (a file on the server; any container
    with an audio stream) or `audio` (base64), `audio_start_time` (s),
    `audio_max_duration` (s; not with a clip length, as upstream). Without a
    length the clip follows the audio (capped at the envelope)."""
    out: dict[str, Any] = {}
    if body.get("audio_path") is not None:
        out["audio_path"] = server_path(body["audio_path"], "audio_path", cfg)
    elif body.get("audio") is not None:
        out["audio_path"] = str(_store_blob(body["audio"], "audio", "source"))
    else:
        raise BadRequest(
            "the a2vid pipeline needs `audio_path` (a file on the server) "
            "or `audio` (base64)"
        )
    for key in ("audio_start_time", "audio_max_duration"):
        if key in body:
            try:
                value = float(body[key])
            except (TypeError, ValueError) as exc:
                raise BadRequest(f"{key} must be a number of seconds") from exc
            if value < 0 or (key == "audio_max_duration" and value <= 0):
                raise BadRequest(f"{key} must be positive")
            out[key] = value
    if "audio_max_duration" in out and params.get("num_frames") is not None:
        raise BadRequest(
            "audio_max_duration and a clip length (seconds / num_frames) are exclusive"
        )
    return out


CHUNK_PIPELINES = ("distilled", "dev", "a2vid", "ic_lora", "dubit")
CHUNK_KEYS = (
    "chunked",
    "chunk_pixel_frames",
    "chunk_carry_frames",
    "chunk_blend_frames",
)


def chunk_config(body: dict[str, Any], pipeline: str):
    """Upstream chunk_config_from_args: `chunked` (the default layout: 97-frame
    windows, 25-frame carry, a crossfade over the whole carry) or
    `chunk_pixel_frames` / `chunk_carry_frames` (either implies the other's
    default) and `chunk_blend_frames`; None when none is given."""
    given = [k for k in CHUNK_KEYS if k in body]
    if not given:
        return None
    if pipeline not in CHUNK_PIPELINES:
        raise BadRequest(
            f"{', '.join(given)}: chunked generation applies to "
            f"{', '.join(CHUNK_PIPELINES)}"
        )
    from slimserve.video.ltx25.chunks import ChunkConfig

    def count(key, default):
        if key not in body:
            return default
        value = body[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise BadRequest(f"{key} must be a non-negative integer")
        return value

    if "chunked" in body and not isinstance(body["chunked"], bool):
        raise BadRequest("chunked must be true or false")
    if body.get("chunked") is False and len(given) == 1:
        return None
    frames = count("chunk_pixel_frames", 97)
    carry = count("chunk_carry_frames", 25)
    blend = count("chunk_blend_frames", None)
    for name, value in (("chunk_pixel_frames", frames), ("chunk_carry_frames", carry)):
        if (value - 1) % 8:
            raise BadRequest(f"{name} must be on the 8k + 1 grid (e.g. 97, 25)")
    if carry < 17 or carry >= frames:
        raise BadRequest(
            "chunk_carry_frames must be at least 17 and less than chunk_pixel_frames"
        )
    try:
        return ChunkConfig(frames, carry, blend)
    except ValueError as exc:
        raise BadRequest(str(exc)) from exc


KEYFRAME_PIPELINES = (
    "distilled",
    "dev",
    "hq",
    "one_stage",
    "a2vid",
    "ic_lora",
    "dubit",
)


def keyframe_fields(body: dict[str, Any], pipeline: str) -> dict[str, Any]:
    """`generated_keyframes` (a count of evenly spaced interior keyframe slots,
    or a list of pixel frames; upstream --num-generated-keyframes) and
    `decode_with_keyframes` (anchor the diffusion decode on them; needs
    slots) on the pipelines upstream offers them on (DFR has its own)."""
    out: dict[str, Any] = {}
    if "generated_keyframes" in body:
        if pipeline not in KEYFRAME_PIPELINES:
            raise BadRequest(
                f"generated_keyframes applies to {', '.join(KEYFRAME_PIPELINES)}"
            )
        value = body["generated_keyframes"]
        if isinstance(value, bool) or not (
            (isinstance(value, int) and value >= 0)
            or (
                isinstance(value, list)
                and value
                and all(
                    isinstance(v, int) and not isinstance(v, bool) and v >= 0
                    for v in value
                )
            )
        ):
            raise BadRequest(
                "generated_keyframes is a non-negative count or a list of pixel frames"
            )
        if value:
            out["generated_keyframes"] = value
    if "decode_with_keyframes" in body:
        if not isinstance(body["decode_with_keyframes"], bool):
            raise BadRequest("decode_with_keyframes must be true or false")
        if body["decode_with_keyframes"]:
            if not out.get("generated_keyframes"):
                raise BadRequest("decode_with_keyframes needs generated_keyframes")
            out["decode_with_keyframes"] = True
    return out


IC_LORA_KEYS = (
    "video_conditioning",
    "attention_strength",
    "attention_mask",
    "skip_stage_2",
    "stage_2_ic_lora",
    "tile",
    "tile_height",
    "tile_width",
)


def ic_lora_fields(
    body: dict[str, Any], params: dict[str, Any], cfg: dict[str, Any]
) -> dict[str, Any]:
    """The IC-LoRA request: `video_conditioning` ([{"path", "strength"}],
    reference clips on the server; required), `loras` (the IC-LoRA adapters;
    required), `attention_strength` (0-1), `attention_mask` (a grayscale mask
    video on the server), `skip_stage_2`, `stage_2_ic_lora`, `tile`,
    `tile_height` / `tile_width` (multiples of 32)."""
    out: dict[str, Any] = {}
    refs = body.get("video_conditioning")
    if not refs:
        raise BadRequest(
            "the ic_lora pipeline needs `video_conditioning` "
            '([{"path": ..., "strength": 1.0}])'
        )
    out["video_conditioning"] = []
    for item in _list(refs, "video_conditioning"):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise BadRequest('each video_conditioning entry needs a "path"')
        path = server_path(item["path"], "video_conditioning", cfg, dirs=True)
        out["video_conditioning"].append(
            (path, _strength(item.get("strength", 1.0), "reference strength"))
        )
    if not params.get("loras"):
        raise BadRequest("the ic_lora pipeline needs `loras` (the IC-LoRA adapter)")
    if "attention_strength" in body:
        out["attention_strength"] = _strength(
            body["attention_strength"], "attention_strength"
        )
    if body.get("attention_mask") is not None:
        out["attention_mask"] = server_path(
            body["attention_mask"], "attention_mask", cfg
        )
    for key in ("skip_stage_2", "stage_2_ic_lora", "tile"):
        if key in body:
            if not isinstance(body[key], bool):
                raise BadRequest(f"{key} must be true or false")
            if body[key]:
                out[key] = True
    for key in ("tile_height", "tile_width"):
        if key in body:
            value = body[key]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 64
                or value % 32
            ):
                raise BadRequest(f"{key} must be a multiple of 32, at least 64")
            out[key] = value
    return out


def lora_field(item: Any, cfg: dict[str, Any]) -> tuple[str, float]:
    """One entry of `loras`: {"path": <.safetensors on the server>, "strength": 1.0}
    (upstream --lora PATH [STRENGTH]); the file is checked when the engine
    loads it."""
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise BadRequest('each loras entry needs a "path"')
    if not item["path"].endswith(".safetensors"):
        raise BadRequest("loras: each path must be a .safetensors file")
    path = server_path(item["path"], "loras", cfg)
    return path, _strength(item.get("strength", 1.0), "lora strength", 2.0)


def still_field(item: Any, frames: int | None, pipeline: str, cfg: dict[str, Any]):
    """One entry of `images`: {"image": <base64 or data: URL>, "frame": N,
    "strength": 0-1, "crf": int} -> sampling.Still (upstream's --image PATH
    FRAME_IDX STRENGTH [CRF]). Frames must be inside the clip when its length
    is known; the engine checks again once the duration head has decided."""
    from slimserve.video.ltx25.sampling import Still

    if not isinstance(item, dict) or (item.get("image") is None) == (
        item.get("path") is None
    ):
        raise BadRequest('each images entry needs an "image" (base64) or a "path"')
    frame = item.get("frame", 0)
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise BadRequest("image frame must be a non-negative integer")
    if frames is not None and frame >= frames:
        raise BadRequest(f"image frame {frame} is outside the clip's {frames} frames")
    crf = item.get("crf")
    if crf is not None and (not isinstance(crf, int) or not 0 <= crf <= 51):
        raise BadRequest("image crf must be an integer in [0, 51] (0: no round trip)")
    if item.get("path") is not None:
        source: Any = server_path(item["path"], "images", cfg)
    else:
        source = decode_image_field(item["image"])
    return Still(source, frame, _strength(item.get("strength", 1.0)), crf)


def _first_still(params: dict[str, Any]):
    """The still the prompt enhancer looks at: upstream passes images[0]."""
    if params.get("image") is not None:
        return params["image"]
    if params.get("images"):
        return params["images"][0].image
    return None


def fast_settings(cfg: dict[str, Any]):
    """The profile's fast tier (`engine.fast`), or None for the exact pipeline."""
    if not cfg.get("fast"):
        return None
    from slimserve.video.ltx25.pipeline import Fast

    fast = Fast.from_config(cfg["fast"])
    return fast if fast.active else None


def decode_image_field(value: Any) -> bytes:
    """`image`: encoded image bytes as base64, optionally a data: URL; raw bytes
    (the CLI's file contents) pass through. The bytes are decoded as an image by
    the engine; here they must at least be a non-empty, valid base64 payload."""
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            header, sep, text = text.partition(",")
            if not sep or ";base64" not in header:
                raise BadRequest("image data: URL must be base64 encoded")
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest("image must be base64 (optionally a data: URL)") from exc
    else:
        raise BadRequest("image must be a base64 string")
    if not data:
        raise BadRequest("image is empty")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except ImportError:
        pass  # the engine's decoder reports it instead
    except Exception as exc:
        raise BadRequest(f"image is not a decodable image: {exc}") from exc
    return data


class VideoService:
    """The queue, the worker and the job table. HTTP-free so tests can drive it."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model_name: str,
        output_dir: Path,
        engine_factory=None,
        input_root: str | os.PathLike | None = None,
        max_body: int = DEFAULT_MAX_BODY_BYTES,
        max_preparing: int | None = None,
    ):
        # Requests may name server files only under input_root (None: never).
        cfg = {**cfg, INPUT_ROOT: resolve_input_root(input_root)}
        self.cfg, self.model_name, self.output_dir = cfg, model_name, output_dir
        self.max_body = max_body
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        if max_preparing is None:
            env = os.environ.get("SLIMSERVE_VIDEO_MAX_PREPARING")
            max_preparing = int(env) if env else DEFAULT_MAX_PREPARING
        if max_preparing < 1:
            raise ValueError("max_preparing must be at least 1")
        self.preparing = threading.BoundedSemaphore(max_preparing)
        self.stopped = False
        self.running: Job | None = None
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.ready = threading.Event()
        self.load_error: str | None = None
        self.completed = 0
        self.failed = 0
        self._engine_factory = engine_factory or self._default_engine
        self.worker = threading.Thread(
            target=self._run, name="ltx25-worker", daemon=True
        )
        self.worker.start()

    def _default_engine(self):
        from slimserve.video.ltx25.pipeline import LTX25Engine

        pipeline = self.cfg["pipeline"]
        engine = LTX25Engine(
            root=self.cfg.get("root"),
            variant="dev" if pipeline in GUIDED_PIPELINES else "distilled",
            user_lora_cache=self.cfg.get("user_lora_cache"),
        )
        if pipeline != "hdr_ic_lora":  # no prompt: the scene embeddings stand in
            engine.load_text()
        engine.load_dit()
        engine.load_vae()
        if pipeline not in ("one_stage", "t2a", "alpha", *SOURCE_PIPELINES):
            engine.load_upscaler()
        if pipeline not in ("alpha", "hdr_ic_lora"):
            engine.load_audio(encoder=pipeline in ("retake", "dubit", "a2vid"))
        engine.load_duration()
        if self.cfg.get("decoder", "diffusion") == "diffusion" and pipeline != "t2a":
            engine.load_diffvae()
        if pipeline in ("dev", "hq", "keyframes", "a2vid"):
            engine.load_distilled_lora()
        elif pipeline == "dfr":
            engine.load_detail_lora()
            engine.load_temporal_upscaler()
        return engine

    # ---- worker (the only thread that touches the engine) -----------------
    def _run(self) -> None:
        try:
            engine = self._engine_factory()
        except (
            Exception
        ) as exc:  # surfaced through /health; the process stays up to report it
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.queue.get()
            if job is None:
                return
            with self.lock:
                if job.status != "queued":  # failed by stop() while waiting
                    continue
                job.status, job.started_at = "in_progress", time.time()
                self.running = job
            try:
                self._generate(engine, job)
                job.status = "completed"
                self.completed += 1
            except Exception as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                self.failed += 1
            finally:
                _remove(job.uploads)  # the uploaded sources are spent either way
                with self.lock:
                    self.running = None
            job.completed_at = time.time()
            job.done.set()
            self._trim()

    def _generate(self, engine, job: Job) -> None:
        p = dict(job.params)
        prompt = p.pop("prompt", None)
        enhance_s = None
        if p.pop("enhance_prompt", False):
            job.progress = {"stage": "enhance"}
            t0 = time.perf_counter()
            # loaded for the rewrite and dropped: the resident set stays the
            # measured one (a dev HD request sits near the active cap)
            prompt = engine.enhance(prompt, _first_still(p))
            engine.unload_enhancer()
            enhance_s = time.perf_counter() - t0
            job.params["enhanced_prompt"] = prompt

        def on_step(stage: str, index: int, sigma: float) -> None:
            job.progress = {"stage": stage, "step": index + 1}

        decoder = p.pop("decoder", None)
        p.pop("source", None)  # reported, not an engine argument
        fast = fast_settings(self.cfg)
        if fast is not None:
            p["fast"] = fast
        # the IC-LoRA pipelines manage their adapters per stage themselves
        loras = (
            None
            if self.cfg["pipeline"] in ("ic_lora", "dubit", "alpha", "hdr_ic_lora")
            else p.pop("loras", None)
        )
        if p.get("chunk") is not None:
            p["decoder"] = decoder  # chunked clips decode per window inside
        with engine.user_loras(loras):
            if prompt is None:  # the HDR IC-LoRA takes no prompt
                result = getattr(engine, self.cfg["pipeline"])(on_step=on_step, **p)
            else:
                result = getattr(engine, self.cfg["pipeline"])(
                    prompt, on_step=on_step, **p
                )
        if result.predicted_seconds is not None:  # auto duration: report the pick
            job.params["num_frames"] = result.num_frames
            job.params["predicted_seconds"] = round(result.predicted_seconds, 2)
        job.progress = {"stage": "decode"}
        suffix = ".wav" if getattr(result, "video_latent", 0) is None else ".mp4"
        path = self.output_dir / f"{job.id}{suffix}"
        engine.render(result, path, seed=p["seed"], decoder=decoder, hdr=p.get("hdr"))
        job.path = path
        tm = result.timings
        spans = {k: round(v, 2) for k, v in tm.spans.items()}
        if enhance_s is not None:
            spans = {"enhance": round(enhance_s, 2), **spans}
        job.timings = {
            "spans_s": spans,
            "peak_gib": round(
                max((m[0] for m in tm.memory.values()), default=0) / 2**30, 1
            ),
        }
        job.progress = {"stage": "done"}

    def _trim(self) -> None:
        with self.lock:
            finished = [
                i for i in self.order if self.jobs[i].status in ("completed", "failed")
            ]
            for old in finished[:-KEEP_FINISHED]:
                self._drop(old)

    def _drop(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        if job_id in self.order:
            self.order.remove(job_id)
        if job:
            _remove(job.uploads)
            if job.path:
                job.path.unlink(missing_ok=True)

    # ---- API ---------------------------------------------------------------
    def _queue_full(self) -> None:
        if self.stopped:
            raise ServiceStopped("the server is shutting down")
        waiting = sum(1 for j in self.jobs.values() if j.status == "queued")
        if waiting >= MAX_QUEUED:
            raise QueueFull(f"{waiting} requests already queued")

    def submit(self, body: dict[str, Any]) -> Job:
        # Refuse a full queue before decoding anything: normalising writes the
        # request's base64 sources to disk.
        with self.lock:
            self._queue_full()
        # Bound the requests decoding and writing sources at once: each can
        # hold a body-sized upload in memory and on disk before it queues.
        if not self.preparing.acquire(blocking=False):
            raise QueueFull("too many requests are being prepared; retry shortly")
        try:
            return self._submit(body)
        finally:
            self.preparing.release()

    def _submit(self, body: dict[str, Any]) -> Job:
        uploads: list[Path] = []
        token = _UPLOADS.set(uploads)
        try:
            params = normalize_request(body, self.cfg)
            if params.get("enhance_prompt") and not self.cfg.get(
                "prompt_enhancer", True
            ):
                raise BadRequest(
                    "enhance_prompt: the prompt enhancer is not installed on this "
                    "server (start it with --prompt-enhancer to fetch it)"
                )
            with self.lock:
                self._queue_full()  # another request may have filled it meanwhile
                job = Job(
                    id="video_" + uuid.uuid4().hex[:24], params=params, uploads=uploads
                )
                self.jobs[job.id] = job
                self.order.append(job.id)
                # under the lock: stop() puts its sentinel under it too, so a
                # job is never queued behind the sentinel (it would never run)
                self.queue.put(job)
        except BaseException:
            _remove(uploads)
            raise
        finally:
            _UPLOADS.reset(token)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status in ("queued", "in_progress"):
                return False
            self._drop(job_id)
            return True

    def health(self) -> dict[str, Any]:
        state = (
            "error" if self.load_error else ("ok" if self.ready.is_set() else "loading")
        )
        counts = {
            s: sum(1 for j in self.jobs.values() if j.status == s)
            for s in ("queued", "in_progress")
        }
        out = {
            "status": state,
            "model": self.model_name,
            "pipeline": self.cfg["pipeline"],
            **counts,
            "completed": self.completed,
            "failed": self.failed,
        }
        if self.load_error:
            out["error"] = self.load_error
        try:  # counters only; safe from any thread
            import mlx.core as mx

            out["memory_gib"] = {
                "active": round(mx.get_active_memory() / 2**30, 1),
                "cache": round(mx.get_cache_memory() / 2**30, 1),
            }
        except ImportError:
            pass
        return out

    def stop(self) -> None:
        with self.lock:  # queued jobs will not run: fail them, free their uploads
            self.stopped = True  # later submits are refused
            for job in self.jobs.values():
                if job.status == "queued":
                    job.status = "failed"
                    job.error = "the server stopped before this job ran"
                    job.completed_at = time.time()
                    _remove(job.uploads)
                    job.done.set()
            # The worker is a daemon thread: the process exits under the job
            # it is running, so its finally never removes that job's sources.
            if self.running is not None:
                _remove(self.running.uploads)
            self.queue.put(None)


def make_handler(service: VideoService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(
            self, fmt: str, *args: Any
        ) -> None:  # quiet; the CLI prints job lines
            pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str) -> None:
            self._json(code, {"error": {"message": message, "code": code}})

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                health = service.health()
                return self._json(200 if health["status"] == "ok" else 503, health)
            if path == "/v1/models":
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": service.model_name, "object": "model"}],
                    },
                )
            parts = path.split("/")
            if len(parts) >= 4 and parts[1:3] == ["v1", "videos"]:
                job = service.get(parts[3])
                if job is None:
                    return self._error(404, "no such video")
                if len(parts) == 4:
                    return self._json(200, job.public(service.model_name))
                if parts[4] == "content":
                    if job.status != "completed" or job.path is None:
                        return self._error(409, f"video is {job.status}")
                    data = job.path.read_bytes()
                    self.send_response(200)
                    kind = "audio/wav" if job.path.suffix == ".wav" else "video/mp4"
                    self.send_header("Content-Type", kind)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self._error(404, "not found")

        def do_POST(self) -> None:
            if self.path.split("?")[0].rstrip("/") != "/v1/videos":
                return self._error(404, "not found")
            if service.load_error:
                return self._error(503, service.load_error)
            if self.headers.get("Transfer-Encoding") is not None:
                # chunked bodies are not read: without a length the request
                # would be taken as empty and its body left on the connection
                self.close_connection = True
                if self.headers.get("Content-Length") is None:
                    return self._error(411, "a Content-Length is required")
                return self._error(400, "send Content-Length without Transfer-Encoding")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.close_connection = True
                return self._error(400, "Content-Length must be an integer")
            if length < 0:
                self.close_connection = True
                return self._error(400, "Content-Length must be an integer")
            if length > service.max_body:
                # the body is never read: drop the connection after answering
                self.close_connection = True
                return self._error(
                    413, f"request body is larger than {service.max_body} bytes"
                )
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise BadRequest("the request body must be a JSON object")
                job = service.submit(body)
            except QueueFull as exc:
                return self._error(429, str(exc))
            except ServiceStopped as exc:
                return self._error(503, str(exc))
            except (BadRequest, json.JSONDecodeError) as exc:
                return self._error(400, str(exc))
            except UnicodeDecodeError:
                return self._error(400, "the request body must be UTF-8 JSON")
            except (ValueError, TypeError, OverflowError) as exc:
                # a field of the wrong type or out of range (int("wide"),
                # float(None), round(inf)): the client's mistake, not a crash
                return self._error(400, f"malformed request field: {exc}")
            if body.get("wait"):
                job.done.wait()
            self._json(200, job.public(service.model_name))

        def do_DELETE(self) -> None:
            parts = self.path.split("?")[0].rstrip("/").split("/")
            if len(parts) == 4 and parts[1:3] == ["v1", "videos"]:
                if service.delete(parts[3]):
                    return self._json(200, {"id": parts[3], "deleted": True})
                return self._error(
                    409, "no such video, or it is still queued or running"
                )
            self._error(404, "not found")

    return Handler


def output_dir() -> Path:
    return Path(
        os.environ.get("SLIMSERVE_VIDEO_DIR", "~/.cache/slimserve/videos")
    ).expanduser()


def serve(
    cfg: dict[str, Any],
    model_name: str,
    host: str,
    port: int,
    input_root: str | os.PathLike | None = None,
    max_body: int = DEFAULT_MAX_BODY_BYTES,
) -> int:
    service = VideoService(
        cfg, model_name, output_dir(), input_root=input_root, max_body=max_body
    )
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        httpd.server_close()
    return 0
