# SPDX-License-Identifier: Apache-2.0
"""`slimserve <video profile>`: serve, or render one clip with --prompt."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from slimserve import term
from slimserve.registry import Plan

USAGE = "slimserve VIDEO_PROFILE [options] [video options]"


def parser() -> argparse.ArgumentParser:
    """The options only a video profile takes.

    `slimserve.cli` parses the options every profile shares (--prompt, --serve,
    --host, --port, --quant, --cache, --served-model-name, --dry-run, -y,
    --download-only) and hands what is left to this parser once the profile
    is known to be a video one, so the LLM CLI neither lists nor silently
    accepts these.
    """
    parser = argparse.ArgumentParser(
        prog="slimserve",
        usage=USAGE,
        description=(
            "Render one clip with --prompt, or serve POST /v1/videos. The shared "
            "options (--prompt, --serve, --host, --port, --quant, --cache, "
            "--served-model-name, --dry-run, -y, --download-only) are listed by "
            "`slimserve --help`."
        ),
        add_help=False,
    )
    video = parser.add_argument_group(
        "video options (with --prompt, or per request when serving)"
    )
    video.add_argument(
        "--output", help="where to write the clip (default <pipeline>-<seed>.mp4)"
    )
    video.add_argument(
        "--size", help="WIDTHxHEIGHT, multiples of 64 (default: the profile's)"
    )
    video.add_argument(
        "--seconds", type=float, help="clip length (default: predicted from the prompt)"
    )
    video.add_argument("--seed", type=int, help="random seed (default 42)")
    video.add_argument(
        "--negative-prompt",
        help="guided pipelines only (dev, hq, keyframes, one_stage)",
    )
    video.add_argument(
        "--enhance-prompt",
        action="store_true",
        help="rewrite the prompt into the model's caption style first "
        "(Gemma-4 E2B-it, looking at --image when given; upstream's "
        "--enhance-prompt)",
    )
    video.add_argument(
        "--temporal-upscalings",
        type=int,
        choices=(0, 1, 2),
        help="DFR only: x2 frame-rate rounds (each re-denoises the clip at 2x fps)",
    )
    video.add_argument(
        "--spatial-upscalings",
        type=int,
        choices=(1, 2),
        help="DFR only: 2 = stage 1 at a quarter, stage 2 at half, then the tiled "
        "full-resolution detailing epilogue (sizes multiples of 128)",
    )
    video.add_argument(
        "--image",
        nargs="+",
        action="append",
        metavar=("PATH", "FRAME STRENGTH CRF"),
        help="a conditioning still (repeatable): PATH [FRAME [STRENGTH [CRF]]]. "
        "Frame 0 (the default) is image-to-video; another frame pins that "
        "moment as a keyframe; CRF overrides the H.264 round trip (0: none)",
    )
    video.add_argument(
        "--image-strength",
        type=float,
        help="strength of the first --image, 0-1 (default 1.0)",
    )
    video.add_argument("--video-path", help="retake: the source clip (mp4/mov)")
    video.add_argument(
        "--start-time", type=float, help="retake: start of the region to regenerate (s)"
    )
    video.add_argument(
        "--end-time", type=float, help="retake: end of the region to regenerate (s)"
    )
    video.add_argument(
        "--hdr",
        choices=["srgb_linear", "acescg", "acescct"],
        help="HDR: the colour space of EXR inputs (stills, frame folders) and of the "
        "outputs (an EXR frame folder beside an HLG master in place of the mp4)",
    )
    video.add_argument(
        "--frame-rate",
        type=float,
        help="the clip's frame rate (default 24; required for EXR frame-folder "
        "sources, which carry none)",
    )
    video.add_argument(
        "--input-colorspace",
        choices=["srgb_gamma", "srgb", "acescg", "acescct"],
        help="hdr-ic-lora: the source's encoding (srgb_gamma: display mp4; srgb: "
        "linear mp4 or EXR; acescg / acescct: EXR folders)",
    )
    video.add_argument(
        "--exr-colorspace",
        choices=["srgb_linear", "acescg", "acescct"],
        help="hdr-ic-lora: the EXR sidecar's colour space (default acescg)",
    )
    video.add_argument(
        "--text-embeddings",
        help="hdr-ic-lora: the adapter's scene embedding .safetensors",
    )
    video.add_argument(
        "--high-quality",
        action="store_true",
        help="hdr-ic-lora: generate 2N - 1 frames",
    )
    video.add_argument(
        "--no-keyframes", action="store_true", help="hdr-ic-lora: no DFR seam keyframes"
    )
    video.add_argument(
        "--keyframe-strength", type=float, help="hdr-ic-lora: seam keyframe strength"
    )
    video.add_argument(
        "--chunked",
        action="store_true",
        help="long clips in overlapping temporal windows (97-frame windows, 25-frame "
        "carry, crossfade over the carry); distilled, dev, a2vid",
    )
    video.add_argument(
        "--chunk-pixel-frames", type=int, help="chunked: frames per window (8k + 1)"
    )
    video.add_argument(
        "--chunk-carry-frames",
        type=int,
        help="chunked: frames each window hands to the next (8k + 1, at least 17)",
    )
    video.add_argument(
        "--chunk-blend-frames",
        type=int,
        help="chunked: decoded frames crossfaded at each seam (default: the carry)",
    )
    video.add_argument(
        "--num-generated-keyframes",
        type=int,
        help="extra generated keyframe slots at evenly spaced interior frames "
        "(distilled, dev, hq, one-stage, a2vid, ic-lora, dubit; DFR has its own)",
    )
    video.add_argument(
        "--decode-with-keyframes",
        action="store_true",
        help="anchor the diffusion decode on the generated keyframes",
    )
    video.add_argument("--audio-path", help="a2vid: the driving audio (any container)")
    video.add_argument(
        "--audio-start-time", type=float, help="a2vid: where in the audio to start (s)"
    )
    video.add_argument(
        "--audio-max-duration",
        type=float,
        help="a2vid: at most this many seconds of audio (not with --seconds)",
    )
    video.add_argument(
        "--reference-video", help="dubit: the clip to dub (video + audio)"
    )
    video.add_argument(
        "--reference-strength",
        type=float,
        help="dubit: strength of the video reference tokens (default 1.0)",
    )
    video.add_argument(
        "--video-conditioning",
        nargs="+",
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="ic_lora: a reference video (the control signal) and its strength "
        "(default 1.0), repeatable",
    )
    video.add_argument(
        "--conditioning-attention-mask",
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        help="ic_lora: a grayscale mask video scaling the reference attention per "
        "region, times STRENGTH",
    )
    video.add_argument(
        "--skip-stage-2", action="store_true", help="ic_lora: ship stage 1 (half size)"
    )
    video.add_argument(
        "--stage-2-ic-lora",
        action="store_true",
        help="ic_lora: keep the adapter and the references in stage 2",
    )
    video.add_argument(
        "--tile",
        action="store_true",
        help="ic_lora: run each transformer call over pinned spatial windows",
    )
    video.add_argument("--tile-height", type=int, help="ic_lora --tile: window height")
    video.add_argument("--tile-width", type=int, help="ic_lora --tile: window width")
    video.add_argument(
        "--lora",
        nargs="+",
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="a user LoRA (.safetensors, ComfyUI LTXV layout) and optional "
        "strength (default 1.0), repeatable; applied in every stage",
    )
    video.add_argument(
        "--num-inference-steps",
        type=int,
        help="guided pipelines: denoising steps of the guided stage (default 30; "
        "hq 15)",
    )
    for modality, other in (("video", "a2v"), ("audio", "v2a")):
        video.add_argument(
            f"--{modality}-cfg-guidance-scale",
            type=float,
            help=f"guided pipelines: {modality} CFG scale (1.0 = off)",
        )
        video.add_argument(
            f"--{modality}-stg-guidance-scale",
            type=float,
            help=f"guided pipelines: {modality} STG scale (0.0 = off)",
        )
        video.add_argument(
            f"--{modality}-stg-blocks",
            type=int,
            nargs="+",
            help=f"guided pipelines: transformer blocks perturbed for {modality} STG",
        )
        video.add_argument(
            f"--{modality}-rescale-scale",
            type=float,
            help=f"guided pipelines: {modality} guidance rescale, 0-1",
        )
        video.add_argument(
            f"--{other}-guidance-scale",
            type=float,
            help=f"guided pipelines: {other} cross-modality guidance scale (1.0 = off)",
        )
        video.add_argument(
            f"--{modality}-skip-step",
            type=int,
            help=f"guided pipelines: run the {modality} stream only on every "
            "(N + 1)th step (0 = every step)",
        )
    video.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        help="hq only: distilled LoRA strength in stage 1 (default 0.25)",
    )
    video.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        help="hq only: distilled LoRA strength in stage 2 (default 0.5)",
    )
    video.add_argument(
        "--decoder",
        choices=["diffusion", "conv"],
        help=(
            "video decoder: diffusion (default, sharper) "
            "or conv (about 4x faster decode)"
        ),
    )
    serving = parser.add_argument_group("video serving")
    serving.add_argument(
        "--prompt-enhancer",
        action="store_true",
        help="fetch the prompt enhancer (google/gemma-4-E2B-it, 10.2 GB) so "
        "requests may set enhance_prompt; --enhance-prompt implies it",
    )
    serving.add_argument(
        "--input-root",
        default=os.environ.get("SLIMSERVE_VIDEO_INPUT_ROOT"),
        metavar="DIR",
        help="serving: the directory request fields that name server files "
        "(video_path, loras, ...) may read from (default "
        "$SLIMSERVE_VIDEO_INPUT_ROOT; unset: such fields are refused)",
    )
    serving.add_argument(
        "--max-body-mb",
        type=float,
        metavar="MIB",
        help="serving: the largest request body accepted, in MiB (default "
        "$SLIMSERVE_VIDEO_MAX_BODY_MB or 256); larger bodies get a 413",
    )
    return parser


ENHANCER = "prompt_enhancer"


def prepare(plan: Plan, args: Any) -> Plan:
    """The plan this run fetches: the prompt enhancer only when asked for."""
    wanted = getattr(args, "enhance_prompt", False) or getattr(
        args, "prompt_enhancer", False
    )
    if wanted and ENHANCER not in plan.optional:
        plan = replace(plan, optional=(*plan.optional, ENHANCER))
    return plan


def optional_ready(plan: Plan, group: str) -> bool:
    """Whether every file of an optional group is already on disk."""
    entries = (plan.source.get("optional") or {}).get(group, [])
    for entry in entries:
        path = plan.model_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            return False
    return bool(entries)


def engine_config(plan: Plan) -> dict[str, Any]:
    cfg = dict(plan.engine)
    cfg["root"] = plan.model_dir
    # requests may ask for enhancement only when the enhancer is installed
    cfg["prompt_enhancer"] = optional_ready(plan, ENHANCER)
    return cfg


def run(plan: Plan, args: Any) -> int:
    cfg = engine_config(plan)
    name = cfg.get("served_model_name", plan.profile_id)
    if args.chat:
        term.fail(
            f"{plan.profile_id} generates video; use --prompt for one clip or serve it"
        )
        return 2
    one_clip = args.prompt or (
        cfg["pipeline"] == "hdr_ic_lora" and getattr(args, "video_path", None)
    )
    if one_clip:
        return _one_clip(cfg, args)
    from slimserve.video import server

    term.ok(
        f"serving {name} ({cfg['pipeline']}) on http://{args.host}:{args.port}"
        "  POST /v1/videos"
    )
    try:
        max_body = server.max_body_bytes(getattr(args, "max_body_mb", None))
        input_root = server.resolve_input_root(getattr(args, "input_root", None))
    except ValueError as error:
        term.fail(str(error))
        return 2
    if input_root:
        term.note(f"requests may name files under {input_root}")
    return server.serve(
        cfg, name, args.host, args.port, input_root=input_root, max_body=max_body
    )


def _add_images(body: dict[str, Any], args: Any) -> None:
    """--image PATH [FRAME [STRENGTH [CRF]]] (repeatable, upstream's form): a
    lone frame-0 still is the `image` shorthand (with --image-strength), the
    rest go through `images`."""
    groups = getattr(args, "image", None) or []
    strength_flag = getattr(args, "image_strength", None)
    if strength_flag is not None and not groups:
        raise ValueError("--image-strength needs --image")
    stills = []
    for group in groups:
        if not 1 <= len(group) <= 4:
            raise ValueError("--image takes PATH [FRAME [STRENGTH [CRF]]]")
        path = Path(group[0]).expanduser()
        if path.suffix.lower() == ".exr":  # HDR stills travel by path
            if not path.is_file():
                raise ValueError(f"cannot read --image {path}")
            data = None
        else:
            try:
                data = path.read_bytes()
            except OSError as error:
                raise ValueError(f"cannot read --image {path}: {error}") from error
        try:
            frame = int(group[1]) if len(group) > 1 else 0
            strength = float(group[2]) if len(group) > 2 else None
            crf = int(group[3]) if len(group) > 3 else None
        except ValueError as error:
            raise ValueError(f"--image {path}: {error}") from error
        entry = {"frame": frame, "strength": strength, "crf": crf}
        entry["image" if data is not None else "path"] = (
            data if data is not None else str(path)
        )
        stills.append(entry)
    if not stills:
        return
    first = stills[0]
    if first["strength"] is None:
        first["strength"] = 1.0 if strength_flag is None else strength_flag
    elif strength_flag is not None:
        raise ValueError("give the strength inline or with --image-strength, not both")
    for still in stills[1:]:
        if still["strength"] is None:
            still["strength"] = 1.0
    if (
        len(stills) == 1
        and first["frame"] == 0
        and first["crf"] is None
        and "image" in first
    ):
        body["image"], body["image_strength"] = first["image"], first["strength"]
        return
    body["images"] = stills


def _add_guidance(body: dict[str, Any], args: Any) -> None:
    """Upstream's guider flags -> `steps`, `guidance`, `lora_strengths`."""
    if getattr(args, "num_inference_steps", None) is not None:
        body["steps"] = args.num_inference_steps
    guidance: dict[str, dict[str, Any]] = {}
    for modality, other in (("video", "a2v"), ("audio", "v2a")):
        fields = {
            "cfg": getattr(args, f"{modality}_cfg_guidance_scale", None),
            "stg": getattr(args, f"{modality}_stg_guidance_scale", None),
            "stg_blocks": getattr(args, f"{modality}_stg_blocks", None),
            "rescale": getattr(args, f"{modality}_rescale_scale", None),
            "modality": getattr(args, f"{other}_guidance_scale", None),
            "skip_step": getattr(args, f"{modality}_skip_step", None),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if fields:
            guidance[modality] = fields
    if guidance:
        body["guidance"] = guidance
    s1 = getattr(args, "distilled_lora_strength_stage_1", None)
    s2 = getattr(args, "distilled_lora_strength_stage_2", None)
    if s1 is not None or s2 is not None:
        from slimserve.video.ltx25 import pipeline as pl

        body["lora_strengths"] = [
            pl.HQ_LORA_STAGE_1 if s1 is None else s1,
            pl.HQ_LORA_STAGE_2 if s2 is None else s2,
        ]


def _add_loras(body: dict[str, Any], args: Any) -> None:
    """--lora PATH [STRENGTH], repeatable (upstream's form)."""
    groups = getattr(args, "lora", None) or []
    loras = []
    for group in groups:
        if not 1 <= len(group) <= 2:
            raise ValueError("--lora takes PATH [STRENGTH]")
        try:
            strength = float(group[1]) if len(group) > 1 else 1.0
        except ValueError as error:
            raise ValueError(f"--lora {group[0]}: {error}") from error
        loras.append({"path": str(Path(group[0]).expanduser()), "strength": strength})
    if loras:
        body["loras"] = loras


def _one_clip(cfg: dict[str, Any], args: Any) -> int:
    from slimserve.video import server
    from slimserve.video.ltx25.pipeline import LTX25Engine

    body = {"prompt": args.prompt} if cfg["pipeline"] != "hdr_ic_lora" else {}
    if getattr(args, "frame_rate", None) is not None:
        body["fps"] = args.frame_rate
    for key in (
        "size",
        "seconds",
        "seed",
        "negative_prompt",
        "decoder",
        "temporal_upscalings",
        "spatial_upscalings",
        "video_path",
        "start_time",
        "end_time",
        "tile_height",
        "tile_width",
        "audio_path",
        "audio_start_time",
        "audio_max_duration",
        "reference_video",
        "reference_strength",
        "chunk_pixel_frames",
        "chunk_carry_frames",
        "chunk_blend_frames",
        "hdr",
        "input_colorspace",
        "exr_colorspace",
        "text_embeddings",
        "keyframe_strength",
    ):
        if getattr(args, key, None) is not None:
            body[key] = getattr(args, key)
    for key in (
        "skip_stage_2",
        "stage_2_ic_lora",
        "tile",
        "decode_with_keyframes",
        "chunked",
    ):
        if getattr(args, key, False):
            body[key] = True
    if getattr(args, "high_quality", False):
        body["high_quality"] = True
    if getattr(args, "no_keyframes", False):
        body["keyframes"] = False
    if getattr(args, "num_generated_keyframes", None):
        body["generated_keyframes"] = args.num_generated_keyframes
    if getattr(args, "video_conditioning", None):
        body["video_conditioning"] = []
        for group in args.video_conditioning:
            if not 1 <= len(group) <= 2:
                term.fail("--video-conditioning takes PATH [STRENGTH]")
                return 2
            body["video_conditioning"].append(
                {
                    "path": str(Path(group[0]).expanduser()),
                    "strength": float(group[1]) if len(group) > 1 else 1.0,
                }
            )
    if getattr(args, "conditioning_attention_mask", None):
        group = args.conditioning_attention_mask
        if len(group) != 2:
            term.fail("--conditioning-attention-mask takes MASK_PATH STRENGTH")
            return 2
        body["attention_mask"] = str(Path(group[0]).expanduser())
        body["attention_strength"] = float(group[1])
    if getattr(args, "enhance_prompt", False):
        body["enhance_prompt"] = True
    try:
        _add_images(body, args)
        _add_guidance(body, args)
        _add_loras(body, args)
        params = server.normalize_request(body, cfg)
    except ValueError as error:
        term.fail(str(error))
        return 2
    ext = ".wav" if cfg["pipeline"] == "t2a" else ".mp4"
    out = Path(args.output or f"{cfg['pipeline']}-{params['seed']}{ext}").expanduser()
    started = time.perf_counter()
    engine = LTX25Engine(
        root=cfg["root"],
        variant="dev" if cfg["pipeline"] in server.GUIDED_PIPELINES else "distilled",
    )
    source = params.pop("source", None)
    prompt = params.pop("prompt", None)
    extra = ""
    if params.pop("enhance_prompt", False):
        t0 = time.perf_counter()
        prompt = engine.enhance(prompt, server._first_still(params))
        extra = f"enhance {time.perf_counter() - t0:.1f}s, "
        term.note(f"enhanced prompt: {prompt}")
        engine.unload_enhancer()
    decoder = params.pop("decoder", None)
    if params.get("chunk") is not None:
        params["decoder"] = decoder
    if source:
        term.note(
            f"source {source['width']}x{source['height']}x{source['num_frames']} "
            f"at {source['fps']:g} fps"
        )
    fast = server.fast_settings(cfg)
    if fast is not None:
        params["fast"] = fast
    loras = (
        None
        if cfg["pipeline"] in ("ic_lora", "dubit", "alpha", "hdr_ic_lora")
        else params.pop("loras", None)
    )
    with engine.user_loras(loras):
        call = getattr(engine, cfg["pipeline"])
        note = lambda stage, i, s: term.note(f"{stage} step {i + 1}")  # noqa: E731
        if prompt is None:  # the HDR IC-LoRA takes no prompt
            result = call(on_step=note, **params)
        else:
            result = call(prompt, keep_text=False, on_step=note, **params)
    out = engine.render(
        result, out, seed=params["seed"], decoder=decoder, hdr=params.get("hdr")
    )
    spans = ", ".join(f"{k} {v:.1f}s" for k, v in result.timings.spans.items())
    auto = (
        f"  (duration head: {result.predicted_seconds:.1f} s)"
        if result.predicted_seconds is not None
        else ""
    )
    term.ok(
        f"{out}  {result.width}x{result.height}x{result.num_frames}{auto}  "
        f"{time.perf_counter() - started:.1f}s  ({extra}{spans})"
    )
    return 0
