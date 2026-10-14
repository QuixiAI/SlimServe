# ruff: noqa: E501
"""N11: one DFR temporal round against upstream's run_one_temporal_round (torch, CPU).

  ref  OUT.npz         upstream ltx_pipelines.dfr_stages.run_one_temporal_round on a fixed
                       random stage-2 canvas (1, 128, 7, 8, 8 = 49 frames of 256x256), two
                       keyframe planes (24, 48), random stage-1 audio and text contexts, the
                       distilled transformer in fp32; records every noise draw
  ours OUT.npz REF.npz our LTX25Engine._temporal_round (fp32 operands and stream) on the
                       same inputs with upstream's noise replayed; prints the diffs
  cmp  REF.npz OURS.npz

Round 1 of a 49-frame canvas: 97 frames at 48 fps, two windows (the second starts on the
plane at 48 with a pinned lead-in), two anchors, two new slots. Run `ref` in the torch env
(conda vllm-mlx, ~84 GiB: through gpu_run.py --need-gb 80), `ours` through gpu_run.py
--need-gb 50 in venv-slimserve."""

import sys

import numpy as np
from paths import MODELS as ROOT
from paths import SCRATCH, UP

STUB = f"{SCRATCH}/n11"
DIT = f"{ROOT}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
VAE = f"{ROOT}/vae/ltx-2.5-video-vae-bf16.safetensors"
TUP = f"{ROOT}/latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
F, H, W, FPS = 7, 8, 8, 24.0
CANVAS = (F - 1) * 8 + 1  # 49
PLANES = (24, 48)
SEED = 7


REAL = f"{SCRATCH}/n11/tr_real_inputs.npz"


def inputs():
    """Random tensors, or (REAL_INPUTS=1) a real stage-2 canvas, its slot planes,
    stage-1 audio and the beat1d text contexts from our pipeline (`real-inputs`)."""
    import os

    if os.environ.get("REAL_INPUTS"):
        r = np.load(REAL)
        planes = {int(p): r[f"plane{p}"] for p in PLANES}
        return r["latent"], planes, r["audio"], r["vtext"], r["atext"]
    rng = np.random.default_rng(11)
    latent = rng.standard_normal((1, 128, F, H, W)).astype(np.float32)
    planes = {
        p: rng.standard_normal((1, 128, 1, H, W)).astype(np.float32) for p in PLANES
    }
    n_audio = round(CANVAS / FPS * 25)
    audio = rng.standard_normal((1, 8, n_audio, 16)).astype(np.float32)
    vtext = (rng.standard_normal((1, 1024, 4096)) * 0.5).astype(np.float32)
    atext = (rng.standard_normal((1, 1024, 2048)) * 0.5).astype(np.float32)
    return latent, planes, audio, vtext, atext


def real_inputs():
    """Run our DFR stages 1-2 at 256x256x49 (fp16 engine, seed 7, the fox
    prompt) and save the round's inputs for both sides."""
    import mlx.core as mx

    from slimserve.video.ltx25 import pipeline

    eng = pipeline.LTX25Engine(variant="distilled")
    prompt = "A red fox trotting through a snowy pine forest at dawn"
    vtext, atext = eng.load_text().encode(prompt)[:2]
    mx.eval(vtext, atext)
    res = eng.dfr(
        prompt,
        height=H * 32,
        width=W * 32,
        num_frames=CANVAS,
        seed=SEED,
        text_embeds=(vtext, atext),
    )
    # the untrimmed canvas is the latent (49 frames is a whole canvas) and the
    # planes are the keyframes
    planes, frames = res.keyframes
    assert list(frames) == list(PLANES), frames
    a = np.array(res.audio_tokens)  # (1, T, 128) = "b t (c f)"
    n_audio = a.shape[1]
    audio = a.reshape(1, n_audio, 8, 16).transpose(0, 2, 1, 3)
    np.savez(
        REAL,
        latent=np.array(res.video_latent),
        audio=np.ascontiguousarray(audio),
        vtext=np.array(vtext),
        atext=np.array(atext),
        **{
            f"plane{p}": np.array(planes[:, :, i : i + 1]) for i, p in enumerate(frames)
        },
    )
    print("saved", REAL, np.array(res.video_latent).shape, audio.shape)


class _Stop(Exception):
    pass


def ref(out, step0=False):
    """step0: capture tile 0's first denoiser call (state + x0) and stop."""
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src", f"{UP}/ltx-pipelines/src", STUB]
    import oiio_stub  # noqa: F401
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.tools import VideoLatentTools
    from ltx_core.types import VideoLatentShape, VideoPixelShape
    from ltx_pipelines.dfr_stages import run_one_temporal_round
    from ltx_pipelines.utils.blocks import (
        DiffusionStage,
        ImageConditioner,
        VideoUpsampler,
    )
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS

    latent, planes, audio, vtext, atext = inputs()
    drawn = []
    real_randn = torch.randn

    def randn(*args, **kw):
        n = real_randn(*args, **kw)
        drawn.append(n.detach().cpu().float().numpy())
        return n

    torch.randn = randn
    device, dtype = torch.device("cpu"), torch.float32
    stage = DiffusionStage.from_checkpoint(DIT, dtype, device)
    # The stage builds the transformer in the checkpoint's dtype (bf16) and
    # disposes it after every window. The reference wants fp32: build it once
    # with ltx_core's builder (as n8 does) and hand that out, dispose a no-op.
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model import X0Model
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    fp32 = (
        SingleGPUModelBuilder(
            model_class_configurator=LTXModelConfigurator,
            model_path=DIT,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        .build(device=device, dtype=torch.float32)
        .eval()
    )

    class _Keep(X0Model):
        def dispose(self):
            pass

    DiffusionStage._build_transformer = lambda self, device=None, **kw: _Keep(
        fp32
    ).eval()
    # the sampler's latent updates default to bf16 (model_dtype); fp32 reference
    import functools

    import ltx_pipelines.dfr_stages as ds
    from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop

    ds.euler_ancestral_denoising_loop = functools.partial(
        euler_ancestral_denoising_loop, model_dtype=torch.float32
    )
    upsampler = VideoUpsampler(VAE, TUP, dtype, device)
    conditioner = ImageConditioner(VAE, dtype, device)
    tools = VideoLatentTools(
        VideoLatentPatchifier(1), VideoLatentShape(1, 128, F, H, W), FPS
    )
    # the stage returns unpatchified states; create_initial_state patchifies
    video_state = tools.unpatchify(
        tools.create_initial_state(
            device, dtype, initial_latent=torch.from_numpy(latent)
        )
    )
    assert tuple(video_state.latent.shape) == (1, 128, F, H, W), (
        video_state.latent.shape
    )
    keyframes = {p: torch.from_numpy(x) for p, x in planes.items()}
    gen = torch.Generator(device=device).manual_seed(SEED)
    if step0:
        from ltx_pipelines.utils.denoisers import SimpleDenoiser

        _call = SimpleDenoiser.__call__
        steps: list = []

        def call(self, transformer, video_state, audio_state, sigmas, step_index):
            res = _call(self, transformer, video_state, audio_state, sigmas, step_index)
            f = lambda t: t.detach().float().cpu().numpy()  # noqa: E731
            steps.append(
                dict(
                    latent=f(video_state.latent),
                    clean=f(video_state.clean_latent),
                    mask=f(video_state.denoise_mask),
                    positions=f(video_state.positions),
                    kf=f(video_state.keyframes_mask),
                    audio=f(audio_state.latent),
                    audio_positions=f(audio_state.positions),
                    x0=f(res.video.denoised),
                    sigma=float(sigmas[step_index]),
                )
            )
            return res

        def save():
            flat = {f"s{i}_{k}": v for i, st in enumerate(steps) for k, v in st.items()}
            flat.update({f"noise{i}": d for i, d in enumerate(drawn)})
            flat["steps"] = len(steps)
            np.savez(out, **flat)

        SimpleDenoiser.__call__ = call
    try:
        with torch.inference_mode():
            state, carry, seams = run_one_temporal_round(
                round_idx=1,
                stage=stage,
                temporal_upsampler=upsampler,
                image_conditioner=conditioner,
                video_state=video_state,
                keyframes=keyframes,
                video_shape=VideoPixelShape(
                    batch=1,
                    frames=2 * (CANVAS - 1) + 1,
                    height=H * 32,
                    width=W * 32,
                    fps=2 * FPS,
                ),
                images=[],
                video_context=torch.from_numpy(vtext),
                audio_context=torch.from_numpy(atext),
                audio_latent=torch.from_numpy(audio),
                source_duration=CANVAS / FPS,
                seed=SEED,
                noiser=GaussianNoiser(generator=gen),
                sigmas=DISTILLED_SIGMAS[4:].to(dtype=torch.float32, device=device),
                device=device,
                dtype=dtype,
                temporal_scale=8,
            )
    except _Stop:
        save()
        print("captured:", out, len(steps), "steps,", len(drawn), "draws")
        return
    if step0:
        save()
        print("captured:", out, len(steps), "steps,", len(drawn), "draws")
        return
    print("noise draws:", [d.shape for d in drawn])
    print("seams", seams, "carry", sorted(carry))
    np.savez(
        out,
        latent=state.latent.float().numpy(),
        plane_positions=np.array(sorted(carry)),
        planes=np.concatenate(
            [carry[p].float().numpy() for p in sorted(carry)], axis=2
        ),
        **{f"noise{i}": d for i, d in enumerate(drawn)},
    )


class _Replay:
    """Serve upstream's recorded noise draws to our code, splitting a draw over
    several requests when ours builds the same token sequence in pieces
    (upstream noises base + anchor + slot tokens with one draw)."""

    def __init__(self, draws):
        self.queue = [d for d in draws]
        self.partial = None

    def __call__(self, shape, *args, **kw):
        import mlx.core as mx

        if self.partial is None:
            self.partial = self.queue.pop(0)
        cur = self.partial
        if tuple(cur.shape) == tuple(shape):
            self.partial = None
            return mx.array(cur)
        if (
            cur.ndim == 3
            and len(shape) == 3
            and cur.shape[0] == shape[0]
            and cur.shape[2] == shape[2]
        ):
            n = shape[1]
            if n > cur.shape[1]:
                # upstream noised the frozen audio too (scale 0); ours never draws it
                self.partial = None
                return self(shape, *args, **kw)
            head, rest = cur[:, :n], cur[:, n:]
            self.partial = rest if rest.shape[1] else None
            return mx.array(np.ascontiguousarray(head))
        if (
            cur.ndim == 4 and len(shape) == 3
        ):  # upstream audio (1, 8, T, 16) -> our (1, T, 128)
            flat = cur.transpose(0, 2, 1, 3).reshape(1, cur.shape[2], -1)
            self.partial = None
            if tuple(flat.shape) != tuple(shape):
                raise RuntimeError(f"noise replay: audio {flat.shape} vs ours {shape}")
            return mx.array(np.ascontiguousarray(flat))
        raise RuntimeError(
            f"noise replay: ours wants {shape}, upstream drew {cur.shape}"
        )


def ours(out, ref_path):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, pipeline
    from slimserve.video.ltx25 import dit as dit_mod

    dit_mod.F = mx.float32
    checkpoints.cast_operands.__defaults__ = (
        mx.float32,
    ) + checkpoints.cast_operands.__defaults__[1:]
    latent, planes, audio, vtext, atext = inputs()
    r = np.load(ref_path)
    count = len([k for k in r if k.startswith("noise")])
    replay = _Replay([r[f"noise{i}"] for i in range(count)])
    real_normal = mx.random.normal
    mx.random.normal = replay
    eng = pipeline.LTX25Engine(variant="distilled")
    dit = eng.load_dit()
    vae = eng.load_vae()
    denoise = pipeline.Denoiser(dit, mx.array(vtext), mx.array(atext))
    # upstream's audio token order: (B, 8, T, 16) -> (B, T, 128)
    audio_tokens = mx.array(audio.transpose(0, 2, 1, 3).reshape(1, audio.shape[2], -1))
    tm = pipeline.Timings()
    stitched, carry = eng._temporal_round(
        1,
        denoise,
        vae,
        mx.array(latent),
        {p: mx.array(x) for p, x in planes.items()},
        2 * (CANVAS - 1) + 1,
        2 * FPS,
        H,
        W,
        audio_tokens,
        CANVAS / FPS,
        SEED,
        None,
        1.0,
        tm,
        None,
    )
    mx.random.normal = real_normal
    mx.eval(stitched)
    if replay.queue or replay.partial is not None:
        print(
            "WARNING: upstream drew more noise than ours consumed:",
            len(replay.queue),
            "left",
        )
    np.savez(
        out,
        latent=np.array(stitched),
        plane_positions=np.array(sorted(carry)),
        planes=np.concatenate([np.array(carry[p]) for p in sorted(carry)], axis=2),
    )
    cmp(ref_path, out)


def ours_step0(out, ref_path):
    """Our tile-0 state and first x0 against upstream's capture (`ref ... step0`)."""
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, pipeline, sampling
    from slimserve.video.ltx25 import dit as dit_mod

    dit_mod.F = mx.float32
    checkpoints.cast_operands.__defaults__ = (
        mx.float32,
    ) + checkpoints.cast_operands.__defaults__[1:]
    latent, planes, audio, vtext, atext = inputs()
    r = np.load(ref_path)
    count = len([k for k in r if k.startswith("noise")])
    replay = _Replay([r[f"noise{i}"] for i in range(count)])
    mx.random.normal = replay
    eng = pipeline.LTX25Engine(variant="distilled")
    dit = eng.load_dit()
    vae = eng.load_vae()
    audio_tokens = mx.array(audio.transpose(0, 2, 1, 3).reshape(1, audio.shape[2], -1))

    steps: list = []
    n_ref = int(r["steps"])
    per_tile = 4  # steps per tile (DISTILLED_SIGMAS[4:])

    class Probe(pipeline.Denoiser):
        def __call__(self, video, audio, vx, ax, sigma, step_cache=None):
            v0, a0 = super().__call__(video, audio, vx, ax, sigma)
            mx.eval(v0, a0)
            steps.append(
                dict(
                    latent=np.array(vx),
                    clean=np.array(video.clean),
                    mask=np.array(video.denoise_mask),
                    positions=np.array(video.positions),
                    kf=np.array(video.keyframes_mask),
                    audio=np.array(ax),
                    audio_positions=np.array(audio.positions),
                    x0=np.array(v0),
                    sigma=sigma,
                )
            )
            if len(steps) == n_ref:
                raise _Stop()
            return v0, a0

    # tile 1 must start from upstream's tile-0 result, not ours: hand the loop
    # upstream's final tile-0 tokens (its last x0 is the terminal output) when
    # the first tile finishes, so the prefix / lead-in / slot-plane logic is
    # compared on identical inputs
    _loop = sampling.euler_ancestral_loop

    def loop(denoise, video, audio, sigmas, noise_seed, **kw):
        vt, at = _loop(denoise, video, audio, sigmas, noise_seed, **kw)
        if n_ref > per_tile and len(steps) == per_tile:
            ref_last = r[f"s{per_tile - 1}_x0"]  # tile 0's terminal x0 (blended)
            mask = r[f"s{per_tile - 1}_mask"]
            clean = r[f"s{per_tile - 1}_clean"]
            vt = mx.array(ref_last * mask + clean * (1.0 - mask))
        return vt, at

    sampling.euler_ancestral_loop = loop

    tm = pipeline.Timings()
    import contextlib

    with contextlib.suppress(_Stop):
        eng._temporal_round(
            1,
            Probe(dit, mx.array(vtext), mx.array(atext)),
            vae,
            mx.array(latent),
            {p: mx.array(x) for p, x in planes.items()},
            2 * (CANVAS - 1) + 1,
            2 * FPS,
            H,
            W,
            audio_tokens,
            CANVAS / FPS,
            SEED,
            None,
            1.0,
            tm,
            None,
        )
    sampling.euler_ancestral_loop = _loop
    np.savez(
        out, **{f"s{i}_{k}": v for i, st in enumerate(steps) for k, v in st.items()}
    )
    rel = lambda a, b: (
        np.linalg.norm(a.astype(np.float64) - b) / max(np.linalg.norm(b), 1e-12)
    )  # noqa: E731
    for i, st in enumerate(steps):
        print(
            f"step {i} (tile {i // per_tile}) sigma ours {st['sigma']:.4f} ref {float(r[f's{i}_sigma']):.4f}"
        )
        for k in ("latent", "clean", "mask", "kf", "positions", "audio", "x0"):
            a, b = st[k], r[f"s{i}_{k}"]
            if k == "positions":
                b = ((b[:, :, :, 0] + b[:, :, :, 1]) / 2.0).transpose(
                    0, 2, 1
                )  # (B, N, 3) midpoints
            if a.shape != b.shape:
                print(f"  {k}: shape ours {a.shape} ref {b.shape}")
                continue
            n_gen = 448 if i < per_tile else 640
            print(
                f"  {k}: rel {rel(a, b):.2e} max-abs {np.abs(a - b).max():.3e}"
                + (
                    f" (gen {rel(a[:, :n_gen], b[:, :n_gen]):.1e}, cond {rel(a[:, n_gen:], b[:, n_gen:]):.1e})"
                    if a.ndim == 3 and a.shape[1] > n_gen
                    else ""
                )
            )


def cmp(a, b):
    ra, rb = np.load(a), np.load(b)
    print(
        "plane positions",
        ra["plane_positions"].tolist(),
        rb["plane_positions"].tolist(),
    )
    for k in ("latent", "planes"):
        x, y = ra[k].astype(np.float64), rb[k].astype(np.float64)
        print(
            f"{k}: shapes {x.shape} {y.shape} rel-L2 {np.linalg.norm(x - y) / np.linalg.norm(x):.2e} "
            f"max-abs {np.abs(x - y).max():.3e}"
        )
        if k == "latent":
            per = [
                np.linalg.norm(x[:, :, t] - y[:, :, t]) / np.linalg.norm(x[:, :, t])
                for t in range(x.shape[2])
            ]
            print("  per cell:", " ".join(f"{p:.1e}" for p in per))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "ref-step0":
        ref(sys.argv[2], step0=True)
    else:
        {
            "ref": ref,
            "ours": ours,
            "ours-step0": ours_step0,
            "cmp": cmp,
            "real-inputs": real_inputs,
        }[mode](*sys.argv[2:])
