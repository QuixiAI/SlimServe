"""N8: one DiT forward against upstream's LTXModel, on the CPU.

  ref  OUT.npz         upstream LTXModel (ltx-core, torch CPU, fp32) from the official
                       dev transformer on fixed random tokens/context at sigma 0.7
  ours OUT.npz         our LTX25DiT (fp16 operands, fp32 stream) on the same inputs
  cmp  REF.npz OURS.npz
  guided-ref OUT.npz / guided-ours OUT.npz REF.npz [fp32]: one dev guided step
       (CFG 3/7, STG 1 on block 28, modality 3, rescale 0.7) through upstream's
       BatchedPerturbationConfig + MultiModalGuider vs our GuidedDenoiser; compares x0.
  skip-ref OUT.npz / skip-ours OUT.npz REF.npz [fp32]: the skip_step forward: one
       modality disabled (upstream Modality.enabled False) -> its stream is left
       as its input, the other still cross-attends to it. Two cases: audio off
       (video output compared), video off (audio output compared).

Both sides see identical latent tokens (upstream's own patchify order), identical
random text context (the text path is audited separately), the first latent
frame marked as a keyframe, and no perturbations. Compares the velocities.
Small: 3x8x12 latent (17 frames of 256x384), 18 audio tokens. Run both halves
through gpu_run.py --need-gb 110: the fp32 torch copy of the 22B model is 84 GiB.
Usage: python n8_dit_parity.py <mode> ..."""

import sys

import numpy as np
from paths import MODELS, UP

DIT = f"{MODELS}/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
F, H, W, FPS = 3, 8, 12, 24.0
FRAMES = (F - 1) * 8 + 1
SIGMA = 0.7


def inputs():
    rng = np.random.default_rng(2)
    video = rng.standard_normal((1, 128, F, H, W)).astype(np.float32)
    audio_t = round(FRAMES / FPS * 25)
    audio = rng.standard_normal((1, 8, audio_t, 16)).astype(np.float32)
    vtext = (rng.standard_normal((1, 1024, 4096)) * 0.5).astype(np.float32)
    atext = (rng.standard_normal((1, 1024, 2048)) * 0.5).astype(np.float32)
    return video, audio, vtext, atext


def negative_inputs():
    rng = np.random.default_rng(3)
    vneg = (rng.standard_normal((1, 1024, 4096)) * 0.5).astype(np.float32)
    aneg = (rng.standard_normal((1, 1024, 2048)) * 0.5).astype(np.float32)
    return vneg, aneg


GUIDANCE = dict(
    video=(3.0, 1.0, 3.0, 0.7), audio=(7.0, 1.0, 3.0, 0.7)
)  # cfg, stg, mod, rescale
STG_BLOCKS = [28]


def _upstream_states():
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape

    video, audio, _, _ = inputs()
    vshape = VideoLatentShape(1, 128, F, H, W)
    vtools = VideoLatentTools(
        patchifier=VideoLatentPatchifier(1), target_shape=vshape, fps=FPS
    )
    vstate = vtools.create_initial_state(
        torch.device("cpu"), torch.float32, initial_latent=torch.from_numpy(video)
    )
    ashape = AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(batch=1, frames=FRAMES, height=H * 32, width=W * 32, fps=FPS)
    )
    assert ashape.frames == audio.shape[2], (ashape, audio.shape)
    atools = AudioLatentTools(patchifier=AudioPatchifier(1), target_shape=ashape)
    astate = atools.create_initial_state(
        torch.device("cpu"), torch.float32, initial_latent=torch.from_numpy(audio)
    )
    return vstate, astate


def _upstream_model(dtype=None):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer.model_configurator import (
        LTXV_MODEL_COMFY_RENAMING_MAP,
        LTXModelConfigurator,
    )

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path=DIT,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    return builder.build(
        device=torch.device("cpu"), dtype=dtype or torch.float32
    ).eval()


def guided_ref(out):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )
    from ltx_core.model.transformer.modality import Modality
    from ltx_core.utils import to_denoised

    _, _, vtext, atext = inputs()
    vneg, aneg = negative_inputs()
    vstate, astate = _upstream_states()
    # passes: cond, uncond, ptb (STG on both modalities), mod (no AV cross-attn)
    ptb = [
        PerturbationConfig.empty(),
        PerturbationConfig.empty(),
        PerturbationConfig(
            [
                Perturbation(
                    type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=STG_BLOCKS
                ),
                Perturbation(
                    type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=STG_BLOCKS
                ),
            ]
        ),
        PerturbationConfig(
            [
                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
            ]
        ),
    ]
    n = 4
    vctx = torch.cat([torch.from_numpy(a) for a in (vtext, vneg, vtext, vtext)])
    actx = torch.cat([torch.from_numpy(a) for a in (atext, aneg, atext, atext)])
    sigma = torch.full((n,), SIGMA)

    def rep(t):
        return None if t is None else t.repeat(n, *([1] * (t.dim() - 1)))

    def mod(state, ctx):
        return Modality(
            latent=rep(state.latent),
            sigma=sigma,
            timesteps=rep(state.denoise_mask * SIGMA),
            positions=rep(state.positions),
            context=ctx,
            context_mask=None,
            attention_mask=None,
            keyframes_mask=rep(state.keyframes_mask),
        )

    vmod, amod = mod(vstate, vctx), mod(astate, actx)
    model = _upstream_model()
    perturbations = BatchedPerturbationConfig(
        ptb,
        num_blocks=model.num_blocks,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        vx, ax = model(vmod, amod, perturbations)
        v0 = to_denoised(vmod.latent, vx, vmod.timesteps)
        a0 = to_denoised(amod.latent, ax, amod.timesteps)

    def guide(x0, key):
        cfg, stg, modality, rescale = GUIDANCE[key]
        g = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg,
                stg_scale=stg,
                rescale_scale=rescale,
                modality_scale=modality,
                skip_step=0,
                stg_blocks=STG_BLOCKS,
            )
        )
        c, u, p, m = x0.chunk(n)
        return g.calculate(c, u, p, m)

    np.savez(
        out,
        video=guide(v0, "video").numpy(),
        audio=guide(a0, "audio").numpy(),
        video_tokens=vstate.latent.numpy(),
        audio_tokens=astate.latent.numpy(),
    )
    print("saved guided x0")


def guided_ours(out, ref_path, operand="fp16"):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, pipeline, sampling
    from slimserve.video.ltx25 import dit as dit_mod
    from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT

    if operand == "fp32":
        dit_mod.F = mx.float32
        checkpoints.cast_operands.__defaults__ = (
            mx.float32,
        ) + checkpoints.cast_operands.__defaults__[1:]
    _, _, vtext, atext = inputs()
    vneg, aneg = negative_inputs()
    r = np.load(ref_path)
    weights, _, tcfg = checkpoints.load_dit("dev")
    dit = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
    n_audio = r["audio_tokens"].shape[1]
    # upstream's own token order for both modalities (sigma 0: tokens as given)
    vstate = sampling.noised_state(
        (1, F * H * W, 128),
        sampling.video_positions(F, H, W, FPS),
        0,
        sigma=0.0,
        initial=mx.array(r["video_tokens"]),
        tokens_per_frame=H * W,
    )
    astate = sampling.noised_state(
        (1, n_audio, 128),
        sampling.audio_positions(n_audio),
        0,
        sigma=0.0,
        initial=mx.array(r["audio_tokens"]),
    )
    vg = sampling.Guidance(
        *GUIDANCE["video"][:3], GUIDANCE["video"][3], tuple(STG_BLOCKS)
    )
    ag = sampling.Guidance(
        *GUIDANCE["audio"][:3], GUIDANCE["audio"][3], tuple(STG_BLOCKS)
    )
    den = pipeline.GuidedDenoiser(
        dit,
        (mx.array(vtext), mx.array(atext)),
        (mx.array(vneg), mx.array(aneg)),
        vg,
        ag,
    )
    v0, a0 = den(vstate, astate, vstate.latent, astate.latent, SIGMA)
    mx.eval(v0, a0)
    assert r["video"].shape == v0.shape, (r["video"].shape, v0.shape)
    np.savez(out, video=np.array(v0), audio=np.array(a0))


def ref(out):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.model.transformer.modality import Modality

    _, _, vtext, atext = inputs()
    vstate, astate = _upstream_states()
    sigma = torch.tensor([SIGMA])

    def mod(state, ctx):
        return Modality(
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * SIGMA,
            positions=state.positions,
            context=torch.from_numpy(ctx),
            context_mask=None,
            attention_mask=None,
            keyframes_mask=state.keyframes_mask,
        )

    model = _upstream_model()
    with torch.inference_mode():
        vx, ax = model(mod(vstate, vtext), mod(astate, atext), None)
    np.savez(
        out,
        video=vx.float().numpy(),
        audio=ax.float().numpy(),
        video_tokens=vstate.latent.numpy(),
        audio_tokens=astate.latent.numpy(),
        video_positions=vstate.positions.numpy(),
        audio_positions=astate.positions.numpy(),
        keyframes_mask=vstate.keyframes_mask.numpy(),
    )
    print("saved", vx.shape, ax.shape)


def ours(out, ref_path, operand="fp16"):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, sampling
    from slimserve.video.ltx25 import dit as dit_mod
    from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT

    if operand == "fp32":  # precision control: fp32 GEMM/attention operands
        dit_mod.F = mx.float32
        checkpoints.cast_operands.__defaults__ = (
            mx.float32,
        ) + checkpoints.cast_operands.__defaults__[1:]

    r = np.load(ref_path)
    _, _, vtext, atext = inputs()
    vt = mx.array(r["video_tokens"])
    at = mx.array(r["audio_tokens"])
    # our patchify must agree with upstream's token order
    video, audio, _, _ = inputs()
    assert np.allclose(np.array(sampling.patchify(mx.array(video))), r["video_tokens"])
    n_audio = at.shape[1]
    weights, _, tcfg = checkpoints.load_dit("dev")
    model = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
    kf = r["keyframes_mask"]
    v, a = model(
        vt,
        at,
        mx.array([SIGMA]),
        mx.array(vtext),
        mx.array(atext),
        sampling.video_positions(F, H, W, FPS),
        sampling.audio_positions(n_audio),
        video_keyframes_mask=mx.array(kf),
    )
    mx.eval(v, a)
    np.savez(out, video=np.array(v), audio=np.array(a))


def skip_ref(out, dtype="fp32"):
    """`dtype` bf16: upstream's own serving precision, to size its error."""
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.model.transformer.modality import Modality

    _, _, vtext, atext = inputs()
    vstate, astate = _upstream_states()
    sigma = torch.tensor([SIGMA])

    def mod(state, ctx, enabled):
        return Modality(
            latent=state.latent,
            sigma=sigma,
            timesteps=state.denoise_mask * SIGMA,
            positions=state.positions,
            context=torch.from_numpy(ctx),
            context_mask=None,
            attention_mask=None,
            keyframes_mask=state.keyframes_mask,
            enabled=enabled,
        )

    model = _upstream_model(torch.bfloat16 if dtype == "bf16" else None)
    if dtype == "bf16":
        cast = lambda m: Modality(  # noqa: E731
            **{
                k: (v.to(torch.bfloat16) if k in ("latent", "context") else v)
                for k, v in m.__dict__.items()
            }
        )
        mod_ = mod
        mod = lambda s, c, e: cast(mod_(s, c, e))  # noqa: E731
    with torch.inference_mode():
        vx, _ = model(mod(vstate, vtext, True), mod(astate, atext, False), None)
        _, ax = model(mod(vstate, vtext, False), mod(astate, atext, True), None)
    np.savez(
        out,
        video=vx.float().numpy(),  # audio disabled
        audio=ax.float().numpy(),  # video disabled
        video_tokens=vstate.latent.numpy(),
        audio_tokens=astate.latent.numpy(),
        keyframes_mask=vstate.keyframes_mask.numpy(),
    )
    print("saved", vx.shape, ax.shape)


def skip_ours(out, ref_path, operand="fp16"):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, sampling
    from slimserve.video.ltx25 import dit as dit_mod
    from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT

    if operand == "fp32":
        dit_mod.F = mx.float32
        checkpoints.cast_operands.__defaults__ = (
            mx.float32,
        ) + checkpoints.cast_operands.__defaults__[1:]
    r = np.load(ref_path)
    _, _, vtext, atext = inputs()
    vt, at = mx.array(r["video_tokens"]), mx.array(r["audio_tokens"])
    weights, _, tcfg = checkpoints.load_dit("dev")
    model = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
    common = dict(
        video_keyframes_mask=mx.array(r["keyframes_mask"]),
    )
    args = (
        vt,
        at,
        mx.array([SIGMA]),
        mx.array(vtext),
        mx.array(atext),
        sampling.video_positions(F, H, W, FPS),
        sampling.audio_positions(at.shape[1]),
    )
    v, a_none = model(*args, run_audio=False, **common)
    assert a_none is None
    v_none, a = model(*args, run_video=False, **common)
    assert v_none is None
    mx.eval(v, a)
    np.savez(out, video=np.array(v), audio=np.array(a))


def t2a_ref(out):
    """The audio-only forward (video=None): upstream LTXModel vs ours."""
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.model.transformer.modality import Modality

    _, _, _, atext = inputs()
    _, astate = _upstream_states()
    sigma = torch.tensor([SIGMA])
    audio = Modality(
        latent=astate.latent,
        sigma=sigma,
        timesteps=astate.denoise_mask * SIGMA,
        positions=astate.positions,
        context=torch.from_numpy(atext),
        context_mask=None,
        attention_mask=None,
        keyframes_mask=None,
    )
    model = _upstream_model()
    with torch.inference_mode():
        _, ax = model(None, audio, None)
    np.savez(out, audio=ax.float().numpy(), audio_tokens=astate.latent.numpy())
    print("saved", ax.shape)


def t2a_ours(out, ref_path, operand="fp16"):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, sampling
    from slimserve.video.ltx25 import dit as dit_mod
    from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT

    if operand == "fp32":
        dit_mod.F = mx.float32
        checkpoints.cast_operands.__defaults__ = (
            mx.float32,
        ) + checkpoints.cast_operands.__defaults__[1:]
    r = np.load(ref_path)
    _, _, _, atext = inputs()
    at = mx.array(r["audio_tokens"])
    weights, _, tcfg = checkpoints.load_dit("dev")
    model = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
    v, a = model(
        None,
        at,
        mx.array([SIGMA]),
        None,
        mx.array(atext),
        None,
        sampling.audio_positions(at.shape[1]),
    )
    assert v is None
    mx.eval(a)
    np.savez(out, audio=np.array(a), video=np.zeros(1))


def v2v_ref(out):
    """The video-only forward (audio=None): upstream LTXModel vs ours."""
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.model.transformer.modality import Modality

    _, _, vtext, _ = inputs()
    vstate, _ = _upstream_states()
    sigma = torch.tensor([SIGMA])
    video = Modality(
        latent=vstate.latent,
        sigma=sigma,
        timesteps=vstate.denoise_mask * SIGMA,
        positions=vstate.positions,
        context=torch.from_numpy(vtext),
        context_mask=None,
        attention_mask=None,
        keyframes_mask=vstate.keyframes_mask,
    )
    model = _upstream_model()
    with torch.inference_mode():
        vx, _ = model(video, None, None)
    np.savez(
        out,
        video=vx.float().numpy(),
        video_tokens=vstate.latent.numpy(),
        keyframes_mask=vstate.keyframes_mask.numpy(),
    )
    print("saved", vx.shape)


def v2v_ours(out, ref_path, operand="fp16"):
    import mlx.core as mx

    from slimserve.video.ltx25 import checkpoints, sampling
    from slimserve.video.ltx25 import dit as dit_mod
    from slimserve.video.ltx25.dit import DiTConfig, LTX25DiT

    if operand == "fp32":
        dit_mod.F = mx.float32
        checkpoints.cast_operands.__defaults__ = (
            mx.float32,
        ) + checkpoints.cast_operands.__defaults__[1:]
    r = np.load(ref_path)
    _, _, vtext, _ = inputs()
    vt = mx.array(r["video_tokens"])
    weights, _, tcfg = checkpoints.load_dit("dev")
    model = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
    v, a = model(
        vt,
        None,
        mx.array([SIGMA]),
        mx.array(vtext),
        None,
        sampling.video_positions(F, H, W, FPS),
        None,
        video_keyframes_mask=mx.array(r["keyframes_mask"]),
    )
    assert a is None
    mx.eval(v)
    np.savez(out, video=np.array(v))


def cmp(a, b):
    ra, rb = np.load(a), np.load(b)
    for key in ("video", "audio"):
        if key not in ra or key not in rb:
            continue
        x, y = ra[key].astype(np.float64), rb[key].astype(np.float64)
        rel = np.linalg.norm(x - y) / np.linalg.norm(x)
        cos = float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))
        print(
            f"{key}: rel-L2 {rel:.2e} cos {cos:.6f} max-abs {np.abs(x - y).max():.3e}"
        )


if __name__ == "__main__":
    {
        "ref": ref,
        "ours": ours,
        "skip-ref": skip_ref,
        "skip-ours": skip_ours,
        "t2a-ref": t2a_ref,
        "t2a-ours": t2a_ours,
        "v2v-ref": v2v_ref,
        "v2v-ours": v2v_ours,
        "cmp": cmp,
        "guided-ref": guided_ref,
        "guided-ours": guided_ours,
    }[sys.argv[1]](*sys.argv[2:])
