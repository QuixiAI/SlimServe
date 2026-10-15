"""N8: the whole keyframe-aware diffusion decode against upstream, on the CPU.

  ref  OUT.npz     upstream DiffusionVideoDecoder (ltx-core, torch CPU, fp32) on a
                   fixed random latent + 2 keyframe planes; records the noise it draws
  ours OUT.npz REF.npz   our DiffusionVAE (fp32 operands and stream, Metal joint NA)
                   decoding the same latent with the same noise
  cmp  REF.npz OURS.npz
Add `--plain` after the mode for the keyframe-less decode of the same latent.

Upstream is imported from $LTX25_UPSTREAM (paths.py; ltx-core + ltx-pipelines
sources on sys.path); run `ref` in the torch env (conda vllm-mlx), `ours` through
gpu_run.py in venv-slimserve. Small on purpose: 3x8x8 latent (17 frames of
256x256), planes at pixel frames 4 and 12."""

import sys

import numpy as np
from paths import MODELS, UP

VAE = f"{MODELS}/vae/ltx-2.5-video-vae-bf16.safetensors"
F, H, W = 3, 8, 8
FRAMES = [4, 12]


def inputs():
    rng = np.random.default_rng(1)
    latent = rng.standard_normal((1, 128, F, H, W)).astype(np.float32)
    planes = rng.standard_normal((1, 128, len(FRAMES), H, W)).astype(np.float32)
    return latent, planes


PLAIN = "--plain" in sys.argv


def ref(out):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src"]
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae.keyframes import DecodeKeyframes
    from ltx_core.model.video_vae.model_configurator import (
        VideoDecoderConfigurator,
        video_decoder_sd_ops_for_checkpoint,
    )
    from ltx_core.model.video_vae.transformer.apply import build_diffvae_mode_op
    from ltx_core.model.video_vae.transformer.config import DiffVAEMode

    latent, planes = inputs()
    drawn = []
    real_randn = torch.randn

    def randn(*args, **kw):
        n = real_randn(*args, **kw)
        drawn.append(n.detach().cpu().float().numpy())
        return n

    torch.randn = randn
    builder = SingleGPUModelBuilder(
        model_class_configurator=VideoDecoderConfigurator,
        model_path=VAE,
        model_sd_ops=video_decoder_sd_ops_for_checkpoint(VAE, diffusion_vae=True),
        module_ops=(build_diffvae_mode_op(DiffVAEMode.CHUNKED_EAGER, None),),
    )
    dec = builder.build(device=torch.device("cpu"), dtype=torch.float32).eval()
    kf = None
    if not PLAIN:
        kf = DecodeKeyframes(
            latents=torch.from_numpy(planes),
            pixel_frame_indices=torch.tensor(FRAMES, dtype=torch.int64),
        )
    gen = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        chunks = list(
            dec.decode_video(torch.from_numpy(latent), generator=gen, keyframes=kf)
        )
    pixels = torch.cat(chunks, dim=0).numpy()  # (f, h, w, c) in [0, 1]
    print("noise draws:", [d.shape for d in drawn], "pixels", pixels.shape)
    np.savez(out, pixels=pixels, **{f"noise{i}": d for i, d in enumerate(drawn)})


def ours(out, ref_path):
    import mlx.core as mx

    from slimserve.video.ltx25 import diffvae

    r = np.load(ref_path)
    count = len([k for k in r if k.startswith("noise")])
    noise = [r[f"noise{i}"] for i in range(count)]
    real_normal = mx.random.normal

    def normal(shape, *args, **kw):
        n = noise.pop(0)
        if tuple(n.shape) != tuple(shape):
            raise RuntimeError(f"noise order: upstream {n.shape}, ours {shape}")
        return mx.array(n)

    mx.random.normal = normal
    latent, planes = inputs()
    vae = diffvae.DiffusionVAE(operand=mx.float32, stream=mx.float32).load()
    kf = None if PLAIN else (mx.array(planes), FRAMES)
    px = vae.decode_raw(mx.array(latent), seed=0, keyframes=kf)
    px = np.array(px)[0].transpose(1, 2, 3, 0)  # (f, h, w, c) in [-1, 1]
    px = np.clip((px + 1) * 0.5, 0, 1)
    mx.random.normal = real_normal
    np.savez(out, pixels=px)


def cmp(a, b):
    x, y = np.load(a)["pixels"], np.load(b)["pixels"]
    print("shapes", x.shape, y.shape)
    err = x - y
    mse = float(np.mean(err**2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"PSNR {psnr:.1f} dB, max-abs {np.abs(err).max():.4f}")
    per = [
        10 * np.log10(1.0 / max(float(np.mean(err[i] ** 2)), 1e-12))
        for i in range(len(x))
    ]
    print("per frame dB:", [f"{p:.0f}" for p in per])


if __name__ == "__main__":
    args = [a for a in sys.argv[2:] if a != "--plain"]
    {"ref": ref, "ours": ours, "cmp": cmp}[sys.argv[1]](*args)
