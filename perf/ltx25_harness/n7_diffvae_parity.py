"""Diffusion VAE decoder parity: engine (official file, Metal neighborhood attention) vs the baseline
runner's port, same latent and seed, with per-stage taps. One process per mode.
  ref  OUT.npz    runner decoder in fp32 (weights upcast), taps + pixels
  ours OUT.npz OPERAND   engine; prints rel-L2 per tap and pixel PSNR against OUT_REF.npz if present
Usage: PYTHONPATH=<worktree> python n7_diffvae_parity.py ref|ours ..."""
import os, sys, time, numpy as np, mlx.core as mx
from paths import BF16_PACK, SCRATCH
LAT = f"{SCRATCH}/gate/out/ref_bf16.video.npy"   # (1,128,7,16,24) 768x512x49
lat = mx.array(np.load(LAT)).astype(mx.float32)
mode = sys.argv[1]; out = sys.argv[2]
taps = {}
def tap(name, x): mx.eval(x); taps[name] = np.array(x.astype(mx.float32))
if mode == "ref":
    from mlx.utils import tree_map
    from ltx_core_mlx.model.video_vae.diffusion_decoder.decoder import load_diffusion_decoder
    dec = load_diffusion_decoder(f"{BF16_PACK}/vae_decoder_av.safetensors")
    dec.update(tree_map(lambda p: p.astype(mx.float32), dec.parameters())); mx.eval(dec.parameters())
    mx.synchronize(); t = time.perf_counter(); px = dec.decode(lat, seed=42, tap=tap); mx.eval(px); mx.synchronize()
    print(f"[diffvae ref] runner fp32 decode {time.perf_counter()-t:.1f} s peak {mx.get_peak_memory()/2**30:.1f} GiB shape {px.shape}", flush=True)
    np.savez(out, pixels=np.array(px.astype(mx.float32)), **taps)
else:
    from slimserve.video.ltx25.diffvae import DiffusionVAE
    operand = {"fp32": mx.float32, "fp16": mx.float16}[sys.argv[3]]
    stream = {"fp32": mx.float32, "fp16": mx.float16}[sys.argv[4]] if len(sys.argv) > 4 else mx.float32
    if len(sys.argv) > 5:  # force temporal chunking on the small latent
        import slimserve.video.ltx25.diffvae as _d
        _d.CHUNK_TOKENS = int(sys.argv[5])
    dec = DiffusionVAE(operand=operand, stream=stream).load()
    mx.synchronize(); t = time.perf_counter(); px = dec.decode_raw(lat, seed=42, tap=tap); mx.eval(px); mx.synchronize(); dt = time.perf_counter() - t
    mx.synchronize(); t = time.perf_counter(); px2 = dec.decode_raw(lat, seed=42); mx.eval(px2); mx.synchronize(); dt2 = time.perf_counter() - t
    print(f"[diffvae ours {sys.argv[3]}] decode {dt:.1f} s (warm {dt2:.1f} s) peak {mx.get_peak_memory()/2**30:.1f} GiB shape {px.shape}", flush=True)
    np.savez(out, pixels=np.array(px), **taps)
    ref_path = f"{SCRATCH}/n7/diffvae_ref.npz"
    if os.path.exists(ref_path):
        r = np.load(ref_path)
        for k in taps:
            if k in r:
                x, y = taps[k].astype(np.float64), r[k].astype(np.float64)
                print(f"[diffvae] {k:10s} rel-L2 {np.linalg.norm(x-y)/np.linalg.norm(y):.2e} shapes {x.shape == y.shape}", flush=True)
        x, y = np.array(px).astype(np.float64), r["pixels"].astype(np.float64)
        if x.shape == y.shape:
            u = lambda p: np.clip((np.clip(p, -1, 1) + 1) * 127.5, 0, 255).round()
            mse = np.mean((u(x) - u(y)) ** 2)
            print(f"[diffvae] pixels rel-L2 {np.linalg.norm(x-y)/np.linalg.norm(y):.2e} uint8 PSNR {10*np.log10(255**2/max(mse,1e-12)):.2f} dB max|d| {np.abs(u(x)-u(y)).max():.0f}")
        else:
            print("[diffvae] pixel shape mismatch", x.shape, y.shape)
