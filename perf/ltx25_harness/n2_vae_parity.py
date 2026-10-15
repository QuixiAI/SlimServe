"""N2 parity: engine video VAE + latent upscalers (official files) vs the baseline runner (converted pack).
Usage: PYTHONPATH=<worktree> <runner venv python> n2_vae_parity.py decode|tiled|encode|upscale|bench121|profile121
Latent: a real final latent of the 768x512x49 distilled clip (gate/out/ref_bf16.video.npy, (1,128,7,16,24)).
Timings are rough when other GPU jobs run. Results: $LTX25_SCRATCH/n2_vae/."""
import os, sys, time, json, numpy as np, mlx.core as mx
from mlx.utils import tree_map
from slimserve.video.ltx25 import vae as V
from slimserve.video.ltx25.vae import VideoVAE, Tiling
from slimserve.video.ltx25.upscaler import LatentUpscaler
from paths import BF16_PACK, SCRATCH
PACK = BF16_PACK
OUT = f"{SCRATCH}/n2_vae"
LAT = mx.array(np.load(f"{SCRATCH}/gate/out/ref_bf16.video.npy").astype(np.float32))
mode = sys.argv[1]

def timed(fn):
    mx.synchronize(); mx.reset_peak_memory(); t = time.perf_counter(); r = fn(); mx.eval(r) if isinstance(r, mx.array) else None
    mx.synchronize(); return r, time.perf_counter() - t, mx.get_peak_memory() / 2**30

def psnr8(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)

def cmp(name, x, ref):
    x, ref = np.array(x.astype(mx.float32)), np.array(ref.astype(mx.float32))
    rel = np.linalg.norm((x - ref).astype(np.float64)) / np.linalg.norm(ref.astype(np.float64))
    line = f"[n2-vae] {name}: rel-L2 {rel:.2e} max|d| {np.abs(x - ref).max():.3e} bit-identical={bool((x == ref).all())}"
    if x.ndim == 5 and x.shape[1] == 3:
        u = lambda p: np.clip((np.clip(p, -1, 1) + 1) * 127.5, 0, 255).astype(np.uint8)
        a, b = u(x), u(ref)
        line += f" | uint8 PSNR {psnr8(a, b):.2f} dB max|d| {int(np.abs(a.astype(int) - b.astype(int)).max())} differing {np.mean(a != b) * 100:.3f}%"
    print(line, flush=True)

def runner_decoder(fp32):
    from ltx_pipelines_mlx.utils.blocks import VideoDecoder
    d = VideoDecoder(PACK, verbose=False).load()
    if fp32: d.update(tree_map(lambda p: p.astype(mx.float32), d.parameters()))
    mx.eval(d.parameters()); return d

def runner_encoder(fp32):
    from ltx_core_mlx.model.video_vae.video_vae import VideoEncoder
    from ltx_core_mlx.utils.weights import load_split_safetensors
    e = VideoEncoder()
    w = load_split_safetensors(f"{PACK}/vae_encoder_conv.safetensors", prefix="vae_encoder_conv.")
    e.load_weights([(k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"), v) for k, v in w.items()])
    if fp32: e.update(tree_map(lambda p: p.astype(mx.float32), e.parameters()))
    mx.eval(e.parameters()); return e

def lat121():  # 16 latent frames = 121 pixel frames, real statistics (the 7-frame latent repeated)
    return mx.concatenate([LAT, LAT, LAT[:, :, :2]], axis=2)

if mode == "decode":
    ours32 = VideoVAE().load(encoder=False)
    p32, t, pk = timed(lambda: ours32.decode_raw(LAT)); print(f"[n2-vae] ours fp32 49f: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    r = runner_decoder(True); ref32, t, pk = timed(lambda: r.decode(LAT)); print(f"[n2-vae] runner fp32 49f: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    cmp("decode ours fp32 vs runner fp32", p32, ref32); del r
    r = runner_decoder(False); refbf, t, pk = timed(lambda: r.decode(LAT)); print(f"[n2-vae] runner bf16 (as shipped) 49f: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    cmp("decode ours fp32 vs runner as shipped (bf16)", p32, refbf); cmp("decode runner bf16 vs runner fp32", refbf, ref32); del r
    ours16 = VideoVAE(dtype=mx.float16).load(encoder=False)
    p16, t, pk = timed(lambda: ours16.decode_raw(LAT)); print(f"[n2-vae] ours fp16 49f: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    cmp("decode ours fp16 vs ours fp32", p16, p32)
    np.save(f"{OUT}/ours_fp32_49f_pixels.npy", np.array(p32))
elif mode == "tiled":
    from ltx_core_mlx.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
    ours = VideoVAE().load(encoder=False); r = runner_decoder(True)
    for sp, tp in (((512, 32), (40, 8)), (None, (40, 8)), ((256, 32), None)):
        a = mx.concatenate(list(ours.decode_chunks(LAT, Tiling(spatial=sp, temporal=tp))), axis=2)
        cfg = TilingConfig(SpatialTilingConfig(*sp) if sp else None, TemporalTilingConfig(*tp) if tp else None)
        b = mx.concatenate(list(r.tiled_decode(LAT, cfg)), axis=2)
        cmp(f"tiled decode spatial={sp} temporal={tp} ({a.shape[2]} frames) ours vs runner fp32", a, b)
    full = ours.decode_raw(LAT); cmp("tiled (512/32, 40/8) vs untiled, ours", mx.concatenate(list(ours.decode_chunks(LAT, Tiling((512, 32), (40, 8)))), axis=2), full)
    for shape in ((1, 128, 16, 16, 24), (1, 128, 16, 32, 48), (1, 128, 31, 34, 60)):
        from ltx_core_mlx.model.video_vae.video_vae import _compute_decode_tiling, describe_decode_tiling
        for gb in (128, 64, 24, 8):
            mine, theirs = V.plan_tiling(shape, 24.0, gb << 30), _compute_decode_tiling(shape, 24.0, gb << 30)
            print(f"[n2-vae] plan {shape} budget {gb} GB: ours {mine.describe() if mine else 'untiled'} | runner {describe_decode_tiling(theirs) if theirs else 'untiled'}", flush=True)
elif mode == "encode":
    px = mx.array(np.load(f"{OUT}/ours_fp32_49f_pixels.npy"))
    ours = VideoVAE().load(); a, t, pk = timed(lambda: ours.encode(px)); print(f"[n2-vae] ours fp32 encode 49f: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    r = runner_encoder(True); b = r.encode(px); cmp("encode ours fp32 vs runner fp32", a, b); del r
    r = runner_encoder(False); c = r.encode(px.astype(mx.bfloat16)); cmp("encode ours fp32 vs runner as shipped (bf16)", a, c)
    cmp("encode round trip vs the latent it was decoded from", a, LAT)
    img = px[:, :, :1]; cmp("encode single image ours vs runner fp32", ours.encode(img), runner_encoder(True).encode(img))
    o16 = VideoVAE(dtype=mx.float16).load(); cmp("encode ours fp16 vs ours fp32", o16.encode(px), a)
    from ltx_core_mlx.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
    r = runner_encoder(True)
    cmp("tiled encode (256/64, 24/16) ours vs runner fp32", ours.encode_tiled(px, Tiling((256, 64), (24, 16))),
        r.tiled_encode(px, TilingConfig(SpatialTilingConfig(256, 64), TemporalTilingConfig(24, 16))))
elif mode == "upscale":
    from ltx_pipelines_mlx.utils.blocks import VideoUpsampler
    vae = VideoVAE().load(encoder=False); den = vae.denormalize(LAT)
    for kind, name in (("spatial", "spatial_upscaler_x2_v1_0"), ("temporal", "temporal_upscaler_x2_v1_0")):
        ours = LatentUpscaler(kind).load(); a, t, pk = timed(lambda: ours(den)); print(f"[n2-vae] ours {kind} fp32 {tuple(den.shape)} -> {tuple(a.shape)}: {t:.2f} s peak {pk:.1f} GiB", flush=True)
        r = VideoUpsampler(PACK, name=name).load(); shipped = r(den); mx.eval(shipped)
        r.update(tree_map(lambda p: p.astype(mx.float32), r.parameters())); b = r(den)
        cmp(f"{kind} upscaler ours fp32 vs runner fp32", a, b); cmp(f"{kind} upscaler ours fp32 vs runner as shipped ({shipped.dtype})", a, shipped)
        o16 = LatentUpscaler(kind, dtype=mx.float16).load(); c, t, pk = timed(lambda: o16(den)); print(f"[n2-vae] ours {kind} fp16: {t:.2f} s", flush=True)
        cmp(f"{kind} upscaler ours fp16 vs ours fp32", c, a)
    up = LatentUpscaler("spatial").load().upscale_normalized(LAT, vae); print("[n2-vae] upscale_normalized ->", up.shape, up.dtype, flush=True)
elif mode == "bench121":
    dt = {"fp32": mx.float32, "fp16": mx.float16}[sys.argv[2]]; lat = lat121()
    ours = VideoVAE(dtype=dt).load(encoder=False)
    for mat in (True, False):
        timed(lambda: ours.decode_raw(LAT[:, :, :2], mat))
        px, t, pk = timed(lambda: ours.decode_raw(lat, mat)); print(f"[n2-vae] ours {sys.argv[2]} 768x512x121 untiled decode_raw materialize={mat}: {t:.2f} s peak {pk:.1f} GiB {tuple(px.shape)}", flush=True)
        np.save(f"{OUT}/bench121_{sys.argv[2]}.npy", np.array(px)); del px; mx.clear_cache()
    fr, t, pk = timed(lambda: ours.decode(lat)); print(f"[n2-vae] ours {sys.argv[2]} decode() to uint8 {fr.shape}: {t:.2f} s peak {pk:.1f} GiB", flush=True)
    if len(sys.argv) > 3:
        cmp("121f ours fp16 vs ours fp32", mx.array(np.load(f"{OUT}/bench121_fp16.npy")), mx.array(np.load(f"{OUT}/bench121_fp32.npy")))
elif mode == "profile121":
    dt = {"fp32": mx.float32, "fp16": mx.float16}[sys.argv[2]]
    ours = VideoVAE(dtype=dt).load(encoder=False); ours.decode_raw(LAT[:, :, :2]); rows = []
    V.CONV_HOOK = lambda tag, xs, ws, s: rows.append((tag, xs, ws, s))
    _, t, _ = timed(lambda: ours.decode_raw(lat121())); V.CONV_HOOK = None
    conv = sum(r[3] for r in rows); lines = [f"# conv3d layers, 768x512x121 decode, {sys.argv[2]}; total {t:.2f} s, conv {conv:.2f} s ({100 * conv / t:.0f}%)",
             "| layer | in ch | out ch | kernel | padded input D x H x W | GFLOP | s | share of conv | TF/s |", "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for tag, xs, ws, s in rows:
        d, h, w = xs[1] - 2, xs[2] - 2, xs[3] - 2; fl = 2 * d * h * w * ws[0] * ws[4] * 27 / 1e9
        lines.append(f"| {tag.replace('decoder.', '')} | {ws[4]} | {ws[0]} | 3x3x3 | {xs[1]}x{xs[2]}x{xs[3]} | {fl:.0f} | {s:.3f} | {100 * s / conv:.1f}% | {fl / s / 1e3:.2f} |")
    lines.append(f"| total | | | | | {sum(2 * (r[1][1] - 2) * (r[1][2] - 2) * (r[1][3] - 2) * r[2][0] * r[2][4] * 27 for r in rows) / 1e9:.0f} | {conv:.3f} | | |")
    open(f"{OUT}/conv3d_layers_121_{sys.argv[2]}.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
