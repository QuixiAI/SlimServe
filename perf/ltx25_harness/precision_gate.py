"""Precision gate: run the distilled two-stage pipeline with the DiT in a chosen dtype,
save latents, per-block activation maxima, timing, and the decoded mp4.
Usage: LTX_DIT_DTYPE=bf16|fp16 python precision_gate.py --model-dir DIR --out PREFIX [--stats]
"""
import argparse, json, os, sys, time
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prompt", default="A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--height", type=int, default=512)
ap.add_argument("--width", type=int, default=768)
ap.add_argument("--frames", type=int, default=49)
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--no-decode", action="store_true")
args = ap.parse_args()

dtype = os.environ.get("LTX_DIT_DTYPE", "bf16")
stats_on = os.environ.get("LTX_DIT_STATS", "") == "1"
import ltx_core_mlx.model.transformer.model as dit_mod
from ltx_pipelines_mlx.distilled import DistilledPipeline
from ltx_pipelines_mlx import _base

if dtype != "bf16":
    target = {"fp16": mx.float16, "fp32": mx.float32}[dtype]
    _orig = _base.BasePipeline._load_transformer_with_optional_streaming if hasattr(_base, "BasePipeline") else None
    cls = DistilledPipeline
    _orig = cls._load_transformer_with_optional_streaming
    def _cast_loader(self, path):
        model = _orig(self, path)
        from mlx.utils import tree_map
        n = [0]
        def cast(a):
            if isinstance(a, mx.array) and a.dtype == mx.bfloat16:
                n[0] += 1; return a.astype(target)
            return a
        model.update(tree_map(cast, model.parameters()))
        mx.eval(model.parameters())
        print(f"[gate] cast {n[0]} bf16 params to {dtype}", flush=True)
        return model
    cls._load_transformer_with_optional_streaming = _cast_loader

if os.environ.get("LTX_PROFILE"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import profile_hook; profile_hook.install(os.environ["LTX_PROFILE"])
pipe = DistilledPipeline(args.model_dir, low_memory=True, low_ram_streaming=False)
mx.random.seed(args.seed)
t0 = time.perf_counter()
video_latent, audio_latent = pipe.generate_two_stage(
    args.prompt, height=args.height, width=args.width, num_frames=args.frames,
    frame_rate=args.fps, seed=args.seed)
mx.eval(video_latent, audio_latent)
t_gen = time.perf_counter() - t0
peak = mx.get_peak_memory() / 2**30
v = np.array(video_latent.astype(mx.float32)); a = np.array(audio_latent.astype(mx.float32))
np.save(args.out + ".video.npy", v); np.save(args.out + ".audio.npy", a)
meta = {"dtype": dtype, "seed": args.seed, "prompt": args.prompt, "size": [args.height, args.width, args.frames],
        "gen_seconds": t_gen, "peak_gib": peak, "video_shape": list(v.shape), "audio_shape": list(a.shape),
        "video_nan": int(np.isnan(v).sum()), "video_inf": int(np.isinf(v).sum()),
        "video_absmax": float(np.abs(v).max()), "stats": dit_mod.STATS if stats_on else None}
json.dump(meta, open(args.out + ".json", "w"), indent=1)
print(f"[gate] dtype={dtype} gen={t_gen:.1f}s peak={peak:.1f}GiB nan={meta['video_nan']} inf={meta['video_inf']} absmax={meta['video_absmax']:.2f}", flush=True)
if stats_on and dit_mod.STATS:
    per_block = {}
    for b, mv, ma in dit_mod.STATS:
        per_block.setdefault(b, [0, 0]); per_block[b][0] = max(per_block[b][0], mv); per_block[b][1] = max(per_block[b][1], ma)
    worst = max(per_block.items(), key=lambda kv: kv[1][0])
    print(f"[gate] residual max|video| over all forwards: block {worst[0]} = {worst[1][0]:.1f}; fp16 limit 65504", flush=True)
if os.environ.get("LTX_PROFILE"):
    profile_hook.report()
if not args.no_decode:
    t1 = time.perf_counter()
    if pipe.low_memory:
        pipe.dit = None; pipe.text_encoder = None; pipe.feature_extractor = None; pipe._loaded = False
        from ltx_core_mlx.utils.memory import aggressive_cleanup; aggressive_cleanup()
    pipe._load_decoders()
    pipe._decode_and_save_video(video_latent, audio_latent, args.out + ".mp4", frame_rate=getattr(pipe, "source_frame_rate", None) or args.fps)
    print(f"[gate] decode+save {time.perf_counter()-t1:.1f}s -> {args.out}.mp4", flush=True)
    if os.environ.get("LTX_PROFILE"):
        profile_hook.report()
