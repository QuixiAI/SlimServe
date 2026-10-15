"""N2 gate: the engine's distilled pipeline end to end (prompt -> mp4), timed.
Usage: PYTHONPATH=<worktree> python n2_e2e.py OUT_DIR [WIDTH HEIGHT FRAMES] [--pipeline=distilled|dev|dfr] [--bf16-noise] [--keep-text] [--sequential]
Run through gpu_run.py (--need-gb 85 for 768x512x121)."""
import json, os, sys, time, numpy as np, mlx.core as mx
t_start = time.perf_counter()
from slimserve.video.ltx25.pipeline import LTX25Engine
flags = [a for a in sys.argv[1:] if a.startswith("--")]; pos = [a for a in sys.argv[1:] if not a.startswith("--")]
out = pos[0]; w, h, n = (int(x) for x in pos[1:4]) if len(pos) >= 4 else (768, 512, 121)
os.makedirs(out, exist_ok=True)
PROMPT = "A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping"
pipe = next((a.split("=")[1] for a in flags if a.startswith("--pipeline=")), "distilled")
eng = LTX25Engine(variant="dev" if pipe == "dev" else "distilled", bf16_noise="--bf16-noise" in flags)
kw = dict(height=h, width=w, num_frames=n, fps=24.0, seed=42, keep_text="--keep-text" in flags)
if pipe == "dev" and "--sequential" in flags: kw["batched"] = False
res = getattr(eng, pipe)(PROMPT, **kw)
if "--repeat" in flags:  # a second clip on the resident engine: the serving steady state
    mx.eval(res.video_latent, res.audio_tokens); eng.render(res, f"{out}/clip_cold.mp4")
    t_start = time.perf_counter(); res = getattr(eng, pipe)(PROMPT, **kw)
mx.eval(res.video_latent, res.audio_tokens)
np.savez(f"{out}/latents.npz", video=np.array(res.video_latent), audio=np.array(res.audio_tokens))
path = eng.render(res, f"{out}/clip.mp4")
wall = time.perf_counter() - t_start
tm = res.timings
rep = {"wall_s": round(wall, 2), "spans_s": {k: round(v, 2) for k, v in tm.spans.items()},
       "mem_gib_peak_active_cache": {k: [round(x / 2**30, 1) for x in v] for k, v in tm.memory.items()}, "steps_s": [(s, i, round(t, 2)) for s, i, t in tm.steps], "peak_gib": round(mx.get_peak_memory() / 2**30, 1),
       "size": [w, h, n], "mp4": str(path)}
json.dump(rep, open(f"{out}/report.json", "w"), indent=1); print(json.dumps(rep), flush=True)
