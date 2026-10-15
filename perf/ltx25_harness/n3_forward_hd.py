"""Time one engine DiT forward at an arbitrary latent grid (synthetic latents, real text embeddings).
Usage: PYTHONPATH=<worktree> python n3_forward_hd.py F H W [batch]   (16 32 48 = 1536x1024x121 stage 2)"""
import os, sys, pickle, time, numpy as np, mlx.core as mx
from slimserve.video.ltx25 import sampling
from slimserve.video.ltx25.pipeline import LTX25Engine
from paths import SCRATCH
f, h, w = (int(x) for x in sys.argv[1:4]); b = int(sys.argv[4]) if len(sys.argv) > 4 else 1
d = pickle.load(open(f"{SCRATCH}/gate/out/stage2_inputs.pkl", "rb"))["kw"]
rep = lambda x: mx.repeat(x, b, axis=0)
eng = LTX25Engine(); dit = eng.load_dit(); at = sampling.audio_token_count(8 * f - 7, 24.0)
st = sampling.noised_state((b, f * h * w, 128), rep(sampling.video_positions(f, h, w, 24.0)), 1, sigma=0.9, tokens_per_frame=h * w)
au = sampling.noised_state((b, at, 128), rep(sampling.audio_positions(at)), 2, sigma=0.9)
args = (st.latent, au.latent, mx.full((b,), 0.909375), rep(mx.array(d["video_text_embeds"][1])), rep(mx.array(d["audio_text_embeds"][1])), st.positions, au.positions)
ts = []
for i in range(3):
    mx.synchronize(); mx.reset_peak_memory(); t = time.perf_counter(); v, a = dit(*args, video_keyframes_mask=st.keyframes_mask); mx.eval(v, a); mx.synchronize(); ts.append(time.perf_counter() - t)
    print(f"[hd] run {i}: {ts[-1]:.2f} s peak {mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
print(f"[hd] tokens={b}x{f*h*w} forward {min(ts):.2f} s finite={bool(mx.isfinite(v).all().item())}")
