"""Per-op profile of the engine DiT forward on the captured stage-2 inputs. Every timed op is evaluated
synchronously (inputs first), so absolutes are inflated; shares and per-op TF/s are the output.
Usage: PYTHONPATH=<worktree> python n3_forward_profile.py   (through gpu_run.py --need-gb 48)"""
import os, pickle, time, collections, numpy as np, mlx.core as mx
from slimserve.video.ltx25 import checkpoints, dit as D
from paths import SCRATCH
weights, _, tcfg = checkpoints.load_dit("distilled")
m = D.LTX25DiT(weights, D.DiTConfig.from_checkpoint(tcfg))
d = pickle.load(open(f"{SCRATCH}/gate/out/stage2_inputs.pkl", "rb"))["kw"]
g = lambda k: None if d.get(k) is None else mx.array(d[k][1])
args = dict(video_latent=g("video_latent"), audio_latent=g("audio_latent"), timestep=g("timestep"), video_text=g("video_text_embeds"),
            audio_text=g("audio_text_embeds"), video_positions=g("video_positions"), audio_positions=g("audio_positions"), video_keyframes_mask=g("video_keyframes_mask"))
def clean():
    mx.synchronize(); t = time.perf_counter(); v, a = m(**args); mx.eval(v, a); mx.synchronize(); return time.perf_counter() - t
clean(); t_clean = min(clean() for _ in range(2))
acc = collections.defaultdict(lambda: [0.0, 0, 0.0])
def cls(name):
    n = name.split(".", 2)[-1] if name.startswith("transformer_blocks") else "top." + name
    for a, b in (("audio_to_video_attn", "a2v"), ("video_to_audio_attn", "v2a")): n = n.replace(a, b)
    return n
def timed(key, fn, ins, flops=0.0):
    mx.eval(*ins); mx.synchronize(); t = time.perf_counter(); y = fn(); mx.eval(y); mx.synchronize()
    r = acc[key]; r[0] += time.perf_counter() - t; r[1] += 1; r[2] += flops; return y
_lin = m.lin
def lin(name, x):
    w = m.w.get(name + ".weight"); n, k = (w.shape if w is not None else (m.split_k[name][0].shape[0], 16384))
    return timed("lin " + cls(name), lambda: _lin(name, x), [x], 2.0 * (x.size // x.shape[-1]) * k * n)
m.lin = lin
_sdpa = mx.fast.scaled_dot_product_attention
def sdpa(q, k, v, **kw):
    fl = 4.0 * q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.shape[3]
    return timed(f"sdpa q{q.shape[2]} k{k.shape[2]} d{q.shape[3]}", lambda: _sdpa(q, k, v, **kw), [q, k, v], fl)
mx.fast.scaled_dot_product_attention = sdpa
mx.synchronize(); t = time.perf_counter(); v, a = m(**args); mx.eval(v, a); mx.synchronize(); t_prof = time.perf_counter() - t
tracked = sum(r[0] for r in acc.values())
print(f"clean forward {t_clean:.2f} s; profiled {t_prof:.2f} s; tracked ops {tracked:.2f} s; glue (norm, modulate, rope, gate, casts, residual) {t_prof - tracked:.2f} s = {100*(t_prof-tracked)/t_prof:.0f}%")
print("| op | calls | s | share | TF/s |\n| --- | ---: | ---: | ---: | ---: |")
for k, r in sorted(acc.items(), key=lambda kv: -kv[1][0]):
    if r[0] > 0.02: print(f"| {k} | {r[1]} | {r[0]:.2f} | {100*r[0]/t_prof:.1f}% | {r[2]/r[0]/1e12:.1f} |")
