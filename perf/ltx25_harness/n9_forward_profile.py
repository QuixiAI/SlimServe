"""N9: per-op profile of one DiT forward at an arbitrary latent shape.

Usage: PYTHONPATH=<worktree> python n9_forward_profile.py F H W [distilled|dev] [BATCH]
       (through gpu_run.py --need-gb 60; BATCH 4 = the dev guided step's batch)
Synthetic tokens and text context at the given shape (16 32 48 = stage 2 of
1536x1024x121; 16 16 24 = its stage 1). Every timed op is evaluated
synchronously (inputs first), so absolutes are inflated; the clean forward
time, the shares and the per-op TF/s against the 19.4 TF/s MMA ceiling
(section 17) are the output."""

import collections
import sys
import time

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints, sampling
from slimserve.video.ltx25 import dit as D

F, H, W = (int(x) for x in sys.argv[1:4])
variant = sys.argv[4] if len(sys.argv) > 4 else "distilled"
B = int(sys.argv[5]) if len(sys.argv) > 5 else 1
FPS, SIGMA, CEIL = 24.0, 0.7, 19.4e12

weights, _, tcfg = checkpoints.load_dit(variant)
m = D.LTX25DiT(weights, D.DiTConfig.from_checkpoint(tcfg))
rng = np.random.default_rng(0)
n_audio = sampling.audio_token_count((F - 1) * 8 + 1, FPS)
vt = mx.array(rng.standard_normal((B, F * H * W, 128)).astype(np.float32))
at = mx.array(rng.standard_normal((B, n_audio, 128)).astype(np.float32))
vtext = mx.array((rng.standard_normal((B, 1024, 4096)) * 0.5).astype(np.float32))
atext = mx.array((rng.standard_normal((B, 1024, 2048)) * 0.5).astype(np.float32))
kf = mx.broadcast_to(
    (mx.arange(F * H * W) < H * W).astype(mx.float32).reshape(1, -1, 1),
    (B, F * H * W, 1),
)
vpos = mx.broadcast_to(sampling.video_positions(F, H, W, FPS), (B, F * H * W, 3))
apos = mx.broadcast_to(sampling.audio_positions(n_audio), (B, n_audio, 1))
args = dict(
    video_latent=vt,
    audio_latent=at,
    timestep=mx.full((B,), SIGMA),
    video_text=vtext,
    audio_text=atext,
    video_positions=vpos,
    audio_positions=apos,
    video_keyframes_mask=kf,
)


def clean():
    mx.synchronize()
    t = time.perf_counter()
    v, a = m(**args)
    mx.eval(v, a)
    mx.synchronize()
    return time.perf_counter() - t


clean()
t_clean = min(clean() for _ in range(2))
acc = collections.defaultdict(lambda: [0.0, 0, 0.0])


def cls(name):
    n = (
        name.split(".", 2)[-1]
        if name.startswith("transformer_blocks")
        else "top." + name
    )
    for a, b in (("audio_to_video_attn", "a2v"), ("video_to_audio_attn", "v2a")):
        n = n.replace(a, b)
    return n


def timed(key, fn, ins, flops=0.0):
    mx.eval(*ins)
    mx.synchronize()
    t = time.perf_counter()
    y = fn()
    mx.eval(y)
    mx.synchronize()
    r = acc[key]
    r[0] += time.perf_counter() - t
    r[1] += 1
    r[2] += flops
    return y


_lin = m.lin


def lin(name, x):
    w = m.w.get(name + ".weight")
    n, k = w.shape if w is not None else (m.split_k[name][0].shape[0], 16384)
    return timed(
        "lin " + cls(name),
        lambda: _lin(name, x),
        [x],
        2.0 * (x.size // x.shape[-1]) * k * n,
    )


m.lin = lin
_sdpa = mx.fast.scaled_dot_product_attention


def sdpa(q, k, v, **kw):
    fl = 4.0 * q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.shape[3]
    return timed(
        f"sdpa q{q.shape[2]} k{k.shape[2]} d{q.shape[3]}",
        lambda: _sdpa(q, k, v, **kw),
        [q, k, v],
        fl,
    )


mx.fast.scaled_dot_product_attention = sdpa
mx.synchronize()
t = time.perf_counter()
v, a = m(**args)
mx.eval(v, a)
mx.synchronize()
t_prof = time.perf_counter() - t
tracked = sum(r[0] for r in acc.values())
flops_total = sum(r[2] for r in acc.values())
useful = flops_total / t_clean
glue = t_prof - tracked
print(
    f"{variant} {F}x{H}x{W} batch {B}: {F * H * W} video + {n_audio} audio tokens; "
    f"clean forward {t_clean:.2f} s = {useful / 1e12:.1f} TF/s useful "
    f"({100 * useful / CEIL:.0f}% of the MMA ceiling); profiled {t_prof:.2f} s; "
    f"tracked {tracked:.2f} s; glue {glue:.2f} s = {100 * glue / t_prof:.0f}%"
)
print("| op | calls | s | share | TF/s | of ceiling |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
groups = collections.defaultdict(lambda: [0.0, 0.0])
for k, r in sorted(acc.items(), key=lambda kv: -kv[1][0]):
    g = (
        "sdpa"
        if k.startswith("sdpa")
        else (
            "lin video"
            if "attn1" in k or "ff." in k or "attn2" in k and "audio" not in k
            else "lin other"
        )
    )
    if k.startswith("lin audio") or "audio_ff" in k or "audio_attn" in k:
        g = "lin audio"
    groups[g][0] += r[0]
    groups[g][1] += r[2]
    if r[0] > 0.02:
        print(
            f"| {k} | {r[1]} | {r[0]:.2f} | {100 * r[0] / t_prof:.1f}% | "
            f"{r[2] / r[0] / 1e12:.1f} | {100 * r[2] / r[0] / CEIL:.0f}% |"
        )
print("groups (s, TF/s, headroom to ceiling in s):")
for g, (s, fl) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
    print(f"  {g:10s} {s:6.2f} s  {fl / s / 1e12:5.1f} TF/s  {s - fl / CEIL:5.2f} s")
