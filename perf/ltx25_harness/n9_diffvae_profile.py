"""N9: where the diffusion decoder's time goes at a production shape.

Usage: PYTHONPATH=<worktree> python n9_diffvae_profile.py [F H W] [--keyframes]
Default latent 16x32x48 = 1536x1024x121. Times every det block, every
diffusion block and the attention / MLP inside them with a synchronize, and
reports the per-stage totals and the share of attention, MLP and the rest.
Run through gpu_run.py --need-gb 60."""

import sys
import time
from collections import defaultdict

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import diffvae

pos = [a for a in sys.argv[1:] if not a.startswith("--")]
F, H, W = (int(x) for x in pos[:3]) if len(pos) >= 3 else (16, 32, 48)
KF = "--keyframes" in sys.argv

totals: dict[str, float] = defaultdict(float)
_stack: list[tuple[str, float]] = []


def timed(name, fn):
    def wrapper(self, *a, **k):
        mx.synchronize()
        t = time.perf_counter()
        out = fn(self, *a, **k)
        mx.synchronize()
        totals[name] += time.perf_counter() - t
        return out

    return wrapper


V = diffvae.DiffusionVAE
V._attn = timed("attn (qkv+rope+kernel+proj)", V._attn)
V._plane_attn = timed("plane_attn", V._plane_attn)
V._swiglu = timed("mlp", V._swiglu)
V.stages_1_to_4 = timed("stages_1_to_4", V.stages_1_to_4)
V.stage_5 = timed("stage_5", V.stage_5)


def timed_fn(name, fn):
    def wrapper(*a, **k):
        mx.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        mx.synchronize()
        totals[name] += time.perf_counter() - t
        return out

    return wrapper


diffvae.na3d = timed_fn("na3d kernel", diffvae.na3d)
diffvae.na3d_joint = timed_fn("na3d_joint kernel", diffvae.na3d_joint)
diffvae._axial_rope = timed_fn("rope", diffvae._axial_rope)
diffvae._rms = timed_fn("rms_norm", diffvae._rms)
V._qkv = timed("qkv (proj+rms+rope)", V._qkv)
V._lin = timed("lin (all)", V._lin)

vae = V().load()
rng = np.random.default_rng(0)
latent = mx.array(rng.standard_normal((1, 128, F, H, W)).astype(np.float32))
kf = None
if KF:
    frames = [24 * i for i in range(1, (8 * F - 7) // 24 + 1)]
    planes = mx.array(
        rng.standard_normal((1, 128, len(frames), H, W)).astype(np.float32)
    )
    kf = (planes, frames)
mx.synchronize()
t0 = time.perf_counter()
px = vae.decode_raw(latent, seed=0, keyframes=kf)
mx.eval(px)
mx.synchronize()
wall = time.perf_counter() - t0
print(
    f"decode {F}x{H}x{W} ({8 * F - 7} frames of {32 * H}x{32 * W}) keyframes={KF}: {wall:.1f} s, peak {mx.get_peak_memory() / 2**30:.1f} GiB"
)
for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"  {k:30s} {v:7.1f} s  {100 * v / wall:5.1f}%")
