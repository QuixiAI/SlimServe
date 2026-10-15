"""N8: keyframe-aware (joint) neighborhood attention parity against upstream.

  ref  OUT.npz        upstream joint_na3d (pure torch, CPU) on fixed random inputs
  ours OUT.npz        our na3d_joint Metal kernel, both passes, on the same inputs
  cmp  REF.npz OURS.npz

The reference is Lightricks' `fallback_na/joint_eager.py` staged standalone in
$LTX25_SCRATCH/n8/ltxkf (torch + no einops). Run `ref` in the torch
env (conda vllm-mlx) and `ours` in venv-slimserve through gpu_run.py.
Usage: python n8_joint_na_parity.py <mode> ..."""

import sys

import numpy as np

T, H, W, NH, HD = 9, 12, 14, 4, 64
P = 3
FRAMES = np.array([0, 3, 7], dtype=np.int64)  # pixel frames of the planes
KERNEL = (3, 5, 5)


def inputs():
    rng = np.random.default_rng(0)
    sh = lambda n: rng.standard_normal((1, n, H, W, NH, HD)).astype(np.float32)
    q, k, v = sh(T), sh(T), sh(T)
    pq, pk, pv = sh(P), sh(P), sh(P)
    # stage with remaining stride 2: t_s(f) = (f + 0.5) / 2, t_s(0) = 0
    times = np.where(FRAMES == 0, 0.0, (FRAMES + 0.5) / 2).astype(np.float32)
    return q * HD**-0.5, k, v, pq * HD**-0.5, pk, pv, times


def ref(out):
    import torch
    from paths import SCRATCH

    sys.path.insert(0, f"{SCRATCH}/n8")
    from ltxkf.joint_eager import joint_na3d

    q, k, v, pq, pk, pv, times = (torch.from_numpy(a) for a in inputs())
    vo, po = joint_na3d(
        q, k, v, pq, pk, pv, times, torch.ones(P, dtype=torch.bool), KERNEL
    )
    np.savez(out, video=vo.numpy(), planes=po.numpy())


def ours(out):
    import mlx.core as mx

    from slimserve.video.ltx25.diffvae import na3d_joint, nearest_slots

    q, k, v, pq, pk, pv, times = inputs()
    to = lambda a: mx.array(a)
    vslots = mx.array(nearest_slots(np.arange(T), times))
    vo = na3d_joint(to(q), to(k), to(v), to(pk), to(pv), vslots, KERNEL)
    pslots = mx.array(nearest_slots(times, np.arange(T)))
    po = na3d_joint(
        to(pq), to(pk), to(pv), to(k), to(v), pslots, (1, KERNEL[1], KERNEL[2])
    )
    np.savez(out, video=np.array(vo), planes=np.array(po))


def cmp(a, b):
    ra, rb = np.load(a), np.load(b)
    for key in ("video", "planes"):
        x, y = ra[key], rb[key]
        rel = np.linalg.norm(x - y) / np.linalg.norm(x)
        print(f"{key}: rel-L2 {rel:.2e} max-abs {np.abs(x - y).max():.2e}")


if __name__ == "__main__":
    {"ref": ref, "ours": ours, "cmp": cmp}[sys.argv[1]](*sys.argv[2:])
