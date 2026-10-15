import os, sys, numpy as np, mlx.core as mx
import slimserve.video.ltx25.diffvae as D
from paths import SCRATCH
mx.set_cache_limit(16 << 30)
variant = sys.argv[1]
orig = D.na3d
if variant == "sync":
    def na3d(q, k, v, kernel, lanes=8):
        mx.synchronize(); o = orig(q, k, v, kernel, lanes); mx.eval(o); mx.synchronize(); return o
    D.na3d = na3d
elif variant == "sync_before":
    def na3d(q, k, v, kernel, lanes=8):
        mx.synchronize(); return orig(q, k, v, kernel, lanes)
    D.na3d = na3d
elif variant == "eval_inputs":
    def na3d(q, k, v, kernel, lanes=8):
        mx.eval(q, k, v); return orig(q, k, v, kernel, lanes)
    D.na3d = na3d
elif variant == "sync_after":
    def na3d(q, k, v, kernel, lanes=8):
        o = orig(q, k, v, kernel, lanes); mx.eval(o); mx.synchronize(); return o
    D.na3d = na3d
elif variant == "keep_inputs":  # no synchronize; keep q,k,v alive until the output is materialised
    def na3d(q, k, v, kernel, lanes=8):
        o = orig(q, k, v, kernel, lanes); mx.eval(o); o._keep = (q, k, v) if hasattr(o, "__dict__") else None; return o
    D.na3d = na3d
elif variant == "nodel":
    import re, types
    src = open(D.__file__).read()
    print("nodel: patching _attn to not del q,k,v")
    import slimserve.video.ltx25.diffvae as M
    code = src.replace("        del q, k, v\n", "        _keep = (q, k, v)\n")
    ns = {}; exec(compile(code, D.__file__, "exec"), M.__dict__)
elif variant == "lanes1":
    D.na3d = lambda q, k, v, kernel, lanes=1: orig(q, k, v, kernel, 1)
elif variant == "shape_cached":
    _shapes = {}
    src_na3d = D.na3d
    def na3d(q, k, v, kernel, lanes=8):
        return src_na3d(q, k, v, kernel, lanes)
    # patch mx.array used for shape: cache by content
    real_array = mx.array
    def cached_array(data, dtype=None, **kw):
        if dtype == mx.uint32 and isinstance(data, list) and len(data) == 4:
            key = tuple(data)
            if key not in _shapes:
                _shapes[key] = real_array(data, dtype=dtype); mx.eval(_shapes[key])
            return _shapes[key]
        return real_array(data, dtype=dtype, **kw) if dtype is not None else real_array(data, **kw)
    D.mx.array = cached_array
lat = mx.array(np.load(f"{SCRATCH}/demo/d2_latents.npz")["video"])
dec = D.DiffusionVAE().load()
padded, _, _ = dec._pad_to_floor(lat.astype(mx.float32))
fails = 0
for i in range(5):
    taps = {}
    feat = dec.stages_1_to_4(padded, tap=lambda n, x: taps.__setitem__(n, bool(mx.isfinite(x).all().item())))
    mx.eval(feat); ok = bool(mx.isfinite(feat).all().item()) and taps["s4.out"]
    fails += (not ok)
    print(f"{variant}: run {i} s4 finite={taps['s4.out']} feat finite={ok}", flush=True)
print(f"{variant}: {fails}/5 failed")
