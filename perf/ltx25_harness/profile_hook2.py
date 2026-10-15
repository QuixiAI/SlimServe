"""Precise per-op profiler: evaluates inputs before timing each op so lazy upstream glue is not
charged to the op. Tracked: nn.Linear (by weight shape), sdpa (by q/k shape), rms_norm, rope, gelu.
Untracked remainder = elementwise glue (AdaLN modulate, residual adds, gating, reshapes, casts)."""
import time, collections
import mlx.core as mx, mlx.nn as nn
acc = collections.defaultdict(lambda: [0.0, 0])
def _time(key, fn, args):
    mx.eval(*[x for x in args if isinstance(x, mx.array)]); mx.synchronize()
    t = time.perf_counter(); out = fn(); outs = out if isinstance(out, (tuple, list)) else (out,)
    mx.eval(*[o for o in outs if isinstance(o, mx.array)]); mx.synchronize()
    acc[key][0] += time.perf_counter() - t; acc[key][1] += 1; return out
def install():
    _lin = nn.Linear.__call__
    def lin(self, x):
        return _time(f"linear {tuple(self.weight.shape)} M={x.shape[-2] if x.ndim>1 else 1}", lambda: _lin(self, x), (x,))
    nn.Linear.__call__ = lin
    _sdpa = mx.fast.scaled_dot_product_attention
    def sdpa(q, k, v, *a, **kw):
        return _time(f"sdpa q{tuple(q.shape[1:])} k{tuple(k.shape[2:3])} mask={'y' if kw.get('mask') is not None else 'n'}", lambda: _sdpa(q, k, v, *a, **kw), (q, k, v, kw.get("mask")))
    mx.fast.scaled_dot_product_attention = sdpa
    _rms = mx.fast.rms_norm
    def rms(x, *a, **kw): return _time(f"rms_norm {tuple(x.shape[-2:])}", lambda: _rms(x, *a, **kw), (x,))
    mx.fast.rms_norm = rms
    _gelu = nn.gelu_approx
    def gelu(x): return _time(f"gelu {tuple(x.shape[-2:])}", lambda: _gelu(x), (x,))
    nn.gelu_approx = gelu
    import ltx_core_mlx.model.transformer.feed_forward as ffm; ffm.nn.gelu_approx = gelu
    from ltx_core_mlx.model.transformer import rope as rm, attention as am
    for name in ("apply_rope_split", "apply_rope_interleaved"):
        _f = getattr(rm, name)
        def mk(_f, name):
            def w(x, c, s): return _time(f"rope {tuple(x.shape[1:])}", lambda: _f(x, c, s), (x, c, s))
            return w
        setattr(rm, name, mk(_f, name)); setattr(am, name, mk(_f, name))
def report(total):
    tracked = sum(v[0] for v in acc.values())
    print(f"{'op':52s} {'sec':>7s} {'calls':>6s} {'share':>6s}")
    for k, (s, n) in sorted(acc.items(), key=lambda kv: -kv[1][0])[:28]:
        print(f"{k:52s} {s:7.2f} {n:6d} {s/total*100:5.1f}%")
    print(f"{'TRACKED total':52s} {tracked:7.2f} {'':6s} {tracked/total*100:5.1f}%")
    print(f"{'UNTRACKED glue (forward total - tracked)':52s} {total-tracked:7.2f} {'':6s} {(total-tracked)/total*100:5.1f}%")
