"""(1) Mixed-dtype GEMM microbench; (2) log activation dtypes seen by nn.Linear / sdpa inside one real forward."""
import sys, time, collections, mlx.core as mx, mlx.nn as nn
def bench(fn, n=5):
    for _ in range(2): mx.eval(fn())
    mx.synchronize(); t = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter() - t) / n * 1e3
w = (mx.random.normal((4096, 16384)) * 0.02); x32 = mx.random.normal((1, 6144, 16384)); mx.eval(w, x32)
for wd in (mx.bfloat16, mx.float16, mx.float32):
    ww = w.astype(wd); mx.eval(ww)
    print(f"proj_out GEMM (6144x16384x4096): x fp32 @ w {wd}: {bench(lambda: x32 @ ww.T):.1f} ms | x {wd} @ w {wd}: {bench(lambda: x32.astype(wd) @ ww.T):.1f} ms", flush=True)
seen = collections.Counter()
_lin = nn.Linear.__call__
def lin(self, x):
    seen[("linear", tuple(self.weight.shape), str(x.dtype), str(self.weight.dtype))] += 1; return _lin(self, x)
nn.Linear.__call__ = lin
_sdpa = mx.fast.scaled_dot_product_attention
def sdpa(q, k, v, *a, **kw):
    seen[("sdpa", tuple(q.shape[1:]), str(q.dtype), str(k.dtype), str(v.dtype))] += 1; return _sdpa(q, k, v, *a, **kw)
mx.fast.scaled_dot_product_attention = sdpa
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_pipelines_mlx.distilled import DistilledPipeline
class Done(Exception): pass
n = [0]; _orig = LTXModel.__call__
def one(self, *a, **k):
    out = _orig(self, *a, **k); n[0] += 1
    if n[0] >= 1: mx.eval(*[o for o in out if isinstance(o, mx.array)]); raise Done()
    return out
LTXModel.__call__ = one
pipe = DistilledPipeline(sys.argv[1], low_memory=True, low_ram_streaming=False); mx.random.seed(42)
try: pipe.generate_two_stage("a fox in a snowy forest", height=512, width=768, num_frames=121, frame_rate=24.0, seed=42)
except Done: pass
print("\nactivation dtypes in one real forward (stage 1, 1536 tokens):")
for k, v in sorted(seen.items(), key=lambda kv: -kv[1])[:16]: print(f"  {v:4d}x {k}")
blk = pipe.dit.transformer_blocks[0]
print("scale_shift_table dtype:", blk.scale_shift_table.dtype, "| ff.proj_in.weight:", blk.ff.proj_in.weight.dtype, "| attn1.to_q.bias:", blk.attn1.to_q.bias.dtype if 'bias' in blk.attn1.to_q else None)
