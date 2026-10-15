"""Isolate why in-model linears run slower than isolated GEMMs. Loads the real DiT via the runner,
times block-0 ops with contiguous random inputs, and compares the block's weight arrays against
fresh contiguous copies and against random weights of the same shape."""
import sys, time, mlx.core as mx, mlx.nn as nn
from ltx_pipelines_mlx.distilled import DistilledPipeline
MODEL = sys.argv[1]
pipe = DistilledPipeline(MODEL, low_memory=True, low_ram_streaming=False); pipe.load()
model = pipe.dit; blk = model.transformer_blocks[0]
print("model type:", type(model).__name__, "| block type:", type(blk).__name__, "| ff.proj_in weight:", blk.ff.proj_in.weight.shape, blk.ff.proj_in.weight.dtype, "| linear types:", type(blk.ff.proj_in).__name__, type(blk.attn1.to_q).__name__)
def bench(fn, n=5):
    for _ in range(2): mx.eval(fn())
    mx.synchronize(); t = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    mx.synchronize(); return (time.perf_counter() - t) / n * 1e3
N = 6144
for dt in (mx.bfloat16, mx.float16):
    x = mx.random.normal((1, N, 4096)).astype(dt); mx.eval(x)
    w = blk.ff.proj_in.weight.astype(dt); w_copy = mx.array(w) + 0; w_rand = (mx.random.normal(w.shape) * 0.02).astype(dt); mx.eval(w, w_copy, w_rand)
    print(f"[{dt}] proj_in x@w.T: model weight {bench(lambda: x @ w.T):.1f} ms | contiguous copy {bench(lambda: x @ w_copy.T):.1f} ms | random same shape {bench(lambda: x @ w_rand.T):.1f} ms")
    ff = blk.ff; ff_w = (ff.proj_in.weight.astype(dt), ff.proj_out.weight.astype(dt)); mx.eval(*ff_w)
    ff.proj_in.weight, ff.proj_out.weight = ff_w
    print(f"[{dt}] block0 ff(x) {bench(lambda: ff(x)):.1f} ms | attn1.to_q(x) {bench(lambda: blk.attn1.to_q(x)):.1f} ms (weight {blk.attn1.to_q.weight.shape}, bias={'bias' in blk.attn1.to_q})")
    # full self-attention op without rope / mask
    for n_, m_ in blk.attn1.named_modules():
        if isinstance(m_, nn.Linear): m_.weight = m_.weight.astype(dt); 
    mx.eval(blk.attn1.parameters())
    print(f"[{dt}] block0 attn1(x) no-rope {bench(lambda: blk.attn1(x)):.1f} ms | attn2(x, text 1024) {bench(lambda: blk.attn2(x, mx.random.normal((1,1024,4096)).astype(dt))):.1f} ms")
