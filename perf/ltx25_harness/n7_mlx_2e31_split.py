import mlx.core as mx, numpy as np
mx.random.seed(0)
rows = 129 * 80 * 152   # 1,568,640 stage-4 tokens
x = (mx.random.normal((rows, 512)) * 0.3).astype(mx.float16); w = (mx.random.normal((1536, 512)) * 0.05).astype(mx.float16); b = mx.zeros((1536,), dtype=mx.float16); mx.eval(x, w, b)
y = x @ w.T + b; mx.eval(y)
print("output elements", y.size / 1e9, "G")
for name, sl in (("first 1000 rows", slice(0, 1000)), ("rows at 1.3M", slice(1_300_000, 1_301_000)), ("rows at 1.40M", slice(1_400_000, 1_401_000)), ("rows at 1.45M", slice(1_450_000, 1_451_000)), ("last 1000 rows", slice(rows - 1000, rows))):
    ref = x[sl] @ w.T + b
    print(f"{name}: max|d| vs row-slice GEMM {float(mx.abs(y[sl].astype(mx.float32) - ref.astype(mx.float32)).max()):.2e}", flush=True)
# where does it start to break?
lo, hi = 0, rows
step = 50_000
first_bad = None
for r0 in range(0, rows, step):
    ref = x[r0:r0 + 64] @ w.T + b
    if float(mx.abs(y[r0:r0 + 64].astype(mx.float32) - ref.astype(mx.float32)).max()) > 1e-2:
        first_bad = r0; break
print("first bad row block:", first_bad, "-> element index", None if first_bad is None else first_bad * 1536 / 2**31, "x 2^31")
# rms_norm on 803M elements, and a split/reshape of the big tensor
z = mx.fast.rms_norm(x, mx.ones((512,), dtype=mx.float16), 1e-6); zr = mx.fast.rms_norm(x[-1000:], mx.ones((512,), dtype=mx.float16), 1e-6)
print("rms_norm last rows max|d|", float(mx.abs(z[-1000:].astype(mx.float32) - zr.astype(mx.float32)).max()))
q, k, v = mx.split(y, 3, axis=-1); mx.eval(q)
print("split last row ok:", bool(mx.array_equal(q[-1], y[-1, :512])))
