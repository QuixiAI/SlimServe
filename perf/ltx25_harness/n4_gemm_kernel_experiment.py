import time, mlx.core as mx
def make(im, jn, variant):
    tm, tn = 8 * im, 8 * jn
    decl = "".join(f"simdgroup_half8x8 a{i}{j} = make_filled_simdgroup_matrix<half, 8, 8>(0.0h);\n" for i in range(im) for j in range(jn))
    decl += "".join(f"simdgroup_half8x8 ma{i};\n" for i in range(im)) + "".join(f"simdgroup_half8x8 mb{j};\n" for j in range(jn))
    loads = "".join(f"simdgroup_load(ma{i}, xp + {i}*8*K + k, K);\n" for i in range(im)) + "".join(f"simdgroup_load(mb{j}, wp + {j}*8*K + k, K, 0, true);\n" for j in range(jn))
    mma = "".join(f"simdgroup_multiply_accumulate(a{i}{j}, ma{i}, mb{j}, a{i}{j});\n" for i in range(im) for j in range(jn))
    store = "".join(f"simdgroup_store(a{i}{j}, op + {i}*8*N + {j}*8, N);\n" for i in range(im) for j in range(jn))
    if variant == "sg":   # one simdgroup per threadgroup, tile from threadgroup position
        head = f"const uint n0 = (thread_position_in_grid.x / 32) * {tn}; const uint m0 = thread_position_in_grid.y * {tm};"
    src = f"""
    const uint K = x_shape[1]; const uint N = w_shape[0];
    {head}
    const device half* xp = x + m0 * K; const device half* wp = w + n0 * K; device half* op = out + m0 * N + n0;
    {decl}
    for (uint k = 0; k < K; k += 8) {{ {loads} {mma} }}
    {store}
    """
    k = mx.fast.metal_kernel(name=f"gemm2_{im}x{jn}", input_names=["x", "w"], output_names=["out"], source=src)
    return lambda x, w: k(inputs=[x, w], grid=(32 * (w.shape[0] // tn), x.shape[0] // tm, 1), threadgroup=(32, 1, 1), output_shapes=[(x.shape[0], w.shape[0])], output_dtypes=[mx.float16])[0]
def bench(f, n=4):
    mx.eval(f()); ts = []
    for _ in range(n):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize(); ts.append(time.perf_counter() - t)
    return min(ts)
M, K, N = 6144, 4096, 4096
x = (mx.random.normal((M, K)) * 0.5).astype(mx.float16); w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16); mx.eval(x, w); fl = 2.0 * M * K * N
t = bench(lambda: x @ w.T); print(f"MLX {t*1e3:.1f} ms {fl/t/1e12:.1f} TF/s")
ref = x @ w.T
for im, jn in ((1, 1), (2, 2), (4, 4), (2, 4), (4, 8), (8, 8)):
    try:
        f = make(im, jn, "sg"); y = f(x, w); mx.eval(y)
        d = float(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32)).max()); t = bench(lambda: f(x, w))
        print(f"unrolled tile {8*im}x{8*jn}: {t*1e3:.1f} ms {fl/t/1e12:.2f} TF/s max|d| vs MLX {d:.3f}", flush=True)
    except Exception as e: print(f"tile {8*im}x{8*jn} FAILED {str(e)[:400]}", flush=True)
# small K to separate memory from compute
for Ks in (256, 1024):
    xs = x[:, :Ks]; ws = mx.contiguous(w[:, :Ks]); xs = mx.contiguous(xs); mx.eval(xs, ws); f = make(4, 4, "sg"); fls = 2.0 * M * Ks * N
    t = bench(lambda: f(xs, ws)); t2 = bench(lambda: xs @ ws.T); print(f"K={Ks}: kernel {fls/t/1e12:.2f} TF/s, MLX {fls/t2/1e12:.1f} TF/s")
