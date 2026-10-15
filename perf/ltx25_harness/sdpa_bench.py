import mlx.core as mx, time
H, D = 32, 128
def bench(f, iters=5):
    for _ in range(2): mx.eval(f())
    mx.synchronize(); t=time.perf_counter()
    for _ in range(iters): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/iters
print(f"{'B':>2} {'N':>6} | {'fp16 ms':>8} {'TF/s':>5} | {'bf16 ms':>8} {'TF/s':>5}   (self-attn, 32 heads x 128)")
for B, N in [(1,1536),(4,1536),(1,6144),(4,6144),(1,14080),(1,24576)]:
    fl = 4*B*H*N*N*D/1e12
    row=[]
    for dt in (mx.float16, mx.bfloat16):
        q = mx.random.normal((B,H,N,D)).astype(dt); k = mx.random.normal((B,H,N,D)).astype(dt); v = mx.random.normal((B,H,N,D)).astype(dt); mx.eval(q,k,v)
        t = bench(lambda: mx.fast.scaled_dot_product_attention(q,k,v,scale=D**-0.5)); row.append((t*1e3, fl/t))
    print(f"{B:2d} {N:6d} | {row[0][0]:8.1f} {row[0][1]:5.1f} | {row[1][0]:8.1f} {row[1][1]:5.1f}")
