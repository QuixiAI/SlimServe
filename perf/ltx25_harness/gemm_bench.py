import mlx.core as mx, time
# LTX-2.5 video-stream DiT shapes: hidden 4096, FFN 16384. M = tokens (stage1 1536, stage2 6144, 720p 14080)
shapes = [(1536,4096,16384),(6144,4096,16384),(6144,16384,4096),(6144,4096,4096),(14080,4096,16384)]
def bench(f, iters=10):
    for _ in range(3): mx.eval(f())
    mx.synchronize(); t=time.perf_counter()
    for _ in range(iters): mx.eval(f())
    mx.synchronize(); return (time.perf_counter()-t)/iters
print(f"{'M':>6} {'K':>6} {'N':>6} | {'fp16 ms':>8} {'TF/s':>6} | {'bf16 ms':>8} {'TF/s':>6} | {'int8g64 ms':>10} {'TF/s':>6} | {'int4g64 ms':>10} {'TF/s':>6}")
for M,K,N in shapes:
    x16 = mx.random.normal((M,K)).astype(mx.float16); w16 = (mx.random.normal((N,K))*0.02).astype(mx.float16)
    xb = x16.astype(mx.bfloat16); wb = w16.astype(mx.bfloat16)
    w8,s8,b8 = mx.quantize(w16, group_size=64, bits=8)
    w4,s4,b4 = mx.quantize(w16, group_size=64, bits=4)
    mx.eval(x16,w16,xb,wb,w8,s8,b8,w4,s4,b4)
    fl = 2*M*K*N/1e12
    t16 = bench(lambda: x16 @ w16.T); tb = bench(lambda: xb @ wb.T)
    t8 = bench(lambda: mx.quantized_matmul(x16,w8,s8,b8,transpose=True,group_size=64,bits=8))
    t4 = bench(lambda: mx.quantized_matmul(x16,w4,s4,b4,transpose=True,group_size=64,bits=4))
    print(f"{M:6d} {K:6d} {N:6d} | {t16*1e3:8.1f} {fl/t16:6.1f} | {tb*1e3:8.1f} {fl/tb:6.1f} | {t8*1e3:10.1f} {fl/t8:6.1f} | {t4*1e3:10.1f} {fl/t4:6.1f}")
