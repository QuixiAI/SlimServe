"""Pure fp16 simdgroup MMA throughput with operands held in registers: the chip's real ceiling for GEMM-shaped work."""
import time, mlx.core as mx
ITERS = 4096
src = f"""
    simdgroup_half8x8 a = make_filled_simdgroup_matrix<half, 8, 8>(half(x[thread_position_in_grid.x % 64]) );
    simdgroup_half8x8 b = make_filled_simdgroup_matrix<half, 8, 8>(half(x[(thread_position_in_grid.x + 7) % 64]));
    simdgroup_half8x8 c0 = make_filled_simdgroup_matrix<half, 8, 8>(0.0h), c1 = c0, c2 = c0, c3 = c0, c4 = c0, c5 = c0, c6 = c0, c7 = c0;
    for (uint i = 0; i < {ITERS}; ++i) {{
        simdgroup_multiply_accumulate(c0, a, b, c0); simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2); simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4); simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6); simdgroup_multiply_accumulate(c7, a, b, c7);
    }}
    threadgroup half tile[8][64];
    simdgroup_store(c0, tile[0], 8); simdgroup_store(c1, tile[1], 8); simdgroup_store(c2, tile[2], 8); simdgroup_store(c3, tile[3], 8);
    simdgroup_store(c4, tile[4], 8); simdgroup_store(c5, tile[5], 8); simdgroup_store(c6, tile[6], 8); simdgroup_store(c7, tile[7], 8);
    half acc = 0.0h;
    for (uint j = 0; j < 8; ++j) acc += tile[j][thread_index_in_simdgroup];
    out[thread_position_in_grid.x] = acc;
"""
k = mx.fast.metal_kernel(name="mma_peak", input_names=["x"], output_names=["out"], source=src)
x = mx.ones((64,), dtype=mx.float32)
for simdgroups in (2048, 8192, 32768):
    threads = 32 * simdgroups
    f = lambda: k(inputs=[x], grid=(threads, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.float16])[0]
    mx.eval(f()); ts = []
    for _ in range(3):
        mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize(); ts.append(time.perf_counter() - t)
    flops = simdgroups * ITERS * 8 * (2 * 8 * 8 * 8)
    print(f"{simdgroups} simdgroups: {min(ts)*1e3:.1f} ms -> {flops/min(ts)/1e12:.1f} TF/s fp16 MMA, register-resident")
# fp32 MMA for comparison
src32 = src.replace("simdgroup_half8x8", "simdgroup_float8x8").replace("<half, 8, 8>", "<float, 8, 8>").replace("0.0h", "0.0f").replace("half(x", "float(x").replace("threadgroup half tile", "threadgroup float tile").replace("half acc = 0.0h", "float acc = 0.0f")
k32 = mx.fast.metal_kernel(name="mma_peak32", input_names=["x"], output_names=["out"], source=src32)
threads = 32 * 8192
f = lambda: k32(inputs=[x], grid=(threads, 1, 1), threadgroup=(128, 1, 1), output_shapes=[(threads,)], output_dtypes=[mx.float32])[0]
mx.eval(f()); mx.synchronize(); t = time.perf_counter(); mx.eval(f()); mx.synchronize(); dt = time.perf_counter() - t
print(f"fp32 MMA: {8192 * ITERS * 8 * 1024 / dt / 1e12:.1f} TF/s")
