/**
 * @file
 * @brief Correctness + throughput harness for the Ampere Q2_K int8-activation matmul.
 *
 * Build (set the arch for YOUR host: sm_80 on A100, sm_86 on RTX 3090):
 *   nvcc q2k_ampere_test.cu -std=c++17 -O3 -arch=sm_80 -o q2k_ampere_test.out
 *   CUDA_VISIBLE_DEVICES=0 ./q2k_ampere_test.out
 *
 * Checks, in order:
 *   1. repack round-trip is EXACT vs the native q2_K dequant formula
 *   2. GEMV matches an fp64 host reference within the int8-activation tolerance
 *   3. effective weight bandwidth vs the measured A100 stream ceiling
 */
#include "q2k_ampere.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace tmq_a100;

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e = (x);                                                      \
        if (e != cudaSuccess) {                                                   \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));     \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Native q2_K dequant, mirroring quant_formats_tables.cuh::q2_K::dequant.
static float native_dequant(const uint8_t* b, int col) {
    const uint8_t* scales = b;
    const uint8_t* qs = b + 16;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 80));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 82));
    const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1,
              l = pos & 15;
    const int is = chunk * 8 + sidx * 2 + sub;
    const int q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
    return d * float(scales[is] & 0xF) * float(q) - dmin * float(scales[is] >> 4);
}

int main() {
    const int N = 32768, K = 6144, M = 4;         // 16 stacked GLM-5.2 w13 experts (> 40 MB L2)
    const int MB = 8;                             // largest batch benchmarked below
    const int nsb = K / 256, nj = K / 16;
    printf("Q2_K Ampere harness: N=%d K=%d M=%d (%.1f MiB of weights)\n", N, K, M,
           (double)N * nsb * 84 / (1 << 20));

    // ---- random but valid q2_K blocks
    std::mt19937 rng(1234);
    std::vector<uint8_t> h_src((size_t)N * nsb * 84);
    for (size_t i = 0; i < (size_t)N * nsb; ++i) {
        uint8_t* b = h_src.data() + i * 84;
        for (int t = 0; t < 16; ++t) b[t] = (uint8_t)(rng() & 0xFF);
        for (int t = 0; t < 64; ++t) b[16 + t] = (uint8_t)(rng() & 0xFF);
        // modest, well-scaled d/dmin so fp16 accumulation is meaningful
        *reinterpret_cast<__half*>(b + 80) = __float2half(0.0035f);
        *reinterpret_cast<__half*>(b + 82) = __float2half(0.0012f);
    }

    uint8_t* d_src;
    uint32_t* d_qp;
    uint8_t* d_sp;
    half2* d_dp;
    CK(cudaMalloc(&d_src, h_src.size()));
    CK(cudaMemcpy(d_src, h_src.data(), h_src.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_qp, (size_t)N * nj * 4));
    CK(cudaMalloc(&d_sp, (size_t)N * nj));
    CK(cudaMalloc(&d_dp, (size_t)N * nsb * 4));

    q2k_repack<<<1024, 256>>>(d_src, d_qp, d_sp, d_dp, N, K);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    // ---- 1. repack must be bit-exact against the native formula
    std::vector<uint32_t> h_qp((size_t)N * nj);
    std::vector<uint8_t> h_sp((size_t)N * nj);
    std::vector<half2> h_dp((size_t)N * nsb);
    CK(cudaMemcpy(h_qp.data(), d_qp, h_qp.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_sp.data(), d_sp, h_sp.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_dp.data(), d_dp, h_dp.size() * 4, cudaMemcpyDeviceToHost));

    long checked = 0, bad = 0;
    for (int n = 0; n < N; n += 1531)
        for (int sb = 0; sb < nsb; ++sb)
            for (int is = 0; is < 16; ++is)
                for (int l = 0; l < 16; ++l) {
                    const int col = q2k_col_of(sb, is, l);
                    const float want = native_dequant(
                        h_src.data() + ((size_t)n * nsb + sb) * 84, col - sb * 256);
                    const float got = q2k_ref_weight(h_qp.data(), h_sp.data(),
                                                     h_dp.data(), n, K, sb * 16 + is, l);
                    ++checked;
                    if (want != got) ++bad;
                }
    printf("  [1] repack round-trip: %ld checked, %ld mismatched -> %s\n", checked, bad,
           bad == 0 ? "EXACT" : "FAIL");
    if (bad) return 1;

    // ---- activations
    std::vector<__half> h_x((size_t)MB * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : h_x) v = __float2half(nd(rng));
    __half* d_x;
    int8_t* d_xq;
    __half* d_xs;
    int32_t* d_xsum;
    float* d_y;
    CK(cudaMalloc(&d_x, (size_t)MB * K * 2));
    CK(cudaMemcpy(d_x, h_x.data(), (size_t)MB * K * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_xq, (size_t)MB * K));
    CK(cudaMalloc(&d_xs, (size_t)MB * nsb * 2));
    CK(cudaMalloc(&d_xsum, (size_t)MB * nj * 4));
    CK(cudaMalloc(&d_y, (size_t)MB * N * 4));

    dim3 qg((nsb + 255) / 256, MB);
    q2k_quant_a8<<<qg, 256>>>(d_x, d_xq, d_xs, d_xsum, MB, K);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    // ---- 2. GEMV vs host reference (double accumulation over the true weights)
    const int WPB = 8, NR = 4, KC = 4096;         // warps/block, rows/warp, k staged
    const size_t SM = q2k_gemv_smem<4, KC>();
    dim3 g((N + WPB * NR - 1) / (WPB * NR)), b(WPB * 32);
    printf("      config: %d warps x %d rows, QB=2, KC=%d, smem %.1f KiB\n", WPB, NR, KC,
           SM / 1024.0);
    q2k_gemv_a8<4, NR, KC, 2><<<g, b, SM>>>(d_qp, d_sp, d_dp, d_xq, d_xs, d_xsum, d_y, M, N, K);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<float> h_y((size_t)MB * N);
    CK(cudaMemcpy(h_y.data(), d_y, (size_t)M * N * 4, cudaMemcpyDeviceToHost));

    double worst = 0.0, refmag = 0.0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; n += 2039) {
            double ref = 0.0;
            for (int sb = 0; sb < nsb; ++sb)
                for (int is = 0; is < 16; ++is)
                    for (int l = 0; l < 16; ++l) {
                        const int col = q2k_col_of(sb, is, l);
                        const double w = native_dequant(
                            h_src.data() + ((size_t)n * nsb + sb) * 84, col - sb * 256);
                        ref += w * (double)__half2float(h_x[(size_t)m * K + col]);
                    }
            worst = fmax(worst, fabs(ref - (double)h_y[(size_t)m * N + n]));
            refmag = fmax(refmag, fabs(ref));
        }
    // int8 activation quantization is the dominant error term (~1/127 relative)
    const double rel = worst / (refmag > 0 ? refmag : 1.0);
    printf("  [2] GEMV vs fp64 ref: max abs %.4f (|ref| max %.1f) -> rel %.4f  %s\n",
           worst, refmag, rel, rel < 0.02 ? "PASS" : "FAIL");
    if (!(rel < 0.02)) return 1;

    // ---- 3. throughput vs batch: shows the DRAM-bound -> ALU-bound crossover
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const double wbytes = (double)N * nsb * 84;   // planes are byte-neutral vs native
    printf("  [3] weight-stream throughput vs batch (ceiling 1769 GB/s measured):\n");
    printf("      %4s %10s %10s %14s %8s\n", "M", "ms", "GB/s", "Gweight/s", "%ceil");
    const int iters = 100;
    #define RUN(MM)                                                                    \
        {                                                                              \
            const size_t sm = q2k_gemv_smem<MM, KC>();                                 \
            for (int i = 0; i < 10; ++i)                                               \
                q2k_gemv_a8<MM, NR, KC, 2><<<g, b, sm>>>(d_qp, d_sp, d_dp, d_xq, d_xs,    \
                                                      d_xsum, d_y, MM, N, K);          \
            CK(cudaDeviceSynchronize());                                               \
            CK(cudaEventRecord(e0));                                                   \
            for (int i = 0; i < iters; ++i)                                            \
                q2k_gemv_a8<MM, NR, KC, 2><<<g, b, sm>>>(d_qp, d_sp, d_dp, d_xq, d_xs,    \
                                                      d_xsum, d_y, MM, N, K);          \
            CK(cudaEventRecord(e1));                                                   \
            CK(cudaDeviceSynchronize());                                               \
            float ms = 0;                                                              \
            CK(cudaEventElapsedTime(&ms, e0, e1));                                     \
            const double gb = wbytes / (ms / iters / 1e3) / 1e9;                        \
            printf("      %4d %10.3f %10.0f %14.0f %7.0f%%\n", MM, ms / iters, gb,     \
                   gb / 0.328125, 100.0 * gb / 1769.0);                                 \
        }
    RUN(1) RUN(2) RUN(4) RUN(8)
    #undef RUN
    return 0;
}
