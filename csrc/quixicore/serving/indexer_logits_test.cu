/**
 * @file
 * @brief Correctness + throughput for the native DSA indexer MQA logits kernel.
 *
 * Build (set the arch for YOUR host: sm_80 on A100, sm_86 on RTX 3090):
 *   nvcc indexer_logits_test.cu -std=c++17 -O3 -arch=sm_80 -I../quant -I. \
 *        -o indexer_logits_test.out
 *   CUDA_VISIBLE_DEVICES=0 ./indexer_logits_test.out
 *
 * Reference is an fp64 host replay of the same formula vLLM's torch fallback
 * computes, so this validates the kernel against the semantics the serving path
 * expects -- not merely against itself.
 */
#include "quant_formats.cuh"
#include "indexer_logits_kernels.cuh"
#include "indexer_logits_mma.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define CK(x)                                                                 \
    do {                                                                      \
        cudaError_t e = (x);                                                  \
        if (e != cudaSuccess) {                                               \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

static float e4m3_host(uint8_t v) {
    const uint32_t s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0) return (s ? -1.f : 1.f) * float(m) * 0.001953125f;
    return (s ? -1.f : 1.f) * ldexpf(1.f + float(m) / 8.f, int(e) - 7);
}

int main() {
    constexpr int D = 128, NTILE = 16;
    const int M = 64, H = 64, N = 512;
    printf("indexer MQA logits: M=%d H=%d D=%d N=%d\n", M, H, D, N);

    std::mt19937 rng(7);
    std::vector<uint8_t> hq((size_t)M * H * D), hk((size_t)N * D);
    for (auto& v : hq) v = (uint8_t)(rng() % 0x7E);      // finite e4m3
    for (auto& v : hk) v = (uint8_t)(rng() % 0x7E);
    std::vector<float> hs(N), hw((size_t)M * H);
    std::uniform_real_distribution<float> ud(0.2f, 1.5f);
    for (auto& v : hs) v = ud(rng);
    for (auto& v : hw) v = ud(rng) - 0.5f;
    std::vector<int> hks(M), hke(M);
    for (int m = 0; m < M; ++m) {
        hks[m] = (int)(rng() % 64);
        hke[m] = hks[m] + 1 + (int)(rng() % (N - hks[m]));
    }

    uint8_t *dq, *dk;
    float *dsc, *dw, *dl;
    int *dks, *dke;
    CK(cudaMalloc(&dq, hq.size()));
    CK(cudaMemcpy(dq, hq.data(), hq.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dk, hk.size()));
    CK(cudaMemcpy(dk, hk.data(), hk.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsc, N * 4));
    CK(cudaMemcpy(dsc, hs.data(), N * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dw, hw.size() * 4));
    CK(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dks, M * 4));
    CK(cudaMemcpy(dks, hks.data(), M * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dke, M * 4));
    CK(cudaMemcpy(dke, hke.data(), M * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dl, (size_t)M * N * 4));

    const int threads = H;
    const size_t sm = tms::indexer_mqa_logits_smem(D, H, threads);
    dim3 grid((N + NTILE - 1) / NTILE, M);
    printf("  smem %.1f KiB, grid %dx%d, %d threads\n", sm / 1024.0, grid.x,
           grid.y, threads);
    tms::indexer_mqa_logits_fp8<D, NTILE>
        <<<grid, threads, sm>>>(dq, dk, dsc, dw, dks, dke, dl, M, N, H);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<float> hl((size_t)M * N);
    CK(cudaMemcpy(hl.data(), dl, hl.size() * 4, cudaMemcpyDeviceToHost));

    // fp64 host replay of the reference formula
    double worst = 0, mag = 0;
    long checked = 0, badmask = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; n += 7) {
            const bool live = (n >= hks[m] && n < hke[m]);
            const float got = hl[(size_t)m * N + n];
            if (!live) {
                if (!(std::isinf(got) && got < 0)) ++badmask;
                continue;
            }
            double sum = 0.0;
            for (int h = 0; h < H; ++h) {
                double dot = 0.0;
                for (int d = 0; d < D; ++d)
                    dot += (double)e4m3_host(hq[((size_t)m * H + h) * D + d]) *
                           (double)e4m3_host(hk[(size_t)n * D + d]);
                double sc = dot * (double)hs[n];
                sum += (sc > 0 ? sc : 0.0) * (double)hw[(size_t)m * H + h];
            }
            worst = std::fmax(worst, std::fabs(sum - (double)got));
            mag = std::fmax(mag, std::fabs(sum));
            ++checked;
        }
    const double rel = worst / (mag > 0 ? mag : 1.0);
    printf("  vs fp64 ref: %ld live checked, max abs %.4f (|ref| %.1f) rel %.2e %s\n",
           checked, worst, mag, rel, rel < 1e-5 ? "PASS" : "FAIL");
    printf("  mask: %ld wrong out of range -> %s\n", badmask,
           badmask == 0 ? "PASS" : "FAIL");
    if (rel >= 1e-5 || badmask) return 1;

    // throughput at a realistic prefill shape
    const int M2 = 512, N2 = 4096;
    uint8_t* dq2;
    float *dw2, *dl2;
    int *dks2, *dke2;
    std::vector<uint8_t> hq2((size_t)M2 * H * D);
    for (auto& v : hq2) v = (uint8_t)(rng() % 0x7E);
    std::vector<float> hw2((size_t)M2 * H, 0.5f);
    std::vector<int> ks2(M2, 0), ke2(M2, N2);
    std::vector<uint8_t> hk2((size_t)N2 * D);
    for (auto& v : hk2) v = (uint8_t)(rng() % 0x7E);
    std::vector<float> hs2(N2, 1.0f);
    uint8_t* dk2;
    float* dsc2;
    CK(cudaMalloc(&dq2, hq2.size()));
    CK(cudaMemcpy(dq2, hq2.data(), hq2.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dk2, hk2.size()));
    CK(cudaMemcpy(dk2, hk2.data(), hk2.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsc2, N2 * 4));
    CK(cudaMemcpy(dsc2, hs2.data(), N2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dw2, hw2.size() * 4));
    CK(cudaMemcpy(dw2, hw2.data(), hw2.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dks2, M2 * 4));
    CK(cudaMemcpy(dks2, ks2.data(), M2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dke2, M2 * 4));
    CK(cudaMemcpy(dke2, ke2.data(), M2 * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dl2, (size_t)M2 * N2 * 4));
    dim3 g2((N2 + NTILE - 1) / NTILE, M2);

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; ++i)
        tms::indexer_mqa_logits_fp8<D, NTILE>
            <<<g2, threads, sm>>>(dq2, dk2, dsc2, dw2, dks2, dke2, dl2, M2, N2, H);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    const int it = 20;
    for (int i = 0; i < it; ++i)
        tms::indexer_mqa_logits_fp8<D, NTILE>
            <<<g2, threads, sm>>>(dq2, dk2, dsc2, dw2, dks2, dke2, dl2, M2, N2, H);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double sec = ms / it / 1e3;
    const double macs = (double)M2 * N2 * H * D;
    printf("  scalar : %.3f ms  -> %6.1f GMAC/s\n", ms / it, macs / sec / 1e9);

    // ---- tensor-core variant: correctness on the small case, then throughput
    constexpr int NT = 64, NWARP = 4;
    const size_t sm2 = tms::indexer_mqa_logits_mma_smem<D, NT, NWARP>();
    dim3 gs((N + NT - 1) / NT, M);
    CK(cudaMemset(dl, 0, (size_t)M * N * 4));
    tms::indexer_mqa_logits_mma<D, NT, NWARP>
        <<<gs, 32 * NWARP, sm2>>>(dq, dk, dsc, dw, dks, dke, dl, M, N, H);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<float> hm((size_t)M * N);
    CK(cudaMemcpy(hm.data(), dl, hm.size() * 4, cudaMemcpyDeviceToHost));
    double w2_ = 0, m2_ = 0;
    long bm = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; n += 7) {
            const bool live = (n >= hks[m] && n < hke[m]);
            const float got = hm[(size_t)m * N + n];
            if (!live) { if (!(std::isinf(got) && got < 0)) ++bm; continue; }
            const float ref = hl[(size_t)m * N + n];
            w2_ = std::fmax(w2_, std::fabs((double)ref - (double)got));
            m2_ = std::fmax(m2_, std::fabs((double)ref));
        }
    const double rel2 = w2_ / (m2_ > 0 ? m2_ : 1.0);
    printf("  mma vs scalar: rel %.2e, mask bad %ld -> %s\n", rel2, bm,
           (rel2 < 5e-3 && bm == 0) ? "PASS" : "FAIL");

    dim3 gs2((N2 + NT - 1) / NT, M2);
    for (int i = 0; i < 3; ++i)
        tms::indexer_mqa_logits_mma<D, NT, NWARP>
            <<<gs2, 32 * NWARP, sm2>>>(dq2, dk2, dsc2, dw2, dks2, dke2, dl2, M2, N2, H);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < it; ++i)
        tms::indexer_mqa_logits_mma<D, NT, NWARP>
            <<<gs2, 32 * NWARP, sm2>>>(dq2, dk2, dsc2, dw2, dks2, dke2, dl2, M2, N2, H);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms2 = 0;
    CK(cudaEventElapsedTime(&ms2, e0, e1));
    printf("  mma    : %.3f ms  -> %6.1f GMAC/s   (%.1fx over scalar; torch ref "
           "= 3.83 ms and needs %.1f GiB scratch)\n",
           ms2 / it, macs / (ms2 / it / 1e3) / 1e9, ms / ms2,
           (double)H * M2 * N2 * 4 / (1 << 30));
    if (rel2 >= 5e-3 || bm) return 1;

    // ---- pre-decoded variant: hoist e4m3 -> half out of the GEMM blocks
    half *qh, *kh;
    CK(cudaMalloc(&qh, (size_t)M2 * H * D * 2));
    CK(cudaMalloc(&kh, (size_t)N2 * D * 2));
    auto run_pre = [&]() {
        tms::indexer_decode_e4m3<<<1024, 256>>>(dq2, qh, (size_t)M2 * H * D);
        tms::indexer_decode_e4m3<<<1024, 256>>>(dk2, kh, (size_t)N2 * D);
        tms::indexer_mqa_logits_mma_pre<D, NT, NWARP>
            <<<gs2, 32 * NWARP, sm2>>>(qh, kh, dsc2, dw2, dks2, dke2, dl2, M2, N2, H);
    };
    for (int i = 0; i < 3; ++i) run_pre();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < it; ++i) run_pre();
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms3 = 0;
    CK(cudaEventElapsedTime(&ms3, e0, e1));
    printf("  mma+pre: %.3f ms  -> %6.1f GMAC/s   (%.2fx vs torch 3.83 ms, "
           "and needs ~9 MiB not %.1f GiB)\n",
           ms3 / it, macs / (ms3 / it / 1e3) / 1e9, 3.83 / (ms3 / it),
           (double)H * M2 * N2 * 4 / (1 << 30));
    return 0;
}
