/**
 * @file
 * @brief Geometry test for the templated sparse MLA fp8 decode.
 *
 * Build (set the arch for YOUR host: sm_80 on A100, sm_86 on RTX 3090):
 *   nvcc mla_glm_test.cu -std=c++17 -O3 -arch=sm_80 -I../quant -I. -o mla_glm_test.out
 *   CUDA_VISIBLE_DEVICES=0 ./mla_glm_test.out
 *
 * Covers both instantiations of mla_decode_fp8_v:
 *   1. GLM-5.2-Vision  <576, 512, 576, SMODE=1>  576 fp8, value = leading 512,
 *      single per-tensor kv_scale (the vLLM fp8 MLA cache layout)
 *   2. DeepSeek-style  <512, 512, 448, SMODE=0>  448 fp8 + 64 bf16 rope, per-64
 *      e8 exponents -- the original geometry, which must not regress
 * Each is checked against an fp64 host replay of the same softmax/gather walk.
 */
#include "quant_formats.cuh"
#include "tm_warp.cuh"
#include "mla_kernels.cuh"
#include "paged_attn_v2_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>

#define CK(x)                                                                     \
    do {                                                                          \
        cudaError_t e = (x);                                                      \
        if (e != cudaSuccess) {                                                   \
            printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));     \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static float e4m3_decode_host(uint8_t v) {
    const uint32_t s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
    if (e == 0) return (s ? -1.f : 1.f) * float(m) * 0.001953125f;   // 2^-9
    return (s ? -1.f : 1.f) * ldexpf(1.f + float(m) / 8.f, int(e) - 7);
}

// Host replay of the kernel's sparse walk, in double.
template <int QW, int VW, int NFP8, int SMODE>
static void host_ref(const std::vector<float>& q, const std::vector<uint8_t>& data,
                     const std::vector<uint8_t>& scl, const std::vector<int>& bt,
                     const std::vector<int>& idx, const std::vector<int>& tlen,
                     int B, int H, int block_size, int bt_stride, float scale,
                     int max_topk, float kv_scale, std::vector<float>& out) {
    constexpr int SLOT_BYTES = NFP8 + 2 * (QW - NFP8);
    out.assign((size_t)B * H * VW, 0.f);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < H; ++h) {
            std::vector<double> acc(VW, 0.0);
            double m = -3.4e38, l = 0.0;
            for (int j = 0; j < tlen[b]; ++j) {
                const int t = idx[(size_t)b * max_topk + j];
                if (t < 0) continue;
                const int col = t / block_size, slot = t - col * block_size;
                const int blk = bt[(size_t)b * bt_stride + col];
                if (blk < 0) continue;
                const size_t ds = (size_t)blk * block_size + slot;
                const uint8_t* d0 = data.data() + ds * SLOT_BYTES;
                std::vector<double> lat(QW);
                double partial = 0.0;
                for (int d = 0; d < QW; ++d) {
                    if (d < NFP8) {
                        const double dq = e4m3_decode_host(d0[d]);
                        lat[d] = SMODE == 0
                                     ? dq * std::exp2((double)scl[ds * (NFP8 / 64) +
                                                                 d / 64] - 127.0)
                                     : dq * (double)kv_scale;
                    } else {
                        const uint16_t raw = *reinterpret_cast<const uint16_t*>(
                            d0 + NFP8 + 2 * (d - NFP8));
                        const uint32_t bits = (uint32_t)raw << 16;
                        float f;
                        memcpy(&f, &bits, 4);
                        lat[d] = f;
                    }
                    partial += (double)q[((size_t)b * H + h) * QW + d] * lat[d];
                }
                const double score = partial * scale;
                const double nm = std::fmax(m, score);
                const double alpha = (l == 0.0) ? 0.0 : std::exp(m - nm);
                const double beta = std::exp(score - nm);
                for (int d = 0; d < VW; ++d) acc[d] = acc[d] * alpha + beta * lat[d];
                l = l * alpha + beta;
                m = nm;
            }
            for (int d = 0; d < VW; ++d)
                out[((size_t)b * H + h) * VW + d] = l == 0.0 ? 0.f : (float)(acc[d] / l);
        }
}

template <int QW, int VW, int NFP8, int SMODE>
static int run_case(const char* name, float kv_scale) {
    constexpr int SLOT_BYTES = NFP8 + 2 * (QW - NFP8);
    const int B = 3, H = 4, block_size = 64, nblocks = 32, bt_stride = 8;
    const int max_topk = 96;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> hq((size_t)B * H * QW);
    for (auto& v : hq) v = nd(rng) * 0.05f;
    std::vector<uint8_t> data((size_t)nblocks * block_size * SLOT_BYTES);
    for (auto& v : data) v = (uint8_t)(rng() & 0x7F);      // keep fp8 finite
    // The trailing QW-NFP8 elements are bf16, not fp8: random bytes there encode
    // ~1e38 and overflow the score, so write real small bf16 values instead.
    if (QW > NFP8)
        for (size_t s0 = 0; s0 < (size_t)nblocks * block_size; ++s0)
            for (int d = NFP8; d < QW; ++d) {
                const float f = nd(rng) * 0.1f;
                uint32_t bits;
                memcpy(&bits, &f, 4);
                const uint16_t raw = (uint16_t)(bits >> 16);
                memcpy(data.data() + s0 * SLOT_BYTES + NFP8 + 2 * (d - NFP8), &raw, 2);
            }
    std::vector<uint8_t> scl((size_t)nblocks * block_size * (NFP8 / 64));
    for (auto& v : scl) v = (uint8_t)(120 + (rng() % 8));   // exponents near 2^0
    std::vector<int> bt((size_t)B * bt_stride);
    for (auto& v : bt) v = (int)(rng() % nblocks);
    std::vector<int> idx((size_t)B * max_topk), tlen(B);
    for (int b = 0; b < B; ++b) {
        tlen[b] = 40 + (int)(rng() % 40);
        for (int j = 0; j < max_topk; ++j)
            idx[(size_t)b * max_topk + j] =
                (j % 11 == 5) ? -1 : (int)(rng() % (bt_stride * block_size));
    }
    const float scale = 0.1f;

    std::vector<__nv_bfloat16> qb(hq.size());
    for (size_t i = 0; i < hq.size(); ++i) qb[i] = __float2bfloat16(hq[i]);
    // round-trip q through bf16 so the reference sees exactly what the kernel does
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = __bfloat162float(qb[i]);

    __nv_bfloat16 *dq, *dout;
    uint8_t *dd, *dsc;
    int *dbt, *didx, *dtl;
    CK(cudaMalloc(&dq, qb.size() * 2));
    CK(cudaMemcpy(dq, qb.data(), qb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dout, (size_t)B * H * VW * 2));
    CK(cudaMalloc(&dd, data.size()));
    CK(cudaMemcpy(dd, data.data(), data.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dsc, scl.size()));
    CK(cudaMemcpy(dsc, scl.data(), scl.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dbt, bt.size() * 4));
    CK(cudaMemcpy(dbt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&didx, idx.size() * 4));
    CK(cudaMemcpy(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dtl, tlen.size() * 4));
    CK(cudaMemcpy(dtl, tlen.data(), tlen.size() * 4, cudaMemcpyHostToDevice));

    tms::mla_decode_fp8_v<true, false, QW, VW, NFP8, SMODE><<<dim3(H, B), 32>>>(
        dq, dd, dsc, dbt, nullptr, didx, dtl, max_topk, dout, nullptr, nullptr, nullptr,
        block_size, bt_stride, scale, H, 1, 0, kv_scale);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());

    std::vector<__nv_bfloat16> ob((size_t)B * H * VW);
    CK(cudaMemcpy(ob.data(), dout, ob.size() * 2, cudaMemcpyDeviceToHost));

    // Partitioned launch + reduce (PART=true). Exercises the length-balanced
    // split: tlen (40..79) is far below P * partition_size, so every partition
    // must take a ceil(tlen/P) share for this to match the reference.
    const int P = 8;
    __nv_bfloat16* dout_p;
    float *dtmp, *dml, *des;
    CK(cudaMalloc(&dout_p, (size_t)B * H * VW * 2));
    CK(cudaMalloc(&dtmp, (size_t)B * H * P * VW * 4));
    CK(cudaMalloc(&dml, (size_t)B * H * P * 4));
    CK(cudaMalloc(&des, (size_t)B * H * P * 4));
    tms::mla_decode_fp8_v<true, true, QW, VW, NFP8, SMODE><<<dim3(H, B, P), 32>>>(
        dq, dd, dsc, dbt, nullptr, didx, dtl, max_topk, nullptr, dtmp, dml, des,
        block_size, bt_stride, scale, H, P, max_topk / P, kv_scale);
    tms::paged_attention_reduce<__nv_bfloat16, VW><<<dim3(H, B), 32>>>(
        dtmp, dml, des, dout_p, H, P);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<__nv_bfloat16> obp((size_t)B * H * VW);
    CK(cudaMemcpy(obp.data(), dout_p, obp.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<float> ref;
    host_ref<QW, VW, NFP8, SMODE>(hq, data, scl, bt, idx, tlen, B, H, block_size,
                                  bt_stride, scale, max_topk, kv_scale, ref);

    double worst = 0, mag = 0, worst_p = 0;
    for (size_t i = 0; i < ob.size(); ++i) {
        worst = std::fmax(worst, std::fabs(__bfloat162float(ob[i]) - ref[i]));
        worst_p = std::fmax(worst_p, std::fabs(__bfloat162float(obp[i]) - ref[i]));
        mag = std::fmax(mag, std::fabs((double)ref[i]));
    }
    const double rel = worst / (mag > 0 ? mag : 1.0);
    const double rel_p = worst_p / (mag > 0 ? mag : 1.0);
    const bool ok = rel < 0.02 && rel_p < 0.02;
    printf("  %-34s QW=%3d VW=%3d NFP8=%3d SMODE=%d  rel %.5f  part(P=%d) rel %.5f  %s\n",
           name, QW, VW, NFP8, SMODE, rel, P, rel_p, ok ? "PASS" : "FAIL");
    cudaFree(dq); cudaFree(dout); cudaFree(dd); cudaFree(dsc);
    cudaFree(dbt); cudaFree(didx); cudaFree(dtl);
    cudaFree(dout_p); cudaFree(dtmp); cudaFree(dml); cudaFree(des);
    return ok ? 0 : 1;
}

int main() {
    printf("templated sparse MLA fp8 decode -- geometry check\n");
    int bad = 0;
    bad |= run_case<576, 512, 576, 1>("GLM-5.2-Vision fp8 cache", 0.0221f);
    bad |= run_case<576, 512, 0, 0>("GLM-5.2-Vision bf16 cache", 1.0f);
    bad |= run_case<512, 512, 448, 0>("DeepSeek-style (original)", 1.0f);
    printf("%s\n", bad ? "FAILED" : "all geometries OK");
    return bad;
}
