// SPDX-License-Identifier: Apache-2.0
// Isolated SM120 mHC prefill experiment; never registered as a serving op.
// Same BF16 values and fused residual expression, but tensor-core FP32 dot
// accumulation changes summation order. Requires an independent oracle.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "mhc_ampere.cuh"
#include "bf16_decode_gemm.cuh"

namespace mhc_tc_probe {
using namespace tms::dsv4_mhc;
using namespace tms::decode_gemm;
using torch::Tensor;
constexpr int H = 4096, FLATS = 512, ROW = 520, TILE = 32;
constexpr int BYTES = (MIXES + TILE) * ROW * sizeof(__nv_bfloat16);
static_assert(HC == 4 && SPLITS == 32 && MIXES == 24);
static_assert(ROW % 8 == 0 && BYTES < 64 * 1024);

template <bool FUSED>
__global__ void __launch_bounds__(256) partials_tc(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    const __nv_bfloat16* __restrict__ fn,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ partial, int tokens) {
    extern __shared__ __align__(16) unsigned char raw[];
    auto* ft = reinterpret_cast<__nv_bfloat16*>(raw);
    auto* vt = ft + MIXES * ROW;
    const int split = blockIdx.x, first = blockIdx.y * TILE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int dim_split = split * 128;

    // Stage the original four stream slices without a persistent repack.
    for (int vec = tid; vec < MIXES * FLATS / 8; vec += 256) {
        const int output = vec / (FLATS / 8);
        const int flat = (vec % (FLATS / 8)) * 8;
        const int stream = flat / 128, dim = dim_split + flat % 128;
        cp_async16(smem_u32(ft + output * ROW + flat),
                   fn + output * (HC * H) + stream * H + dim, true);
    }
    cp_async_commit();

    // The same vectorized fused post-mix expression/order as partials_prefill.
    // Eight threads per token, two 8-wide chunks per thread/stream. Invalid
    // token rows are explicitly zeroed because warp MMA reads full tiles.
    const int t = tid >> 3, g = tid & 7, token = first + t;
    auto* vr = vt + t * ROW;
    if (token < tokens) {
        if constexpr (FUSED) {
            float pm[HC], cm[HC * HC];
#pragma unroll
            for (int i = 0; i < HC; ++i) pm[i] = post[token * HC + i];
#pragma unroll
            for (int i = 0; i < HC * HC; ++i) cm[i] = comb[token * HC * HC + i];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int dim = dim_split + g * 16 + c * 8;
                const uint4 xv = *reinterpret_cast<const uint4*>(x + size_t(token) * H + dim);
                uint4 rv[HC];
#pragma unroll
                for (int i = 0; i < HC; ++i)
                    rv[i] = *reinterpret_cast<const uint4*>(
                        residual + (size_t(token) * HC + i) * H + dim);
                const auto* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                for (int out_stream = 0; out_stream < HC; ++out_stream) {
                    __align__(16) __nv_bfloat16 rounded[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float value = pm[out_stream] * float(xb[j]);
#pragma unroll
                        for (int in_stream = 0; in_stream < HC; ++in_stream) {
                            const auto* rb = reinterpret_cast<const __nv_bfloat16*>(&rv[in_stream]);
                            value += cm[in_stream * HC + out_stream] * float(rb[j]);
                        }
                        rounded[j] = __float2bfloat16_rn(value);
                    }
                    *reinterpret_cast<uint4*>(
                        residual_out + (size_t(token) * HC + out_stream) * H + dim) =
                        *reinterpret_cast<const uint4*>(rounded);
                    *reinterpret_cast<uint4*>(vr + out_stream * 128 + g * 16 + c * 8) =
                        *reinterpret_cast<const uint4*>(rounded);
                }
            }
        } else {
#pragma unroll
            for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int dim = dim_split + g * 16 + c * 8;
                    *reinterpret_cast<uint4*>(vr + stream * 128 + g * 16 + c * 8) =
                        *reinterpret_cast<const uint4*>(
                            residual + (size_t(token) * HC + stream) * H + dim);
                }
            }
        }
    } else {
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                *reinterpret_cast<uint4*>(vr + stream * 128 + g * 16 + c * 8) =
                    make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Six warps cover 2 token tiles x 3 mix tiles. Use the proven BF16
    // decode-GEMM fragment mapping; 520-element rows align every ldmatrix
    // address to 16 bytes. The seventh warp preserves the old square-sum
    // order; no BF16-rounded norm or approximate reduction is introduced.
    if (warp < 6) {
        const int m0 = (warp / 3) * 16, n0 = (warp % 3) * 8;
        const int mat = lane >> 3, row = lane & 7;
        float acc[4] = {};
#pragma unroll 4
        for (int k = 0; k < FLATS; k += 16) {
            uint32_t a[4], b[2];
            ldmatrix_x4(a, smem_u32(vt + (m0 + row + 8 * (mat & 1)) * ROW + k + 8 * (mat >> 1)));
            ldmatrix_x2(b, smem_u32(ft + (n0 + row) * ROW + k + 8 * (mat & 1)));
            mma_bf16_16816(acc, a, b[0], b[1]);
        }
        const int r0 = first + m0 + (lane >> 2), col = n0 + (lane & 3) * 2;
        if (r0 < tokens) {
            float* dst = partial + (size_t(r0) * SPLITS + split) * (MIXES + 1);
            dst[col] = acc[0];
            dst[col + 1] = acc[1];
        }
        if (r0 + 8 < tokens) {
            float* dst = partial + (size_t(r0 + 8) * SPLITS + split) * (MIXES + 1);
            dst[col] = acc[2];
            dst[col + 1] = acc[3];
        }
    } else if (warp == 6) {
        float sum = 0.0f;
        const auto* src = vt + lane * ROW;
#pragma unroll 8
        for (int k = 0; k < FLATS; k += 2) {
            const float2 v = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(src + k));
            sum += v.x * v.x;
            sum += v.y * v.y;
        }
        if (first + lane < tokens)
            partial[(size_t(first + lane) * SPLITS + split) * (MIXES + 1) + MIXES] = sum;
    }
}

template <bool FUSED>
static std::vector<Tensor> run_typed(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base, bool partial_only) {
    const int tokens = residual.size(0);
    auto fopt = fn.options().dtype(torch::kFloat32);
    auto ro = FUSED ? torch::empty_like(residual) : residual;
    auto partial = torch::empty({tokens, SPLITS, MIXES + 1}, fopt);
    auto bp = [](Tensor t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); };
    const auto stream = at::cuda::getCurrentCUDAStream();
    static thread_local int configured_device = -1;
    if (configured_device != residual.get_device()) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(partials_tc<FUSED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, BYTES));
        configured_device = residual.get_device();
    }
    partials_tc<FUSED><<<dim3(SPLITS, (tokens + TILE - 1) / TILE), 256, BYTES, stream>>>(
        bp(x), bp(residual), post.data_ptr<float>(), comb.data_ptr<float>(),
        bp(fn), bp(ro), partial.data_ptr<float>(), tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (partial_only) return {ro, partial};
    auto next_post = torch::empty({tokens, HC}, fopt);
    auto next_comb = torch::empty({tokens, HC, HC}, fopt);
    auto out = torch::empty({tokens, H}, residual.options());
    finalize_pre_mix<<<tokens, 32, 0, stream>>>(
        partial.data_ptr<float>(), scale.data_ptr<float>(), base.data_ptr<float>(),
        next_post.data_ptr<float>(), next_comb.data_ptr<float>(),
        H, 1e-5f, 1e-6f, 1e-6f, 2.0f, 20);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    apply_pre_mix<MIXES + 1><<<dim3(H / 256, tokens), 256, 0, stream>>>(
        partial.data_ptr<float>(), bp(ro), bp(out), H);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ro, next_post, next_comb, out};
}

static std::vector<Tensor> run(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base, bool fused, bool partial_only) {
    TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 &&
        residual.size(0) >= 64 && residual.size(0) <= 7616 &&
        residual.size(1) == HC && residual.size(2) == H,
        "expected residual [64..7616,4,4096]");
    const c10::cuda::CUDAGuard guard(residual.device());
    const auto* device = at::cuda::getDeviceProperties(residual.get_device());
    TORCH_CHECK(device->major == 12 && device->minor == 0, "SM120 probe only");
    for (const auto& t : {x, residual, post, comb, fn, scale, base})
        TORCH_CHECK(t.device() == residual.device() && t.is_contiguous(),
                    "all inputs must be contiguous on the same device");
    for (const auto& t : {x, residual, fn})
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16 &&
                    reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                    "expected 16-byte-aligned BF16 activations/fn");
    for (const auto& t : {post, comb, scale, base})
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "expected FP32 parameters");
    const int tokens = residual.size(0);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({tokens, H}) &&
        post.sizes() == at::IntArrayRef({tokens, HC}) &&
        comb.sizes() == at::IntArrayRef({tokens, HC, HC}) &&
        fn.sizes() == at::IntArrayRef({MIXES, HC * H}) &&
        scale.numel() == 3 && base.numel() == MIXES, "invalid probe shapes");
    if (fused) return run_typed<true>(x, residual, post, comb, fn, scale, base, partial_only);
    return run_typed<false>(x, residual, post, comb, fn, scale, base, partial_only);
}

static pybind11::list resources() {
    pybind11::list result;
    for (bool fused : {false, true}) {
        cudaFuncAttributes attr{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,
            fused ? partials_tc<true> : partials_tc<false>));
        pybind11::dict row;
        row["fused"] = fused;
        row["registers"] = attr.numRegs;
        row["local_bytes"] = attr.localSizeBytes;
        row["dynamic_shared_bytes"] = BYTES;
        result.append(row);
    }
    return result;
}
}  // namespace mhc_tc_probe

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &mhc_tc_probe::run, pybind11::arg("x"), pybind11::arg("residual"),
        pybind11::arg("post"), pybind11::arg("comb"), pybind11::arg("fn"),
        pybind11::arg("scale"), pybind11::arg("base"), pybind11::arg("fused"),
        pybind11::arg("partial_only") = false);
    m.def("resources", &mhc_tc_probe::resources);
}
