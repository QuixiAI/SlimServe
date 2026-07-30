// tk_cuda MF-M6 bindings: random-sign FWHT rotation, GPTQ column permute,
// per-LoRA MoE align (kernels/quant/turboquant.cuh). Registered by init_m6(m).
#include "../quant/turboquant.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;
using namespace tm6;

#define M6CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t m6st() { return at::cuda::getCurrentCUDAStream(); }

#define DISPATCH_M6(t, ...) do {                                                  \
    if (t.scalar_type() == torch::kFloat)        { using scalar_t = float;          __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kHalf)    { using scalar_t = __half;         __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kBFloat16){ using scalar_t = __nv_bfloat16;  __VA_ARGS__ } \
    else TORCH_CHECK(false, "want fp32/fp16/bf16"); } while (0)
template <typename S> static const S* c6(const torch::Tensor& t) { return reinterpret_cast<const S*>(t.data_ptr()); }
template <typename S> static S* m6(torch::Tensor& t) { return reinterpret_cast<S*>(t.data_ptr()); }

// Templated launcher (a #define inside a macro arg won't expand — use a helper).
template <typename S, int D>
static void launch_fwht(const torch::Tensor& x, torch::Tensor& out,
                        const torch::Tensor& sign, int rows, bool inverse, cudaStream_t s) {
    if (inverse)
        fwht_rotate<S, D, true><<<rows, 32, 0, s>>>(c6<S>(x), m6<S>(out), sign.data_ptr<float>(), rows);
    else
        fwht_rotate<S, D, false><<<rows, 32, 0, s>>>(c6<S>(x), m6<S>(out), sign.data_ptr<float>(), rows);
}

// Random-sign FWHT rotation (rows, D). sign (D,) fp32 +-1. inverse toggles
// whether the sign is applied before (forward) or after (inverse).
static torch::Tensor py_fwht_rotate(torch::Tensor x, torch::Tensor sign, bool inverse) {
    M6CK(x); M6CK(sign);
    const int rows = x.numel() / x.size(-1), D = x.size(-1);
    auto out = torch::empty_like(x);
    auto s = m6st();
    DISPATCH_M6(x, {
        if (D == 64) launch_fwht<scalar_t, 64>(x, out, sign, rows, inverse, s);
        else if (D == 128) launch_fwht<scalar_t, 128>(x, out, sign, rows, inverse, s);
        else if (D == 256) launch_fwht<scalar_t, 256>(x, out, sign, rows, inverse, s);
        else if (D == 512) launch_fwht<scalar_t, 512>(x, out, sign, rows, inverse, s);
        else TORCH_CHECK(false, "D must be 64/128/256/512");
    });
    return out;
}

// GPTQ/act-order column permute: out[r,c] = in[r, perm[c]]. 16-bit dtypes.
static torch::Tensor py_permute_cols(torch::Tensor input, torch::Tensor perm) {
    M6CK(input); M6CK(perm);
    const int rows = input.size(0), cols = input.size(1);
    auto out = torch::empty_like(input);
    dim3 grid((cols + 255) / 256, rows);
    DISPATCH_M6(input, permute_cols<scalar_t><<<grid, 256, 0, m6st()>>>(
        c6<scalar_t>(input), m6<scalar_t>(out), perm.data_ptr<int>(), rows, cols););
    return out;
}

// Per-LoRA MoE align -> (sorted_token_ids, expert_ids, num_tokens_post_pad),
// all sized per max_loras.
static std::vector<torch::Tensor> py_moe_lora_align(torch::Tensor topk_ids,
        torch::Tensor token_lora_mapping, torch::Tensor lora_ids, torch::Tensor adapter_enabled,
        int64_t num_experts, int64_t topk, int64_t block_size, int64_t sorted_capacity,
        int64_t expert_capacity) {
    M6CK(topk_ids); M6CK(token_lora_mapping); M6CK(lora_ids); M6CK(adapter_enabled);
    const int max_loras = lora_ids.numel(), assignments = topk_ids.numel();
    auto opt = topk_ids.options();
    auto sorted = torch::zeros({max_loras * sorted_capacity}, opt);
    auto expert = torch::zeros({max_loras * expert_capacity}, opt);
    auto ntp = torch::zeros({max_loras}, opt);
    moe_lora_align<<<max_loras, 128, 0, m6st()>>>(topk_ids.data_ptr<int>(),
        token_lora_mapping.data_ptr<int>(), lora_ids.data_ptr<int>(),
        adapter_enabled.data_ptr<uint8_t>(), sorted.data_ptr<int>(), expert.data_ptr<int>(),
        ntp.data_ptr<int>(), max_loras, int(num_experts), assignments, int(topk),
        int(block_size), int(sorted_capacity), int(expert_capacity));
    return {sorted, expert, ntp};
}

void init_m6(py::module_& m) {
    m.def("fwht_rotate", &py_fwht_rotate, py::arg("x"), py::arg("sign"), py::arg("inverse") = false);
    m.def("permute_cols", &py_permute_cols);
    m.def("moe_lora_align", &py_moe_lora_align, py::arg("topk_ids"), py::arg("token_lora_mapping"),
          py::arg("lora_ids"), py::arg("adapter_enabled"), py::arg("num_experts"),
          py::arg("topk"), py::arg("block_size"), py::arg("sorted_capacity"),
          py::arg("expert_capacity"));
}
