# MLA decode kernel research: removing the 16-head floor

## 10-line summary

1. **Best starting point: `/home/hotaisle/QuixiCore/QuixiCore-ROCm/kernels/serving/mla_absorbed/variants/rocm_cdna3/mla_absorbed.cu:88` (`k_mla_absorbed`) combined with `.../variants/rocm_cdna3/mla_kernels.cuh:289` (`mla_decode_partition`).** The first is already gfx942/wave64-native and head-count agnostic; the second already has the paged block-table + split-K partition + `paged_attention_reduce<bf16,512>` structure you need.
2. Second choice: `/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh:1025` (`..._split_group8_partial_valid_kernel`) — best-tuned register/shared staging, but wave32-shaped and it *has* the exact bug class you're trying to escape.
3. **The 16-head constraint is NOT from MFMA tile shape in any fundamental sense — it comes from AITER shipping hand-written *pre-assembled* kernels** (`/home/hotaisle/aiter/hsa/gfx942/mla/*_QH16*.co`, `*gqaratio16*.co`) whose query-head count is baked into the binary; `aiter/mla.py:367` literally reshapes `(total_s*(nhead//16), 16, ...)` to fake other head counts, and `vllm/v1/attention/backends/mla/rocm_aiter_mla.py:643` hard-codes `_AITER_MIN_MLA_HEADS = 16`.
4. The *technical* seed of "16" is that 16 query heads = the M rows of `v_mfma_f32_16x16x16bf16_1k` / `v_mfma_f32_16x16x32_fp8_fp8` A-fragment (see the fragment map in `/home/hotaisle/SlimServe/csrc/quixicore/rocm/mfma_fp8_dot.cuh:14-18`). llama.cpp proves it's soft: `ggml/src/ggml-cuda/fattn.cu:232` picks `ncols2=16` when `gqa_ratio % 16 == 0` and silently falls back to `ncols2=4` otherwise — 12 heads works fine there at 3×4.
5. A softer, sneakier constraint also exists: cooperative shared-memory staging loops hard-coded to the full block width (`ds4_rocm_glm.cuh:1101` uses `off += 256u` after an early `return` for `head >= n_head`), which is why ds4 gates on `(n_head % 8u) != 0u` at `ds4_rocm_glm.cuh:3661`. Avoid that pattern.
6. **Design decision 1 — don't use MFMA at all for the decode QK/PV.** Arithmetic intensity at 12 heads is ~23 FLOP/byte (read 1152 B/token of 576-elem bf16 latent, do ~26 kFLOP) vs MI300X machine balance ~246 FLOP/byte. You are ~10× memory bound; MFMA buys nothing and costs you all the head-count rigidity. One **wave64 per head**, latent split 512 = 64 lanes × 8 floats (or 64 × `float4` ×2), `__shfl_xor` reduction for the score.
7. **Design decision 2 — grid `(num_heads, batch, num_kv_splits)` with one wave per block (or 12 waves = 1 block per request).** Head count becomes a pure grid dimension, so 12, 13, or 7 heads all work. Use the absorbed form: `score = <q_latent, latent[row]> + <q_pe, k_pe[row]>` over the 576-wide row, then accumulate only the 512 latent dims; `W_UV` unabsorb happens after the split-K reduce, once per head.
8. **Design decision 3 — keep the vLLM/Kimi-K3 cache layout unchanged**: `kv_c_and_k_pe_cache[block, block_size, 576]` = `[0:512)` latent, `[512:576)` RoPE'd k_pe, MQA (no head axis) — constants at `/home/hotaisle/vllm2/csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu:87-98`. RoPE is applied at *insert* time, so the decode kernel does zero trig; the nope/rope split is just a dot-product-range split within one contiguous 576-element row, giving fully coalesced 1152 B/token/wave reads.
9. Port hazard to watch: QuixiCore's `mla_kernels.cuh` family launches `<<<dim3(H,B),32>>>` and `tm_warp.cuh:16` reduces with `off=16..1` — that is a 32-lane reduction on a 64-lane gfx942 wave, i.e. **half the SIMD idle**. `mla_absorbed.cu:80` (`wave_sum`, `off=32..1`) is the correct wave64 form; take that one.
10. Expected outcome: a ~250-line HIP kernel + a 30-line reduce kernel, no head-count assert anywhere, bandwidth-bound at >80% of HBM peak, drop-in behind `forward_mqa` at `vllm/v1/attention/backends/mla/rocm_aiter_mla.py:911` replacing the `get_mla_padded_q`/`get_mla_unpadded_o` hack.

---

## 1. Scope

Target: Kimi K3 MLA decode on gfx942 (MI300X), TP8 → **12 query heads per rank**. Geometry (from `/home/hotaisle/vllm2/csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu:87-98`):

```
constexpr int kKvLoraRank    = 512;   // L   compressed latent
constexpr int kQkNopeHeadDim = 128;   // P
constexpr int kQkRopeHeadDim = 64;    // R
constexpr int kQkHeadDim     = 192;   // P+R
constexpr int kVHeadDim      = 128;   // V
constexpr int kCacheEntry    = 576;   // L+R  <- one paged KV row, MQA (no head axis)
```

---

## 2. `/home/hotaisle/ds4` — antirez's C engine (GLM-DSA / DeepSeek-style)

ds4 targets GLM-5.2 DSA, which is the same absorbed-MLA + sparse-indexer shape as DeepSeek-V3.2 / Kimi K3.

### 2.1 Reference scalar kernel (CUDA path)

`/home/hotaisle/ds4/ds4_cuda.cu:26530` — `glm_attention_lora_causal_kernel`:

> `/* Scalar-correct MLA attention: one warp per head, grid (ceil(n_head/8), n_tokens). ... mirrors the Metal online softmax so numerics stay comparable. */`

Guard at `ds4_cuda.cu:26558`: `kv_lora_dim != 512u || qk_rope != 64u` → return. Latent is held as 4× `float4` per lane (`low4[lane]`, `low4[lane+32]`, `low4[lane+64]`, `low4[lane+96]` = 32 lanes × 16 floats = 512).

### 2.2 Fused single-token indexed decode

`/home/hotaisle/ds4/ds4_cuda.cu:27100` — `glm_attention_indexed_decode_kernel`. Header comment states the math exactly:

> `Fuses score (qk_low . kv_lora + q_rope . rope(k_rope@row)), softmax over the indexer-selected rows, the weighted kv_lora sum, and the per-head value projection (q8_0).`

**One block per head** (`const uint32_t head = blockIdx.x;`), dynamic shared = `red[256] + scores[n_selected] + lora_sum[kv_lora_dim]`. **No head-count constraint at all.** The `W_UV` unabsorb is fused in the epilogue: one warp per output dim `d`, two lanes per q8_0 block of 32.

Note `k_rope_cache` here stores **un-rotated** k_pe; RoPE is applied *on read* via `glm_cache_rope_pair_f16_dev` (`ds4_cuda.cu:26515`) which recomputes `powf(freq_base, ...)` and YaRN corr factors per element. That's a lot of transcendental work in the inner loop — a design ds4 chose for cache-format simplicity and one you should **not** copy (vLLM/Kimi K3 rotates at insert time).

### 2.3 ROCm split-K decode

Three kernels in `/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh`:

| line | kernel | role |
|---|---|---|
| 892 | `glm_attention_indexed_decode_split_partial_kernel` | generic, 256 threads, `dim3(n_head, n_blocks)` — **no head constraint** |
| 1025 | `glm_attention_indexed_decode_split_group8_partial_valid_kernel` | fast path, `dim3(n_head/8, n_blocks)` × `dim3(32,8)` — **requires `n_head % 8 == 0`** |
| 1190 | `glm_attention_indexed_decode_split_reduce_kernel` | combines partials + q8_0 `W_UV` projection |

Split-K: `partial_lora[block][head][512]` + `partial_ms[block][head][2]` (running max `M` and sum `S`), classic flash-decoding rescale in the reducer.

The group8 kernel's tiling (`ds4_rocm_glm.cuh:1069-1075`):

```c
extern __shared__ float sh[];
float *kv_shared   = sh;                                    // 16 rows x 512
float *rope_shared = kv_shared + stage_rows * kv_lora_dim;  // 16 rows x 64
```

= 16 × 576 × 4 B = **36 KB LDS**, staged 16 KV rows at a time, shared by all 8 heads in the block. Q is register-resident: `low0..low3` (4× `float4` = 16 VGPRs) + `qrope` (lane<16 only). Inner loop is a scalar `glm_rocm_dot4` chain + `warp_sum_f32` + `__shfl(score,0,32)` broadcast, then an online-softmax rescale of `o0..o3`.

**This is where the head-count constraint bites.** The staging loops are:

```c
for (uint32_t off = tid; off < rows * kv_lora_dim; off += 256u) { ... }
for (uint32_t off = tid; off < rows * rope_pairs;  off += 256u) { ... }
```

`256u` is hard-coded to the *full* `32×8` block, but the kernel does an early `return` at `ds4_rocm_glm.cuh:1052` for `head >= n_head`. With `n_head = 12` the second block would have only 4 live warps (128 threads), so the stride-256 staging would leave half of `kv_shared` uninitialized *and* the surviving warps would hang or race on `__syncthreads()`. Hence the launcher gate:

`/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh:3661`:

```c
n_head == 0u || (n_head % 8u) != 0u ||
kv_lora_dim != 512u ||
qk_nope == 0u || qk_rope != 64u ||
```

and `ds4_rocm_glm.cuh:3690`: `<<<dim3(n_head / 8u, n_blocks, 1), dim3(32u, 8u, 1), partial_shmem>>>`.

**Lesson: this constraint is 100% software — it is a `blockDim`-hardcoded cooperative-load stride plus a divergent early-return, not an MFMA or wavefront property.** Fixing it needs only (a) `off += blockDim.x*blockDim.y`, and (b) masking rather than returning for `head >= n_head`. But note ds4 is wave32-shaped (`lane >= 32u` returns), so on gfx942 it wastes half of every 64-lane SIMD anyway.

### 2.4 ds4 indexer

`/home/hotaisle/ds4/rocm/ds4_rocm_indexer.cuh:128` `indexer_scores_wmma128_kernel` is the only matrix-core path in ds4's attention stack, and it's for the *sparse indexer* logits, not MLA decode. Nothing to reuse for 12-head decode.

---

## 3. `/home/hotaisle/QuixiCore/QuixiCore-ROCm` — the strongest reference

### 3.1 `kernels/serving/variants/rocm_cdna3/mla_kernels.cuh` (ThunderMittens port)

File header (`:3-9`) describes the set precisely: `mla_q_norm_rope` (GPT-J interleaved RoPE + optional RMSNorm), `mla_kv_insert` / `mla_kv_insert_fp8`, `mla_decode` (bf16 absorb path), `mla_decode_fp8`, `mla_decode_partition`, `mla_decode_fp8_v<SPARSE,PART>`.

**`mla_decode` — `mla_kernels.cuh:196`.** The cleanest statement of MLA decode I found anywhere in these trees:

```c
template <int LATENT, int ROPE>
__global__ void mla_decode(const bf16* q, const bf16* kv_cache, const int* block_table,
                           const int* context_lens, bf16* out,
                           int block_size, int bt_stride, float scale, int num_heads) {
    constexpr int QK = LATENT + ROPE, VQK = QK / 32, VAV = LATENT / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
```

Math: `qv[]` = the absorbed query (`q_nope @ W_UK` concat `q_pe`), 576 wide, held wholly in registers (`VQK = 18` floats/lane). Per KV token:

```c
const int64_t base = (int64_t(block) * block_size + slot) * QK;   // MQA: no head axis
float partial = 0.0f;
for (int i = 0; i < VQK; i++) partial += qv[i] * float(kv_cache[base + lane + 32 * i]);
const float score = warp_sum_f(partial) * scale;
...
for (int i = 0; i < VAV; i++) acc[i] = acc[i]*alpha + beta*float(kv_cache[base + lane + 32*i]);
```

**The nope/rope split is handled by simply not existing in the inner loop**: score runs over all 576, accumulate runs over the first 512. That is the entire trick. Because k_pe was already rotated at insert time (`mla_kv_insert`, `:134-142`), the decode kernel does zero trig.

**Paged layout expected:** `kv_cache[num_blocks][block_size][LATENT+ROPE]`, addressed as `(block*block_size + slot)*576`, with `block = block_table[batch*bt_stride + t/block_size]` and `block < 0` meaning "skip". Identical to vLLM's `kv_c_and_k_pe_cache`.

**Parallelization:** `<<<dim3(H, B), 32>>>` (`mla.cu:89`). One warp per (head, batch). **Zero head-count constraint** — `num_heads` is a plain grid dim.

**Split-K:** `mla_decode_partition` (`mla_kernels.cuh:289`) adds `blockIdx.z` = partition, writes `tmp_out / max_logits / exp_sums` at

```c
const int64_t stat = (int64_t(batch)*num_heads + head)*num_partitions + part;
```

and empty partitions emit `MLA_NEG_INF/0` so the reducer skips them. Combined by `paged_attention_reduce<bf16,512>` (`mla.cu:418`, `mla.cu:466`).

**FP8 variant:** `mla_decode_fp8` (`:237`) reads a packed 576 B data row + 8 B scale row: `[0,448)` NoPE e4m3 with per-64 UE8M0 scales, `[448,512)` RoPE bf16. Dequant-on-read inside the score loop (`:262-267`).

**Sparse (DSA / Kimi) variant:** `mla_decode_fp8_v<SPARSE,PART>` (`:345`) walks `indices[batch*max_topk + j]`, skipping `t < 0`, and can partition the *index list* rather than the token range.

**Porting hazard:** `warp_sum_f` (`tm_warp.cuh:16`) is `for (int off = 16; off > 0; off >>= 1) v += __shfl_xor(v, off);` — a 32-lane reduction. Every launch in `mla.cu` is `<<<..., 32>>>`. On gfx942 that means one 64-lane wave with 32 lanes permanently masked off. **Half the SIMD is idle.** Fix by widening to `off = 32` and `LATENT/64` per lane.

### 3.2 `kernels/serving/mla_absorbed/variants/rocm_cdna3/mla_absorbed.cu` — already gfx942-native

Header (`:5-13`): *"CDNA3 (gfx942) quantized MLA decode against an absorbed kv_b, dense and sparse (DSA) variants... This is the GLM-5.2 / DeepSeek-V3.2 decode shape."*

The math, verbatim from `:26-31`:

```
query_latent[l] = sum_d q[d] * W[head*(nope+value) + d, l]      d < nope_dim
score           = <query_latent, latent[row]> + <q[nope..], rope[row]>
mixed_latent    = softmax-weighted sum of latent[row]
out[v]          = sum_l W[head*(nope+value) + nope + v, l] * mixed_latent[l]
```

**This kernel does the absorb *in-kernel*** (`:118-125`) rather than expecting a pre-BMM'd query, then unabsorbs in the epilogue (`:172-179`). `W` is GGUF q8_0 row-packed (`:63-67`, `row_bytes = (latent_dim/32)*34`).

**Correct wave64 reduction** (`mla_absorbed.cu:80`):

```c
__device__ __forceinline__ float wave_sum(float v) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
    return v;
}
```

with the design note at `:100-103`: *"One 64-lane wavefront per block: wave_sum's xor reduction leaves the full sum in every lane, so the score needs no shared staging and every lane runs the identical online-softmax update in lockstep."*

**Parallelization:** `const int item = blockIdx.x; if (item >= batch*heads) return; const int request = item/heads, head = item%heads;` — a **flat 1-D grid over `batch*heads`**. This is the single most head-count-agnostic mapping in any of the four trees, and it's already the right shape for 12. Its test harness runs `heads = 4` (`:200`).

Weaknesses to fix in a production port: `double`-precision online softmax (matches a CPU reference bit-for-bit but is ~1/64 rate on MI300X), the asymmetric two-branch softmax update (documented at `:26-42`, deliberately non-standard), a `__syncthreads()` **per KV token** (`:167`), no split-K, and an O(latent × nope) in-kernel absorb that should be a batched GEMM outside the kernel.

### 3.3 MFMA reference on gfx942

`/home/hotaisle/SlimServe/csrc/quixicore/rocm/mfma_fp8_dot.cuh` is the authoritative fragment map for CDNA3 (identical copy at `QuixiCore-ROCm/kernels/serving/variants/rocm_cdna3/mfma_fp8_dot.cuh`):

```
A[M=16,K=32] : lane l, byte v in 0..7 -> A[m = l%16      ][k = 8*(l/16)+v]
B[K=32,N=16] : lane l, byte v in 0..7 -> B[k = 8*(l/16)+v][n = l%16      ]
D[M=16,N=16] : lane l, reg  v in 0..3 -> D[m = 4*(l/16)+v][n = l%16      ]
```

```c
__device__ __forceinline__ f32x4 mfma_16x16x32_fp8(long a, long b, f32x4 acc) {
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
}
```

`fp8_mqa_logits_kernel.cuh:71` shows how heads map onto that tile:

```c
constexpr int WH = NUM_HEADS / 32;   // waves over heads
constexpr int WN = NWAVE / WH;       // waves over kv
constexpr int H_TILES = 2;           // the intra-lane pair
constexpr int N_TILES = N_PER_WAVE / 16;
```

> *"A wave covers 32 heads: 16 lanes x the 2-head intra-lane pair."*

So the head axis lands on the MFMA `N`/`M` = **16** axis, doubled by an intra-lane pair → 32 heads per wave. **Here is the literal origin of the 16 (and 32) magic numbers.** Note also `load_frag_at_row(..., (NUM_HEADS/2)*j + 16*wh + hrev, k)` with `bitrev4` — the head→lane mapping is bit-reversed, which is exactly the sort of thing that makes generalizing an MFMA head axis to 12 painful.

---

## 4. `/home/hotaisle/llama.cpp` — proof that 16 is soft

llama.cpp routes MLA through the generic flash-attn MMA kernel with `DKQ=576, DV=512` and `K`/`V` aliasing the same buffer (`fattn-mma-f16.cuh:601`: *"For MLA K and V have the same data."* — the `V_is_K_view` template parameter).

`ggml/src/ggml-cuda/fattn.cu:182-236` is the DeepSeek dispatch:

```c
case 576: {
    GGML_ASSERT(V->ne[0] == 512);
    ...
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio == 20) { /* GLM 4.7 Flash — long special-case ladder */ }
    else if (gqa_ratio % 16 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
    } else {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
    }
}
```

`ncols2` = number of heads packed into one MMA column tile. Heads are folded into the tile's `J` axis and recovered by integer division (`fattn-mma-f16.cuh:702`, `:768`):

```c
const int j = ((threadIdx.y / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;
```

and the warp/column mapping at `:564`:

```c
constexpr int np = cols_per_warp > ncols ? nwarps : nwarps * cols_per_warp/ncols;
```

**The constraint is `ncols2 | gqa_ratio`, with supported `ncols2 ∈ {1,2,4,8,16,32}`.** For 12 heads llama.cpp picks `ncols2 = 4` and iterates 3 tiles — correct, just ~25% less matrix-core efficient than the 16 path. There is no assert, no failure, no padding hack. This is the single clearest demonstration that AITER's `% 16` is a packaging decision, not a hardware one.

On CDNA3 llama.cpp uses `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` (`ggml/src/ggml-cuda/mma.cuh:1286`), `..._16x16x16f16` (`:1051`, `:1250`), `..._i32_16x16x32_i8` (`:1312`), with matrix C in `DATA_LAYOUT_J_MAJOR` for CDNA (`mma.cuh:80`) — worth reading if you do decide to go MFMA.

---

## 5. `/home/hotaisle/vllm2/csrc` — no ROCm MLA decode

The only MLA *attention* kernel is Blackwell CUTLASS: `csrc/libtorch_stable/attention/mla/sm100_cutlass_mla_kernel.cu` (+ `cutlass_sm100_mla/`). Not portable to gfx942.

What *is* useful is the Kimi-K3 cache-side kernel, `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu` — it defines the exact cache contract your decode kernel must consume:

- **Decode path** (`:6-9`): *"runs after BMM1 (`q_nope x W_UK`) right before `forward_mqa`, concatenating `mqa_q = [ql_nope | q_pe]` and inserting the latent cache."* → **the absorb is already done outside the kernel**; your decode kernel receives a 576-wide query per head.
- **bf16 cache row**: `cache[slot(t)] = [kv_c_normed[t] | k_pe[t]]`, 576 elements, RoPE already applied (`:17-19`).
- **fp8_ds_mla cache row** (`:100-102`): *"656B entry: `[0,512)` NoPE fp8 (4 tiles of 128), `[512,528)` 4 fp32 tile scales, `[528,656)` RoPE 64 bf16"* — bit-compatible with `concat_and_cache_ds_mla_kernel`.
- gfx942 fp8 note (`:96-100`): `kFp8Max = 224.0f` on `__gfx942__` (fnuz e4m3), vs 448 elsewhere. Do not get this wrong.
- Block structure (`:35-39`): *"one warp per (token, slot) with `slotsPerToken = num_heads + 1`"* — again, `num_heads` is a plain runtime value.

---

## 6. Where the 16-head constraint actually comes from

Three layers, in decreasing hardness:

**(a) Packaging — the real reason (hard, but only for AITER).**
AITER's MLA decode on gfx942 is not HIP at all; it is hand-written assembly shipped as pre-linked code objects:

```
/home/hotaisle/aiter/hsa/gfx942/mla/mla_a16w16_qh16_m16x4_n16x1_coex0_mask1.co
/home/hotaisle/aiter/hsa/gfx942/mla/MLA_A16W16_1TG_4W_64mx1_16nx1_Coex0_Msk1_QH16.co
/home/hotaisle/aiter/hsa/gfx942/mla/mla_a8w8_qh16_qseqlen4_gqaratio16.co
/home/hotaisle/aiter/hsa/gfx942/mla/mla_a8w8_qh128_m32x4_n16x2_msk0_ps.co
```

`QH16` / `qh16` / `gqaratio16` are baked into the binary. `aiter/mla.py:323-367` is a giant `if nhead == 16 or (gfx942 and nhead == 128 and fp8) or ...` ladder of "natively supported cases", followed by:

```python
elif nhead in range(32, 128 + 1, 16) and persistent_mode:
    # we use nhead=16 to simulate such cases by customized metadata
    # metadata also views qo's tensor as shape (total_s * (nhead // 16), 16, ...)
```

i.e. every non-16 head count is emulated by *reshaping the query so it looks like 16 heads*. 12 is not expressible that way. vLLM then encodes the whole mess as a helper:

`/home/hotaisle/SlimServe/vllm/v1/attention/backends/mla/rocm_aiter_mla.py:640-682`:

```python
class AiterMLAHelper:
    """AITER MLA implementation requires num_heads >= 16. If num_heads < 16 and
    16 % num_heads == 0, we can pad q to 16 heads; otherwise AITER has to fail."""
    _AITER_MIN_MLA_HEADS: Final = 16
    ...
    return (num_heads % 16 == 0 if num_heads >= 16 else 16 % num_heads == 0)
```

with `get_mla_padded_q` doing `q.repeat_interleave(16 // num_heads, dim=1)` and `get_mla_unpadded_o` slicing back. 12 fails both branches: `12 % 16 != 0` and `16 % 12 != 0`. Error message at `:649`: *"Try adjusting tensor_parallel_size value."*

**(b) MFMA tile shape (real, but costs ≤25%, not correctness).**
CDNA3 f16/bf16 matrix cores are `16x16x16` and `32x32x8`; fp8 is `16x16x32`. If you put the query-head axis on the MFMA M or N axis, the natural granularity is 16. With 12 heads you pad to 16 → 25% of the QK^T MACs wasted. But QK^T is a tiny fraction of the decode's work, and llama.cpp's `ncols2=4` fallback (`fattn.cu:232`) shows you can instead tile 12 as 3×4. There is also `v_mfma_f32_4x4x4f16` (16 blocks) if you truly want M=4. **This layer is soft.**

**(c) Wavefront/block mapping (entirely soft, but the most common bug source).**
The ds4 `% 8` gate (`ds4_rocm_glm.cuh:3661`) is caused by `off += 256u` cooperative-staging loops (`:1095`, `:1104`) combined with an early `return` for out-of-range heads (`:1052`). Nothing about the hardware requires it. Fix with `blockDim.x*blockDim.y` strides and predication instead of early return, or just don't group heads into a block at all.

**Crucially: none of the three layers is a memory-layout constraint.** The paged KV cache is MQA — a single 576-element row shared by all heads — so the head count never appears in any address calculation.

---

## 7. The arithmetic-intensity argument (why you can ignore MFMA entirely)

Per KV token, per request, at 12 heads:

- Bytes read: 576 × 2 B (bf16) = **1152 B** (or 656 B for fp8_ds_mla, or 576 B for plain fp8).
- FLOPs: 12 heads × (576 MAC for the score + 512 MAC for the PV accumulate) × 2 = **26,112 FLOP**.
- Intensity ≈ **22.7 FLOP/byte** (bf16), ~40 FLOP/byte (fp8).

MI300X balance point: ~1307 TFLOP/s bf16 MFMA ÷ 5.3 TB/s ≈ **246 FLOP/byte**.

You are ~10× below the balance point. Even at TP1 (128 heads) MLA decode only reaches ~240 FLOP/byte, i.e. *exactly* at balance — which is precisely why AITER bothered with MFMA for the `qh128` case and why it doesn't matter for `qh12`. With MTP/spec-decode `q_len=4` you'd reach ~90 FLOP/byte; still memory bound.

**Conclusion: at 12 heads/rank, a well-written vector kernel that saturates HBM is within a few percent of what any MFMA kernel could achieve.** Spend the effort on load coalescing, `buffer_load` widths, split-K balance and occupancy — not on matrix cores.

---

## 8. Proposed design: `mla_decode_hip_gfx942`

### 8.1 Signature and contract

```
q_absorbed : [B, H, 576] bf16   // [q_nope @ W_UK | q_pe], produced by existing BMM1
kv_cache   : [num_blocks, block_size, 576] bf16   // [kv_c_normed | rope(k_pe)], MQA
block_table: [B, max_blocks] int32
seq_lens   : [B] int32
out        : [B, H, 512] bf16   // latent; W_UV unabsorb done by existing BMM2
```

Identical to what `forward_mqa` (`rocm_aiter_mla.py:911`) already builds, minus the padding hack. Keep `AiterMLAHelper` for the AITER backend; add a new backend class that skips it.

### 8.2 Stage 1 kernel — partials

- **Grid:** `dim3(H, B, num_kv_splits)`. `H = 12` is a grid dim; no constraint whatsoever. (Alternative: flat `blockIdx.x` over `B*H` as in `mla_absorbed.cu:105-108`, better for tail-effect balancing when `B*H` is not a multiple of 304 CUs.)
- **Block:** 64 threads = **one wave64**. Optionally 256 threads = 4 waves, one head each, if you want `H=12 → 3 blocks/request` and LDS-shared KV staging — but then use `blockDim.x` strides, never a literal.
- **Registers:** `qv[9]` = 576 / 64 lanes = 9 floats/lane (or 4×`bf16x2` vectors + 1). `acc[8]` = 512 / 64. Total ~17 VGPRs of live state — trivially fits, allowing 8 waves/SIMD occupancy.
- **Inner loop** (adapted from `mla_kernels.cuh:212-229`, widened to wave64):

```c
const int64_t base = (int64_t(block)*block_size + slot) * 576;
float partial = 0.f;
#pragma unroll
for (int i = 0; i < 9; i++) partial += qv[i] * float(kv_cache[base + lane + 64*i]);
const float score = wave_sum(partial) * scale;          // off = 32..1
const float nm = fmaxf(m, score);
const float alpha = expf(m - nm), beta = expf(score - nm);
#pragma unroll
for (int i = 0; i < 8; i++) acc[i] = acc[i]*alpha + beta*float(kv_cache[base + lane + 64*i]);
l = l*alpha + beta;  m = nm;
```

Note the same load serves both the score and the accumulate for `i < 8` — hoist it into a `float kvv[9]` so each 576-element row is read from HBM exactly once (ds4's group8 kernel re-loads from LDS, QuixiCore's `mla_decode` re-loads from L1/L2; on gfx942 a `kvv[]` register stage is strictly better).

- **Vectorization:** at `lane + 64*i` with bf16, adjacent lanes are 2 B apart → a 128 B per-wave transaction per `i`, 9 of them = 1152 B, perfectly coalesced. If you prefer `dwordx4` per lane, use `lane*4` indexing with 512/(64*4)=2 iterations for the accumulate and a 576/(64*4)=2.25 awkwardness for the score — the `lane + 64*i` stride form is cleaner and equally fast on CDNA3's coalescer.
- **Output:** `tmp_out[((b*H + h)*S + s)*512 + ...]`, `max_logits[...]`, `exp_sums[...]` exactly as `mla_decode_partition` (`mla_kernels.cuh:328-336`). Empty splits emit `-FLT_MAX / 0`.

### 8.3 Stage 2 kernel — reduce

Reuse the structure of `paged_attention_reduce<bf16, 512>` (`paged_attn_v2_kernels.cuh`, launched at `mla.cu:418`): `dim3(H, B)` × 64 threads, global max over splits, rescale, weighted sum, divide. ~30 lines.

### 8.4 Split-K sizing

`num_kv_splits = clamp(ceil(304 / (B*H)), 1, ...)` targeting ≥1 wave per CU. At `B=1, H=12` you want ~25 splits to fill MI300X; at `B=64, H=12` (768 blocks) one split suffices. AITER's `get_meta_param` (`aiter/mla.py:109-153`) is a reasonable model for the heuristic; ds4 caps `n_blocks <= 64` (`ds4_rocm_glm.cuh:3665`).

### 8.5 If you later want MFMA (for MTP / large `q_len`)

Put the **KV-token axis on M and N**, not the head axis. Tile 12 heads as 3×`ncols2=4` like llama.cpp, or pad Q to 16 rows and mask — the padding costs 25% of a component that is <10% of runtime. Use `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` with the fragment layout documented in `mfma_fp8_dot.cuh:14-18`, and remember the `bitrev4` head-permutation gotcha from `fp8_mqa_logits_kernel.cuh:88-90` if you copy that fragment loader.

### 8.6 Correctness plan

1. Port `mla_absorbed.cu`'s self-checking harness (`:189+`, fp64 host replica) — it already runs `heads = 4`, so bump to 12.
2. Cross-check against `mla_decode<512,64>` at `mla.cu:89` (tolerance `6e-3`, already validated).
3. Then A/B against AITER at `num_heads=16` to confirm the new kernel matches the shipped path before trusting it at 12.

---

## 9. File index

| Path | What it is |
|---|---|
| `/home/hotaisle/QuixiCore/QuixiCore-ROCm/kernels/serving/mla_absorbed/variants/rocm_cdna3/mla_absorbed.cu:88` | **gfx942 wave64 absorbed MLA decode, head-agnostic — primary starting point** |
| `/home/hotaisle/QuixiCore/QuixiCore-ROCm/kernels/serving/variants/rocm_cdna3/mla_kernels.cuh:196` | `mla_decode` — cleanest absorbed-path math |
| `…/mla_kernels.cuh:289` | `mla_decode_partition` — split-K structure to copy |
| `…/mla_kernels.cuh:345` | `mla_decode_fp8_v<SPARSE,PART>` — sparse/DSA + partition |
| `…/mla_kernels.cuh:102`, `:148` | KV insert (bf16 / packed fp8) — defines cache layout |
| `…/variants/rocm_cdna3/tm_warp.cuh:16` | `warp_sum_f` — **32-lane, must widen for gfx942** |
| `…/mla_absorbed.cu:80` | `wave_sum` — correct 64-lane form |
| `…/variants/rocm_cdna3/mla.cu:89,183,416,464` | launch configs + validation harness |
| `/home/hotaisle/SlimServe/csrc/quixicore/rocm/mfma_fp8_dot.cuh:14` | CDNA3 MFMA fragment layout (origin of "16") |
| `/home/hotaisle/SlimServe/csrc/quixicore/rocm/fp8_mqa_logits_kernel.cuh:71` | head-axis-on-MFMA example, incl. `bitrev4` gotcha |
| `/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh:1025` | ds4 group8 fast decode (best LDS staging) |
| `/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh:3661` | **`(n_head % 8u) != 0u` gate — the soft-constraint case study** |
| `/home/hotaisle/ds4/rocm/ds4_rocm_glm.cuh:892`, `:1190` | ds4 generic split partial + reduce |
| `/home/hotaisle/ds4/ds4_cuda.cu:27100` | ds4 fused indexed MLA decode (one block/head, no constraint) |
| `/home/hotaisle/llama.cpp/ggml/src/ggml-cuda/fattn.cu:182` | 576/512 MLA dispatch; `:232` `gqa_ratio % 16 == 0` → `ncols2` 16 vs 4 |
| `/home/hotaisle/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh:564`, `:702` | head→MMA-tile-column packing |
| `/home/hotaisle/llama.cpp/ggml/src/ggml-cuda/mma.cuh:1286` | CDNA3 `mfma_f32_16x16x16bf16_1k` |
| `/home/hotaisle/vllm2/csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu:87` | Kimi K3 geometry + cache formats (bf16 / fp8 / fp8_ds_mla 656B) |
| `/home/hotaisle/SlimServe/vllm/v1/attention/backends/mla/rocm_aiter_mla.py:640` | `AiterMLAHelper` — the `%16` assert to delete |
| `/home/hotaisle/SlimServe/vllm/v1/attention/backends/mla/rocm_aiter_mla.py:911` | `forward_mqa` — integration point |
| `/home/hotaisle/aiter/aiter/mla.py:323-400` | AITER's nhead dispatch ladder + nhead-16 emulation |
| `/home/hotaisle/aiter/hsa/gfx942/mla/*.co` | pre-assembled `QH16` binaries — the actual root cause |
