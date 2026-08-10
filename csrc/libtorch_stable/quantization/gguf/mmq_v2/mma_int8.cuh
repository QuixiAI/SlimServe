// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Adapted from llama.cpp ggml/src/ggml-cuda/mma.cuh at commit
// 9b2a088819cda774bdbf713168ee1eee8498cda5. Reduced to the int8 tile shapes and
// the sm80 mma/ldmatrix instructions used by the Q8_0 MMQ path.

#pragma once

#ifndef USE_ROCM

  #include <cuda_runtime.h>

namespace vllm_mmq_v2 {

// Fragment of an int32 matrix distributed over one warp. The (i, j) element of
// logical tile <I, J> lives in lane get_i(l)/get_j(l) register x[l], matching
// the PTX mma.sync operand layout.
template <int I_, int J_>
struct tile {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / 32;
  int x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return ((l / 2) * 8) + (threadIdx.x / 4);
    } else {
      return -1;
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 8) {
      return (l * 4) + (threadIdx.x % 4);
    } else if constexpr (I == 16 && J == 8) {
      return ((threadIdx.x % 4) * 2) + (l % 2);
    } else {
      return -1;
    }
  }
};

template <int I, int J>
static __device__ __forceinline__ void load_generic(tile<I, J>& t,
                                                    const int* __restrict__ xs0,
                                                    const int stride) {
  #pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

// Requires a 16 byte aligned shared memory address; the tile strides are padded
// to guarantee it.
static __device__ __forceinline__ void load_ldmatrix(
    tile<16, 8>& t, const int* __restrict__ xs0, const int stride) {
  int* xi = (int*)t.x;
  const int* xs = xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * 4;
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
}

static __device__ __forceinline__ void mma(tile<16, 8>& D, const tile<16, 8>& A,
                                           const tile<8, 8>& B) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]),
        "r"(B.x[1]));
}

// K=16 (4-int) fragments for quants whose scales change every 16 values
// (Q2_K per-16 scale/min pairs): A lane l holds ints at
// (row = l*8 + tx/4, k-int = tx%4); B at (col = tx/4, k-int = tx%4).
template <>
struct tile<16, 4> {
  static constexpr int I = 16;
  static constexpr int J = 4;
  static constexpr int ne = 2;
  int x[ne] = {0};
  static __device__ __forceinline__ int get_i(const int l) {
    return l * 8 + threadIdx.x / 4;
  }
  static __device__ __forceinline__ int get_j(const int) {
    return threadIdx.x % 4;
  }
};

template <>
struct tile<8, 4> {
  static constexpr int I = 8;
  static constexpr int J = 4;
  static constexpr int ne = 1;
  int x[ne] = {0};
  static __device__ __forceinline__ int get_i(const int) {
    return threadIdx.x / 4;
  }
  static __device__ __forceinline__ int get_j(const int) {
    return threadIdx.x % 4;
  }
};

static __device__ __forceinline__ void mma(tile<16, 8>& D,
                                           const tile<16, 4>& A,
                                           const tile<8, 4>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
}

}  // namespace vllm_mmq_v2

#endif  // USE_ROCM
