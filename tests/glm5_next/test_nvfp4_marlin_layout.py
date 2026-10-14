# SPDX-License-Identifier: Apache-2.0
"""Address/scale contracts for a future cache-neutral NVFP4 decode kernel.

CPU enumeration mirrors gptq_marlin_repack.cu's non-act-order, 4-bit,
16-bit-activation tile. Native GPU repack comparison is a separate gate.
No dequantized serving cache or replacement GEMM is implemented here.
"""

import numpy as np
import pytest
import torch


def addresses(k, n, size_n):
    word = ((k // 16) * (size_n // 64) + n // 64) * 128
    word = word + (n % 8) * 16 + ((k % 8) // 2) * 4 + (n % 64) // 16
    shift = 4 * (4 * (k % 2) + 2 * ((n % 16) // 8) + (k % 16) // 8)
    return word, shift


def enumerate_repack(codes):
    size_k, size_n = codes.shape
    output = np.empty(size_k * size_n // 8, dtype=np.uint32)
    permutation = (0, 2, 4, 6, 1, 3, 5, 7)
    for kt in range(size_k // 16):
        for nt in range(size_n // 64):
            offset = (kt * (size_n // 64) + nt) * 128
            for warp in range(4):
                for lane in range(32):
                    col, row = lane // 4, lane % 4 * 2
                    values = [
                        int(codes[kt * 16 + row + dk, nt * 64 + warp * 16 + col + dn])
                        for dn in (0, 8)
                        for dk in (0, 1, 8, 9)
                    ]
                    output[offset + lane * 4 + warp] = sum(
                        values[p] << (4 * i) for i, p in enumerate(permutation)
                    )
    return output


@pytest.mark.parametrize("size_k,size_n", [(16, 64), (32, 128), (256, 512)])
def test_inverse_address_against_cpp_tile_enumeration(size_k, size_n):
    codes = np.random.default_rng(19).integers(0, 16, (size_k, size_n), dtype=np.uint8)
    packed = enumerate_repack(codes)
    k, n = np.arange(size_k)[:, None], np.arange(size_n)[None, :]
    word, shift = addresses(k, n, size_n)
    decoded = (packed[word].astype(np.uint64) >> shift.astype(np.uint64)) & 15
    np.testing.assert_array_equal(decoded, codes)
    # Every nibble is used exactly once, including all tile boundaries.
    assert len(np.unique(word * 8 + shift // 4)) == size_k * size_n


def test_scale_byte_inverse_matches_owned_converter():
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )

    values = torch.tensor(
        [0, 1 / 512, 1 / 64, 1 / 32, 0.5, 1, 32, 448], dtype=torch.float16
    )
    source = values.repeat(32).reshape(4, 64)
    permuted = marlin_permute_scales(source, 64, 64, 16)
    encoded, factor = nvfp4_marlin_process_scales(
        permuted, scale_factor=1.0, a_dtype=torch.bfloat16
    )
    assert factor == 1.0
    flat = torch.arange(source.numel())
    transposed = (flat // 64) * 64 + (flat % 8) * 8 + (flat % 64) // 8
    address = (transposed // 4) * 4 + (transposed % 2) * 2 + (transposed % 4) // 2
    bits = encoded.view(torch.uint8).flatten()[address].to(torch.int16) << 7
    decoded = (bits.view(torch.float16).float() / 128).reshape_as(source)
    # Preserve the existing converter's small-scale clipping in an oracle;
    # do not silently count it as a new kernel's quantization error.
    expected = torch.where(source.float() * 128 < 2, 0, source.float())
    torch.testing.assert_close(decoded, expected, atol=0, rtol=0)
    scale = torch.tensor([0.0078125], dtype=torch.float32)
    transformed = nvfp4_marlin_process_global_scale(scale, torch.bfloat16)
    torch.testing.assert_close(transformed * 2.0**-119, scale, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_k,size_n", [(16, 64), (32, 128), (256, 512)])
def test_inverse_address_against_native_repack(size_k, size_n):
    from vllm import _custom_ops as ops

    torch.manual_seed(73)
    codes = torch.randint(0, 16, (size_k, size_n), device="cuda", dtype=torch.int32)
    shifts = torch.arange(8, device="cuda", dtype=torch.int32) * 4
    original = (
        (codes.view(size_k // 8, 8, size_n).long() << shifts[None, :, None])
        .sum(1)
        .int()
    )
    packed = ops.gptq_marlin_repack(
        original,
        torch.empty(0, device="cuda", dtype=torch.int32),
        size_k,
        size_n,
        4,
        False,
    )
    k = torch.arange(size_k, device="cuda", dtype=torch.int64)[:, None]
    n = torch.arange(size_n, device="cuda", dtype=torch.int64)[None, :]
    word, shift = addresses(k, n, size_n)
    decoded = (packed.flatten()[word].long() >> shift) & 15
    torch.testing.assert_close(decoded.int(), codes, atol=0, rtol=0)
