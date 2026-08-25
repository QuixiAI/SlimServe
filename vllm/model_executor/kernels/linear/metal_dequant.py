# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bring-up dequantization for compressed-tensors checkpoints on Metal.

torch-MPS cannot allocate or cast float8 tensors, so the Metal linear kernels
store FP8 payloads as uint8 bytes and decode them through small lookup tables
built on the CPU (where fp8 views and casts work, subnormals included) and
moved to the device once. These helpers are the correctness-first path behind
MetalNvFp4LinearKernel and MetalWFp8A16LinearKernel; the optimized QuixiCore
Metal GEMV kernels replace the apply path, not this decode math.

Dequantization walks the weight in row chunks: the int64 indices that
device-side LUT gathers require are 8 bytes per element, and an unchunked
lm_head ([248320, 5120]) would transiently need ~10 GiB of them.
"""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    kE2M1ToFloat_handle,
)

_ROW_CHUNK = 4096

# Per-device caches: one 256-entry E4M3 table and one 16-entry signed E2M1
# nibble table.
_E4M3_LUT: dict[str, torch.Tensor] = {}
_E2M1_LUT: dict[str, torch.Tensor] = {}


def e4m3_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    lut = _E4M3_LUT.get(key)
    if lut is None:
        lut = (
            torch.arange(256, dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
        )
        # 0x7f/0xff encode NaN; real checkpoints never contain them, but a NaN
        # would poison the whole output row, so decode them as 0.
        lut = torch.nan_to_num(lut, nan=0.0).to(device)
        _E4M3_LUT[key] = lut
    return lut


def e2m1_lut(device: torch.device) -> torch.Tensor:
    """Signed 16-entry table over the full nibble: bit 3 is the sign."""
    key = str(device)
    lut = _E2M1_LUT.get(key)
    if lut is None:
        magnitudes = kE2M1ToFloat_handle.val.to(torch.float32)
        lut = torch.cat([magnitudes, -magnitudes]).to(device)
        _E2M1_LUT[key] = lut
    return lut


def dequant_nvfp4(
    weight_packed: torch.Tensor,  # uint8 [N, K/2], low nibble = even column
    weight_scale: torch.Tensor,  # uint8 (E4M3 bytes) [N, K/16]
    weight_global_scale: torch.Tensor,  # fp32 scalar, already the multiplier
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """NVFP4 group-16 weight to a dense [N, K] tensor."""
    assert weight_packed.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8
    n, k_half = weight_packed.shape
    k = k_half * 2
    device = weight_packed.device
    nibbles = e2m1_lut(device)
    scales = e4m3_lut(device)

    out = torch.empty(n, k, dtype=out_dtype, device=device)
    for row in range(0, n, _ROW_CHUNK):
        chunk = weight_packed[row : row + _ROW_CHUNK].to(torch.long)
        rows = chunk.shape[0]
        values = torch.empty(rows, k, dtype=torch.float32, device=device)
        values[:, 0::2] = nibbles[chunk & 0x0F]
        values[:, 1::2] = nibbles[chunk >> 4]
        block_scale = scales[weight_scale[row : row + _ROW_CHUNK].to(torch.long)]
        values = values.view(rows, k // 16, 16) * block_scale.unsqueeze(-1)
        out[row : row + _ROW_CHUNK] = (values.view(rows, k) * weight_global_scale).to(
            out_dtype
        )
    return out


def dequant_fp8_channel(
    weight: torch.Tensor,  # uint8 (E4M3 bytes) [N, K]
    weight_scale: torch.Tensor,  # [N, 1], any float dtype
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """FP8-per-channel weight to a dense [N, K] tensor."""
    assert weight.dtype == torch.uint8
    device = weight.device
    scales = e4m3_lut(device)
    scale_f32 = weight_scale.to(torch.float32)

    n, k = weight.shape
    out = torch.empty(n, k, dtype=out_dtype, device=device)
    for row in range(0, n, _ROW_CHUNK):
        chunk = scales[weight[row : row + _ROW_CHUNK].to(torch.long)]
        out[row : row + _ROW_CHUNK] = (chunk * scale_f32[row : row + _ROW_CHUNK]).to(
            out_dtype
        )
    return out
