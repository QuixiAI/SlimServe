# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-CUDA TurboQuant launchers over the QuixiCore kernels.

Drop-in equivalents of `triton_turboquant_store` and
`triton_turboquant_decode_attention` for the no-Triton Ampere deployment.
The store path is bitwise-equal to the Triton kernels (e4b15 fp8 keys);
the decode path matches Triton's op ordering for head_size 64 and the
split-KV reduction exactly. Tensor preprocessing (rotation GEMM, norms)
stays in the same cuBLAS/ATen calls the Triton launchers use.
"""

from functools import cache
from typing import Any

import torch

from vllm.quixicore.ops import quixicore_ops
from vllm.v1.attention.ops.triton_turboquant_decode import _use_fp8_e4b15


@cache
def native_turboquant_available() -> bool:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    return quixicore_ops.is_available() and quixicore_ops.has(
        "turboquant_decode_stage1"
    )


def native_turboquant_supported(head_size: int, key_fp8: bool) -> bool:
    """Whether the native kernels cover this TQ configuration."""
    if not native_turboquant_available():
        return False
    if head_size % 32 != 0 or head_size > 128:
        return False
    # Native fp8 keys implement the e4b15 (Ampere/Ada) format only; on
    # sm89+ the Triton path keeps using e4nv.
    if key_fp8 and not _use_fp8_e4b15(0):
        return False
    return True


def native_turboquant_store(
    key: torch.Tensor,  # [N, H, D] — raw keys (post-RoPE)
    value: torch.Tensor,  # [N, H, D] — raw values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    slot_mapping: torch.Tensor,  # [N]
    PiT: torch.Tensor,  # [D, D] float32
    midpoints: torch.Tensor,  # [n_centroids-1] float32
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
) -> None:
    """Native TQ store; mirrors `triton_turboquant_store` exactly."""
    N, H, D = key.shape
    NH = N * H

    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()
        quixicore_ops.turboquant_store_fp8(
            k_flat, v_flat, kv_cache, slot_mapping, H,
            key_packed_size, value_quant_bits,
        )
        return

    # Normalize + rotation GEMM externally — the identical ATen calls the
    # Triton launcher makes, so the kernel inputs match bitwise.
    k_flat = key.float().reshape(NH, D)
    norms = k_flat.norm(dim=1, keepdim=True)
    x_hat = k_flat / (norms + 1e-8)
    y = x_hat @ PiT
    v_flat = value.float().reshape(NH, D)
    quixicore_ops.turboquant_store_mse(
        y.contiguous(), norms.squeeze(1).contiguous(), v_flat.contiguous(),
        midpoints, kv_cache, slot_mapping, H, mse_bits, key_packed_size,
        value_quant_bits,
    )


def native_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
    sliding_window: int = 0,
) -> torch.Tensor:
    """Native TQ decode; mirrors `triton_turboquant_decode_attention`."""
    B, Hq, D = query.shape
    device = query.device

    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    quixicore_ops.turboquant_decode_stage1(
        q_rot, kv_cache, block_table, seq_lens, centroids, mid_o,
        NUM_KV_SPLITS, mse_bits, key_packed_size, value_quant_bits, scale,
        key_fp8, norm_correction, sliding_window,
    )

    # Stage 2 derives per-split occupancy from seq_len; with a window the
    # visible span is min(seq_len, W), so hand it the clamped lengths.
    if sliding_window > 0:
        seq_lens = seq_lens.clamp(max=sliding_window)

    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    quixicore_ops.turboquant_decode_stage2(
        mid_o, output, lse, seq_lens, NUM_KV_SPLITS
    )
    return output
