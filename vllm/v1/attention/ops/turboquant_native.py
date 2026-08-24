# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native TurboQuant launchers over the QuixiCore CUDA and Metal kernels.

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


# Read-only constants the Metal launch path would otherwise rebuild on every
# call. Centroid entries keep a strong reference to the source tensor so a
# recycled id() can never alias a different tensor. Two contracts: the
# source centroids must never be mutated in place (the cache would keep
# serving the stale scaled copy), and callers must treat every returned
# tensor as an immutable shared alias. Entries live for the process
# lifetime — fine for one long-lived server, a leak across model reloads.
_metal_scaled_centroids_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
_metal_const_cache: dict[tuple, torch.Tensor] = {}


def metal_scaled_centroids(centroids: torch.Tensor, head_size: int) -> torch.Tensor:
    entry = _metal_scaled_centroids_cache.get(id(centroids))
    if entry is None or entry[0] is not centroids:
        entry = (centroids, (centroids * head_size**0.5).contiguous())
        _metal_scaled_centroids_cache[id(centroids)] = entry
    return entry[1]


def metal_ones(size: int, device: torch.device) -> torch.Tensor:
    key = ("ones", size, str(device))
    t = _metal_const_cache.get(key)
    if t is None:
        t = torch.ones(size, dtype=torch.float32, device=device)
        _metal_const_cache[key] = t
    return t


def metal_arange(width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = ("arange", width, str(device), dtype)
    t = _metal_const_cache.get(key)
    if t is None:
        t = torch.arange(width, device=device, dtype=dtype)
        _metal_const_cache[key] = t
    return t


def metal_neg_inf_sinks(num_heads: int, device: torch.device) -> torch.Tensor:
    key = ("sinks", num_heads, str(device))
    t = _metal_const_cache.get(key)
    if t is None:
        t = torch.full(
            (num_heads,), -float("inf"), dtype=torch.float32, device=device
        )
        _metal_const_cache[key] = t
    return t


@cache
def native_turboquant_available() -> bool:
    from vllm.platforms import current_platform

    if current_platform.is_metal():
        return quixicore_ops.is_available() and quixicore_ops.has(
            "turboquant_attention_metal"
        )
    if not current_platform.is_cuda():
        return False
    return quixicore_ops.is_available() and quixicore_ops.has(
        "turboquant_decode_stage1"
    )


def native_turboquant_supported(head_size: int, key_fp8: bool) -> bool:
    """Whether the native kernels cover this TQ configuration."""
    if not native_turboquant_available():
        return False
    from vllm.platforms import current_platform

    if current_platform.is_metal():
        return head_size in (64, 128, 256, 512)
    if head_size % 32 != 0 or head_size > 512:
        return False
    # Native fp8 keys implement the e4b15 (Ampere/Ada) format only; on
    # sm89+ the Triton path keeps using e4nv.
    return not (key_fp8 and not _use_fp8_e4b15(0))


def _select_num_kv_splits(
    batch_size: int,
    num_query_heads: int,
    max_num_kv_splits: int,
    sliding_window: int,
) -> int:
    """Choose enough split-KV work to occupy the GPU without excess scratch.

    The native stage-1 grid has one warp-sized block per
    ``(request, query_head, split)``. A fixed 32 splits is useful for a single
    full-attention decode, but is actively harmful for DSpark's large
    synthetic-decode batches: its partial-output tensor scales with every
    query token and head. The batch shape is fixed for each CUDA graph, so this
    launch choice is capture/replay stable.
    """
    if sliding_window <= 0:
        return max_num_kv_splits

    # Roughly ten warp blocks per A100 SM is enough to cover the long register
    # dependency chain in the D=512 kernel. Never create more partitions than
    # visible tokens or the configured graph-safe maximum.
    blocks_per_split = max(1, batch_size * num_query_heads)
    occupancy_splits = (1024 + blocks_per_split - 1) // blocks_per_split
    return max(
        1,
        min(max_num_kv_splits, sliding_window, occupancy_splits),
    )


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
    from vllm.platforms import current_platform

    if current_platform.is_metal():
        centroids = torch.cat(
            (
                -midpoints.new_tensor([float("inf")]),
                midpoints,
                midpoints.new_tensor([float("inf")]),
            )
        )
        centroids = (centroids[:-1] + centroids[1:]) * 0.5
        centroids = centroids.clamp(-4.0, 4.0)
        signs = torch.ones(key.shape[-1], dtype=torch.float32, device=key.device)
        quixicore_ops.turboquant_encode_metal(
            key,
            value,
            kv_cache,
            slot_mapping,
            centroids,
            signs,
            8 if key_fp8 else mse_bits,
            key_fp8,
            value_quant_bits,
        )
        return

    N, H, D = key.shape
    NH = N * H

    if key_fp8:
        k_flat = key.reshape(NH, D).contiguous()
        v_flat = value.reshape(NH, D).contiguous()
        quixicore_ops.turboquant_store_fp8(
            k_flat,
            v_flat,
            kv_cache,
            slot_mapping,
            H,
            key_packed_size,
            value_quant_bits,
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
        y.contiguous(),
        norms.squeeze(1).contiguous(),
        v_flat.contiguous(),
        midpoints,
        kv_cache,
        slot_mapping,
        H,
        mse_bits,
        key_packed_size,
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
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Native TQ decode; mirrors `triton_turboquant_decode_attention`."""
    from vllm.platforms import current_platform

    B, Hq, D = query.shape
    if current_platform.is_metal():
        block_size = kv_cache.shape[1]
        width = (
            sliding_window if sliding_window > 0 else block_table.shape[1] * block_size
        )
        positions = metal_arange(width, query.device, seq_lens.dtype)
        visible = seq_lens.clamp(max=width)
        logical = (seq_lens - visible).unsqueeze(1) + positions.unsqueeze(0)
        valid = positions.unsqueeze(0) < visible.unsqueeze(1)
        block_col = torch.div(logical, block_size, rounding_mode="floor")
        block_col = block_col.clamp(min=0, max=block_table.shape[1] - 1)
        blocks = block_table.gather(1, block_col.to(torch.long))
        slots = blocks * block_size + torch.remainder(logical, block_size)
        slots = torch.where(valid, slots, -1).to(torch.int32).contiguous()
        metal_centroids = metal_scaled_centroids(centroids, D)
        signs = metal_ones(D, query.device)
        metal_sinks = (
            sinks
            if sinks is not None
            else metal_neg_inf_sinks(query.shape[1], query.device)
        )
        return quixicore_ops.turboquant_attention_metal(
            query,
            kv_cache,
            slots,
            visible,
            metal_centroids,
            signs,
            metal_sinks,
            scale,
            kv_cache.shape[2],
            8 if key_fp8 else mse_bits,
            key_fp8,
            value_quant_bits,
        )

    device = query.device

    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = _select_num_kv_splits(B, Hq, max_num_kv_splits, sliding_window)

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
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        NUM_KV_SPLITS,
        mse_bits,
        key_packed_size,
        value_quant_bits,
        scale,
        key_fp8,
        norm_correction,
        sliding_window,
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

    quixicore_ops.turboquant_decode_stage2(mid_o, output, lse, seq_lens, NUM_KV_SPLITS)
    if sinks is not None:
        sink_scale = torch.sigmoid(lse - sinks.to(lse.dtype).unsqueeze(0))
        output.mul_(sink_scale.unsqueeze(-1).to(output.dtype))
    return output
