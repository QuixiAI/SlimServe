# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton
from vllm.utils.math_utils import cdiv


def _quixicore_disabled() -> bool:
    """VLLM_QC_DISABLE_NATIVE=1 forces the Triton path.

    Exists so the native and Triton routes can be compared on an otherwise
    identical server; the kernels are bitwise-equal, so greedy output must
    match token for token.
    """
    import os

    return os.environ.get("VLLM_QC_DISABLE_NATIVE") == "1"


@cache
def _use_native_sample_kernels() -> bool:
    """Prefer the native sampler kernels over the Triton ones."""
    if _quixicore_disabled():
        return False
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available() and quixicore_ops.has("v2_gumbel_sample")


_GUMBEL_LOGITS_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    if _use_native_sample_kernels() and logits.dtype == torch.float32:
        # Native CUDA equivalent, bit-identical to the Triton kernel below.
        from vllm.quixicore import quixicore_ops

        quixicore_ops.v2_apply_temperature(logits, expanded_idx_mapping, temperature)
        return

    if logits.device.type == "mps":
        # Torch fallback, matching the kernel: greedy rows (temperature 0)
        # divide by 1 and are resolved by argmax downstream.
        temps = temperature[expanded_idx_mapping.to(torch.long)].to(logits.dtype)
        temps = torch.where(temps == 0, torch.ones_like(temps), temps)
        logits.div_(temps.unsqueeze(1))
        return

    BLOCK_SIZE = 8192
    num_blocks = cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# splitmix64 constants as signed int64 (torch integer arithmetic wraps mod
# 2**64, so the bit patterns match the unsigned reference).
_SM64_GAMMA = 0x9E3779B97F4A7C15 - (1 << 64)
_SM64_M1 = 0xBF58476D1CE4E5B9 - (1 << 64)
_SM64_M2 = 0x94D049BB133111EB - (1 << 64)


def _lshr64(z: torch.Tensor, k: int) -> torch.Tensor:
    """Logical (zero-fill) right shift for int64 tensors."""
    return (z >> k) & ((1 << (64 - k)) - 1)


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    z = x + _SM64_GAMMA
    z = (z ^ _lshr64(z, 30)) * _SM64_M1
    z = (z ^ _lshr64(z, 27)) * _SM64_M2
    return z ^ _lshr64(z, 31)


def mix64_int(a: int, b: int) -> int:
    """Python-int splitmix64 mix of two 64-bit values (for CPU generators)."""
    mask = (1 << 64) - 1

    def mix(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)

    return mix(mix(a & mask) ^ (b & mask))


def stateless_uniform_2d(
    seed: torch.Tensor,  # [rows] int64
    pos: torch.Tensor,  # [rows] int64
    n: int,
) -> torch.Tensor:
    """Deterministic uniforms in (0, 1), keyed by (seed, pos, column).

    Counter-based replacement for Triton's stateless Philox on devices
    without Triton (MPS): same (seed, pos) always yields the same row of
    noise, so per-request seeded sampling is reproducible across repeats
    and boots. Not bit-identical to the CUDA/Triton Philox stream, which
    no correctness gate requires; determinism and independence are what
    matter here.
    """
    row_key = _splitmix64(_splitmix64(seed.to(torch.int64)) ^ pos.to(torch.int64))
    idx = torch.arange(n, dtype=torch.int64, device=seed.device)
    z = _splitmix64(row_key.unsqueeze(1) + idx.unsqueeze(0) * _SM64_GAMMA)
    # Top 24 bits -> [0, 1), then clamp away exact zero (matching the
    # includes_zero=False contract of tl_rand32).
    u = _lshr64(z, 40).to(torch.float32) * (1.0 / (1 << 24))
    return u.clamp_(min=1.0 / (1 << 24))


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit
def gumbel_noised_argmax(
    logits,
    keys,
    mask,
    seed,
    pos,
    temp,
    USE_FP64: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr = True,
):
    """Argmax of logits under Gumbel-max sampling, or plain argmax at temp 0.

    `keys` indexes the noise, so the same token draws the same noise wherever it
    appears; `pos` and `seed` place the draw in the request's stream, which is
    what lets a draft and its verification agree. (Upstream also routes
    gumbel_block_argmax through this helper; our block kernel keeps its own
    inlined copy because it has diverged with native-kernel plumbing.)
    """
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Match the behavior of _temperature_kernel: if that kernel uses
        # tl.div_rn, this must too.
        logits = logits / temp

    # fp32 is the default reduction dtype; fp64 is ~1/32-1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        gumbel_seed = tl.randint(seed, pos)
        if USE_FP64:
            u = tl_rand64(gumbel_seed, keys, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, keys, includes_zero=False)
            # Draw the large-noise tail (which decides the argmax winner) from
            # u -> 0, where fp32 has fine resolution, instead of u -> 1, where
            # fp32 spacing is ~2**-24. The naive `-log(-log(u))` puts the winning
            # tail at u -> 1, hard-capping the noise at ~16.6 and coarsely
            # quantizing it; `log1p(-u)` == `log(1 - u)` keeps the tail in the
            # well-resolved region. `1 - u` would lose precision for small u, so
            # log1p is required.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    return tl.max(logits, axis=0, return_indices=True)


@triton.jit
def gumbel_block_argmax(
    logits,
    block,
    mask,
    token_idx,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr = False,
):
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )
    if temp != 0.0 and APPLY_TEMPERATURE:
        # Apply temperature.
        # NOTE(woosuk): Match the behavior of _temperature_kernel.
        # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
        logits = logits / temp

    if processed_logits_ptr is not None:
        # Store the temperature-applied logits.
        if processed_logits_col_ptr is not None:
            if PER_TOKEN_COL:
                col = tl.load(processed_logits_col_ptr + token_idx)
            else:
                col = tl.load(processed_logits_col_ptr)
        else:
            col = 0
        tl.store(
            processed_logits_ptr
            + req_state_idx * processed_logits_stride
            + col * vocab_size
            + block,
            logits,
            mask=mask & is_valid_req,
        )

    # fp32 is the default reduction dtype; fp64 is ~1/32–1/64x the throughput
    # on H100/Ada/Blackwell and empirically indistinguishable for Gumbel-max.
    if USE_FP64:
        logits = logits.to(tl.float64)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        gumbel_seed = tl.randint(seed, pos)

        if USE_FP64:
            u = tl_rand64(gumbel_seed, block, includes_zero=False)
            gumbel_noise = -tl.log(-tl.log(u))
        else:
            u = tl_rand32(gumbel_seed, block, includes_zero=False)
            # Draw the large-noise tail (which decides the argmax winner) from u -> 0,
            # where fp32 has fine resolution, instead of u -> 1, where fp32 spacing is
            # ~2**-24. The naive `-log(-log(u))` puts the winning tail at u -> 1,
            # hard-capping the noise at ~16.6 and coarsely quantizing it; using
            # `log1p(-u)` == `log(1 - u)` keeps the tail in the well-resolved region.
            # Note `1 - u` would lose precision for small u, so `log1p` is required.
            gumbel_noise = -tl.log(-tldevice.log1p(-u))

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    value, idx = tl.max(logits, axis=0, return_indices=True)
    return value, idx


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        processed_logits_ptr,
        processed_logits_stride,
        processed_logits_col_ptr,
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
        PER_TOKEN_COL=PER_TOKEN_COL,
    )
    token_id = block_idx * BLOCK_SIZE + idx
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    output_processed_logits: torch.Tensor | None = None,
    output_processed_logits_col: torch.Tensor | None = None,
    use_fp64: bool = False,
    all_greedy: bool | None = None,
) -> torch.Tensor:
    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if output_processed_logits_col is not None:
        output_processed_logits_col = output_processed_logits_col.contiguous()
    num_tokens, vocab_size = logits.shape
    if logits.device.type == "mps":
        req_indices = expanded_idx_mapping.to(torch.int64)
        row_temperatures = temperature[req_indices]
        processed = logits.float()
        if apply_temperature:
            divisors = torch.where(
                row_temperatures == 0, 1, row_temperatures
            ).unsqueeze(1)
            processed = processed / divisors

        greedy = row_temperatures == 0
        greedy_sampled = processed.argmax(dim=-1)
        if all_greedy is None:
            # Legacy callers do not carry CPU-side sampling state. The normal
            # sampler passes the hint and avoids this MPS queue drain.
            all_greedy = bool(greedy.all().cpu())
        if all_greedy:
            sampled = greedy_sampled
        else:
            # Stateless (seed, pos)-keyed Gumbel noise, mirroring the Triton
            # kernel's semantics (splitmix64 counter stream instead of Philox
            # -- deterministic per request, not bit-identical to CUDA). The
            # previous torch fallback drew from the unseeded global RNG,
            # which silently ignored per-request seeds: same-seed repeats
            # diverged at the first position with sampling entropy.
            row_seeds = seed[req_indices]
            row_pos = pos.to(torch.int64)
            u = stateless_uniform_2d(row_seeds, row_pos, vocab_size)
            # Same tail trick as the Triton kernel: -log(-log1p(-u)) keeps
            # the argmax-deciding large-noise tail in fp32's well-resolved
            # region near u -> 0.
            gumbel = -torch.log(-torch.log1p(-u))
            random_sampled = (processed + gumbel).argmax(dim=-1)
            sampled = torch.where(greedy, greedy_sampled, random_sampled)

        if output_processed_logits is not None:
            if output_processed_logits_col is None:
                output_processed_logits.copy_(processed)
            else:
                # A 0-dim column index broadcasts against req_indices, so the
                # scalar and per-token cases share one sync-free expression.
                cols = output_processed_logits_col.to(torch.int64)
                output_processed_logits[req_indices, cols].copy_(processed)
        return sampled.to(torch.int64)

    BLOCK_SIZE = 1024
    num_blocks = cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = logits.new_empty(num_tokens, num_blocks, dtype=torch.int64)
    local_max_dtype = torch.float64 if use_fp64 else torch.float32
    local_max = logits.new_empty(num_tokens, num_blocks, dtype=local_max_dtype)
    per_token_col = (
        output_processed_logits_col is not None
        and output_processed_logits_col.dim() > 0
    )
    if (
        _use_native_sample_kernels()
        and logits.dtype in _GUMBEL_LOGITS_DTYPES
        and pos.dtype == torch.int64
        and (
            output_processed_logits is None
            or output_processed_logits.dtype == torch.float32
        )
    ):
        # Native CUDA equivalent (bit-identical, incl. the Philox-based
        # Gumbel noise); shares the local argmax/max reduction below.
        from vllm.quixicore import quixicore_ops

        quixicore_ops.v2_gumbel_sample(
            local_argmax,
            local_max,
            output_processed_logits,
            output_processed_logits_col,
            logits,
            expanded_idx_mapping,
            seed,
            pos,
            temperature,
            apply_temperature,
            per_token_col,
        )
        max_block_idx = local_max.argmax(dim=-1, keepdim=True)
        return local_argmax.gather(dim=-1, index=max_block_idx).view(-1)

    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        output_processed_logits,
        output_processed_logits.stride(0) if output_processed_logits is not None else 0,
        output_processed_logits_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled
