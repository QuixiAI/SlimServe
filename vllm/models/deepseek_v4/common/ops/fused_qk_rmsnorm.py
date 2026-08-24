# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

# fp32 copies of the (load-time-constant) norm weights, keyed by data_ptr —
# weights are permanent for the server lifetime, so this never grows past
# two entries per attention layer.
_W32_CACHE: dict[int, torch.Tensor] = {}


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    if current_platform.is_metal():
        # Native single-dispatch path: qc_rms_norm_w32_* computes
        # T(float(x) * rrms * w_f32), the same fp32-multiply formula as the
        # eager oracle below (parity to reduction-order ulps).
        if qr.dtype in (torch.float16, torch.bfloat16):
            from vllm.quixicore.ops import quixicore_ops

            if quixicore_ops.is_available() and quixicore_ops.has("rms_norm"):
                # The fp32 weight copies are step-constant; cache them keyed
                # by storage (fp16 -> fp32 widening is exact) instead of
                # casting per call. Keyed on data_ptr because callers pass
                # `.weight.data`, a fresh Tensor object every access.
                qw = _W32_CACHE.get(q_weight.data_ptr())
                if qw is None:
                    qw = q_weight.float().contiguous()
                    _W32_CACHE[q_weight.data_ptr()] = qw
                kw = _W32_CACHE.get(kv_weight.data_ptr())
                if kw is None:
                    kw = kv_weight.float().contiguous()
                    _W32_CACHE[kv_weight.data_ptr()] = kw
                return (
                    quixicore_ops.rms_norm(qr, qw, eps),
                    quixicore_ops.rms_norm(kv, kw, eps),
                )

        # Triton has no Metal target. Preserve the kernel's fp32 reduction and
        # single output cast so this is also a correctness oracle for the
        # fused MSL implementation above.
        def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_fp32 = x.float()
            rrms = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
            return (x_fp32 * rrms * weight.float()).to(x.dtype)

        return rms_norm(qr, q_weight), rms_norm(kv, kv_weight)

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
    )
    return qr_out, kv_out
