# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import torch

import vllm._custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    get_marlin_input_dtype,
    marlin_make_workspace,
    marlin_make_workspace_new,
    marlin_permute_bias,
    marlin_permute_scales,
    should_use_atomic_add_reduce,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)


def is_fp8_marlin_supported():
    if current_platform.is_rocm():
        if not torch.cuda.is_available():
            return False
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx90", "gfx94", "gfx95"])
    return current_platform.has_device_capability(75)


def fp8_fused_exponent_bias_into_scales(scales):
    fp8_exponent = 4
    if scales.dtype == torch.half:
        target_exponent = 5
    elif scales.dtype == torch.bfloat16:
        target_exponent = 8
    # exponent_bias_fp16 = 2 ** 4 - 2 ** 3 = 8
    # exponent_bias_bf16 = 2 ** 7 - 2 ** 3 = 120
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp8_exponent - 1)
    s = torch.ones_like(scales) * 2
    s = s**exponent_bias
    return scales * s


def apply_fp8_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    workspace: torch.Tensor,
    size_n: int,
    size_k: int,
    bias: torch.Tensor | None,
    input_dtype: torch.dtype | None = None,
    fp8_is_fnuz: bool | None = None,
    use_fp32_reduce: bool = USE_FP32_REDUCE_DEFAULT,
    use_fp8_mfma: bool = False,
    use_fp8_activation: bool = True,
) -> torch.Tensor:
    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (size_n,)

    if use_fp8_mfma and use_fp8_activation:
        # W8A8: FP8 activations + FP8 weights via native FP8 MFMA.
        a_fp8, a_scales = ops.scaled_fp8_quant(
            reshaped_x, use_per_token_if_dynamic=True
        )
        output = ops.fp8_mfma_marlin_gemm(
            a=a_fp8.view(torch.uint8),
            b_q_weight=weight,
            b_scales=weight_scale,
            a_scales=a_scales.view(-1),
            fp8_is_fnuz=False,
            size_m=reshaped_x.size(0),
            size_n=size_n,
            size_k=size_k,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    if use_fp8_mfma and not use_fp8_activation:
        # W8A16 MFMA: BF16/FP16 activations + FP8 weights via FP16 MFMA.
        output = ops.fp8_mfma_marlin_gemm(
            a=reshaped_x,
            b_q_weight=weight,
            b_scales=weight_scale,
            a_scales=None,
            fp8_is_fnuz=fp8_is_fnuz or False,
            size_m=reshaped_x.size(0),
            size_n=size_n,
            size_k=size_k,
        )
        if bias is not None:
            output = output + bias
        return output.reshape(out_shape)

    # W8A16 gptq_marlin fallback (non-CDNA3 ROCm or CUDA).
    use_atomic_add = False
    if not current_platform.is_rocm():
        use_atomic_add = should_use_atomic_add_reduce(
            m=reshaped_x.size(0),
            n=size_n,
            k=size_k,
            device=input.device,
            dtype=input.dtype,
        )

    inputs = reshaped_x
    a_scales = None
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise RuntimeError("Marlin W8A8 is not supported.")

    output = ops.gptq_marlin_gemm(
        a=inputs,
        c=None,
        b_q_weight=weight,
        b_bias=bias,
        b_scales=weight_scale,
        a_scales=a_scales,
        global_scale=None,
        b_zeros=None,
        g_idx=None,
        perm=None,
        workspace=workspace,
        b_q_type=scalar_types.float8_e4m3fn,
        size_m=reshaped_x.size(0),
        size_n=size_n,
        size_k=size_k,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )

    return output.reshape(out_shape)


def _is_cdna3_or_later() -> bool:
    """Check if the current ROCm GPU is CDNA3+ (gfx94x, gfx95x).

    These GPUs have native FP8 MFMA instructions
    (mfma_f32_16x16x32_fp8_fp8). Older CDNA (gfx90x) and RDNA (gfx12)
    do NOT have this instruction.
    """
    if not current_platform.is_rocm():
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return any(gfx in arch for gfx in ["gfx94", "gfx95"])


def _repack_weight_mfma_tiled_dense(
    weight: torch.Tensor, size_n: int, size_k: int
) -> torch.Tensor:
    """Repack FP8 [N, K] weight to MFMA-tiled int32 format for dense GEMM.

    MFMA 16x16x32 FP8 B operand: lane l supplies 8 FP8 bytes at
    B[(l//16)*8..(l//16)*8+7, l%16].  We tile B as
    [N//16, K//32, 4_halves, 16_cols, 8_k_per_half] so that lane l
    loads from a contiguous 8-byte aligned address at offset
    (l//16)*128 + (l%16)*8 within each 512-byte tile.
    """
    assert weight.shape == (size_n, size_k), (
        f"Expected [N={size_n}, K={size_k}], got {weight.shape}"
    )
    assert size_k % 32 == 0 and size_n % 16 == 0

    K_tiles = size_k // 32
    N_tiles = size_n // 16

    w = weight.view(torch.uint8)  # [N, K]
    # Reshape to [N_tiles, 16_cols, K_tiles, 4_halves, 8_k_per_half]
    w = w.reshape(N_tiles, 16, K_tiles, 4, 8)
    # N-first layout: [N_tiles, K_tiles, 4_halves, 16_cols, 8_k]
    w = w.permute(0, 2, 3, 1, 4).contiguous()
    # View as int32: last dim 8 uint8 -> 2 int32
    w = w.view(torch.int32)  # [N_tiles, K_tiles, 4, 16, 2]
    # Reshape to [size_k//16, size_n*4] — same outer shape as Marlin packed
    w = w.reshape(size_k // 16, -1)
    return w


def prepare_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    size_k_first: bool = True,
    input_dtype: torch.dtype | None = None,
) -> None:
    # W8A8 via input_dtype is now supported on CDNA3+ through the MFMA path.
    # Only reject it when MFMA is not available.
    if (input_dtype is not None and input_dtype.itemsize == 1
            and not _is_cdna3_or_later()):
        raise RuntimeError("Marlin W8A8 is not supported on this platform.")

    fp8_is_fnuz = False
    if hasattr(torch, "float8_e4m3fnuz"):
        fp8_is_fnuz = layer.weight.dtype == torch.float8_e4m3fnuz
    layer.fp8_is_fnuz = fp8_is_fnuz

    part_size_n = layer.output_size_per_partition
    part_size_k = layer.input_size_per_partition
    weight_block_size = getattr(layer, "weight_block_size", None)

    if size_k_first:
        assert layer.weight.shape == (part_size_k, part_size_n)
    else:
        assert layer.weight.shape == (part_size_n, part_size_k)

    device = layer.weight.device

    # Auto-detect CDNA3+ (gfx94x/gfx95x) for MFMA-tiled weight format.
    # On CDNA3+: ALWAYS use MFMA-tiled weights (both W8A8 and W8A16).
    #   VLLM_DENSE_FP8_MFMA=1 (default): W8A8 (quantize activations to FP8)
    #   VLLM_DENSE_FP8_MFMA=0: W8A16 (BF16/FP16 activations, FP16 MFMA)
    # On older CDNA (gfx90x) or CUDA: falls back to gptq_marlin W8A16.
    is_cdna3 = _is_cdna3_or_later()
    v = os.getenv("VLLM_DENSE_FP8_MFMA", "1").strip().lower()
    fp8_mfma_not_disabled = v not in ("0", "false", "no", "off")

    # MFMA-tiled weights: always on CDNA3+ if shapes allow
    use_fp8_mfma = (
        is_cdna3
        and part_size_k % 32 == 0
        and part_size_n % 16 == 0
    )
    # FP8 activation quantization: only when MFMA env not disabled
    use_fp8_activation = use_fp8_mfma and fp8_mfma_not_disabled

    if current_platform.is_rocm():
        if use_fp8_mfma and use_fp8_activation:
            logger.info_once(
                "Marlin FP8 dense: native FP8 MFMA (W8A8) enabled on CDNA3+.",
                scope="global",
            )
        elif use_fp8_mfma:
            logger.info_once(
                "Marlin FP8 dense: W8A16 MFMA enabled on CDNA3+.",
                scope="global",
            )
        else:
            logger.info_once(
                "Marlin FP8 dense: using weight-only FP8 path (W8A16).",
                scope="global",
            )

    layer.use_fp8_mfma = use_fp8_mfma
    layer.use_fp8_activation = use_fp8_activation

    # WORKSPACE (only needed for W8A16 path; MFMA path uses no workspace)
    if current_platform.is_rocm():
        layer.workspace = marlin_make_workspace(part_size_n, device)
    else:
        layer.workspace = marlin_make_workspace_new(device)

    if use_fp8_mfma:
        # --- MFMA-tiled path ---
        # Handle FNUZ: on MI300X, weights may be e4m3fnuz. The FP8 MFMA
        # instruction uses platform-native format, so no conversion needed.
        # We only need weights as raw uint8 bytes.
        weight_for_repack = layer.weight
        if size_k_first:
            # [K, N] -> [N, K] for repack
            weight_for_repack = weight_for_repack.T.contiguous()
        # weight_for_repack is [N, K] now

        marlin_qweight = _repack_weight_mfma_tiled_dense(
            weight_for_repack, part_size_n, part_size_k
        ).to(device)
        replace_parameter(layer, "weight", marlin_qweight)

        # WEIGHT SCALES: linear layout [num_groups, N] (no Marlin permutation)
        if "weight_scale" in dir(layer):
            scales = layer.weight_scale.to(layer.orig_dtype)
        elif "weight_scale_inv" in dir(layer):
            scales = layer.weight_scale_inv.to(layer.orig_dtype)

        if weight_block_size is None:
            logical_widths = getattr(layer, "logical_widths", [])
            if scales.nelement() == 1:
                scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
            elif scales.nelement() == len(logical_widths):
                assert sum(logical_widths) == part_size_n
                lw_tensor = scales.new_tensor(
                    logical_widths, dtype=torch.int64
                )
                scales = scales.view(1, -1).repeat_interleave(
                    lw_tensor, dim=1
                )
            elif scales.nelement() > 1 and scales.nelement() != part_size_n:
                assert part_size_n % scales.nelement() == 0
                s_size = scales.nelement()
                scales = scales.view(1, s_size)
                scales = scales.repeat_interleave(part_size_n // s_size, 1)
            else:
                scales = scales.view(1, part_size_n)
        else:
            if not size_k_first:
                scales = scales.T.contiguous()
            block_n = weight_block_size[0]
            scales = scales.repeat_interleave(block_n, 1)
            scales = scales[:, :part_size_n]

        # FNUZ scale correction: when weights were converted fn->fnuz, scales
        # were doubled. The native MFMA uses platform-native FP8 directly,
        # so we undo the doubling to get correct numerics.
        if fp8_is_fnuz:
            want_fnuz = (
                hasattr(torch, "float8_e4m3fnuz")
                and current_platform.fp8_dtype() == torch.float8_e4m3fnuz
            )
            if want_fnuz:
                # Scales already correct for platform-native fnuz.
                pass
            else:
                # fnuz weights treated as fn: halve scales.
                scales = scales * 0.5

        # Keep scales in linear layout — no marlin_permute_scales.
        if hasattr(layer, "weight_scale"):
            replace_parameter(layer, "weight_scale", scales.contiguous())
        elif hasattr(layer, "weight_scale_inv"):
            replace_parameter(
                layer, "weight_scale_inv", scales.contiguous()
            )

        # Bias: no Marlin permutation needed for MFMA path.
        return

    # --- W8A16 fallback path (existing) ---
    perm = torch.empty(0, dtype=torch.int, device=device)
    qweight = pack_fp8_to_int32(layer.weight, size_k_first)
    if not size_k_first:
        qweight = qweight.T.contiguous()

    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=qweight,
        perm=perm,
        size_k=part_size_k,
        size_n=part_size_n,
        num_bits=8,
    )
    replace_parameter(layer, "weight", marlin_qweight)

    # WEIGHT SCALES
    if "weight_scale" in dir(layer):
        scales = layer.weight_scale.to(layer.orig_dtype)
    elif "weight_scale_inv" in dir(layer):
        scales = layer.weight_scale_inv.to(layer.orig_dtype)

    group_size = -1 if weight_block_size is None else weight_block_size[1]

    if weight_block_size is None:
        logical_widths = getattr(layer, "logical_widths", [])
        if scales.nelement() == 1:
            scales = scales.view(1, 1).repeat_interleave(part_size_n, 1)
        elif scales.nelement() == len(logical_widths):
            assert sum(logical_widths) == part_size_n, (
                f"Sum of logical_widths ({sum(logical_widths)}) must be equal "
                f"to part_size_n ({part_size_n})"
            )
            lw_tensor = scales.new_tensor(logical_widths, dtype=torch.int64)
            scales = scales.view(1, -1).repeat_interleave(lw_tensor, dim=1)
        elif scales.nelement() > 1 and scales.nelement() != part_size_n:
            assert part_size_n % scales.nelement() == 0
            s_size = scales.nelement()
            scales = scales.view(1, s_size)
            scales = scales.repeat_interleave(part_size_n // s_size, 1)
        else:
            scales = scales.view(1, part_size_n)
    else:
        if not size_k_first:
            scales = scales.T.contiguous()
        block_n = weight_block_size[0]
        scales = scales.repeat_interleave(block_n, 1)
        scales = scales[:, :part_size_n]

    marlin_scales = marlin_permute_scales(
        s=scales, size_k=part_size_k, size_n=part_size_n, group_size=group_size
    )
    if torch.version.hip is None and input_dtype != torch.float8_e4m3fn:
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)
    if hasattr(layer, "weight_scale"):
        replace_parameter(layer, "weight_scale", marlin_scales)
    elif hasattr(layer, "weight_scale_inv"):
        replace_parameter(layer, "weight_scale_inv", marlin_scales)

    if hasattr(layer, "bias") and layer.bias is not None:
        assert layer.bias.shape == (part_size_n,)
        bias = marlin_permute_bias(layer.bias)
        replace_parameter(layer, "bias", bias)


def prepare_fp8_moe_layer_for_marlin(
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shuffle weights and scales into marlin format.

    Note that this function has the side effect of adding a `workspace`
    attribute to the layer. This `workspace` does not need to be
    registered as a Parameter as it is not used during weight reloading.
    """

    # Marlin MoE has an optional native FP8 MFMA path on ROCm CDNA3+.
    # Log a single line so users don't misinterpret which compute path is used.
    v = os.getenv("VLLM_MARLIN_MOE_FP8_MFMA", "1").strip().lower()
    fp8_mfma_enabled = v not in ("0", "false", "no", "off")
    if fp8_mfma_enabled and current_platform.is_rocm() and current_platform.supports_fp8():
        logger.info_once(
            "Marlin MoE: native FP8 MFMA (W8A8) is enabled on this ROCm GPU.",
            scope="global",
        )
    else:
        logger.warning_once(
            "Marlin MoE: native FP8 MFMA (W8A8) is disabled or unavailable; "
            "falling back to the 16-bit compute path. This may degrade "
            "performance for compute-heavy workloads.",
            scope="global",
        )
    input_dtype = get_marlin_input_dtype()
    if input_dtype is not None and input_dtype.itemsize == 1:
        raise NotImplementedError("Marlin W8A8 is not supported.")

    # ROCm platforms often use the e4m3fnuz dtype for FP8 (platform-native).
    # Ensure weights/scales are consistent with the kernel's decode path.
    want_fnuz = (
        current_platform.is_rocm()
        and hasattr(torch, "float8_e4m3fnuz")
        and current_platform.fp8_dtype() == torch.float8_e4m3fnuz
    )
    has_fnuz = hasattr(torch, "float8_e4m3fnuz")
    fp8_is_fnuz = has_fnuz and w13_weight.dtype == torch.float8_e4m3fnuz
    layer.fp8_is_fnuz = want_fnuz

    if want_fnuz:
        # Convert e4m3fn -> e4m3fnuz if needed, and sanitize the NaN bit pattern
        # (0x80 / -128) that represents 0 in e4m3fn but NaN in e4m3fnuz.
        ROCM_FP8_NAN_AS_INT = -128
        if w13_weight.dtype == torch.float8_e4m3fn:
            w13_i8 = w13_weight.view(torch.int8)
            w13_i8[w13_i8 == ROCM_FP8_NAN_AS_INT] = 0
            w13_weight = w13_i8.view(torch.float8_e4m3fnuz)
            w2_i8 = w2_weight.view(torch.int8)
            w2_i8[w2_i8 == ROCM_FP8_NAN_AS_INT] = 0
            w2_weight = w2_i8.view(torch.float8_e4m3fnuz)
            # For the same bits representation, e4m3fnuz value is half of
            # the e4m3fn value, so we should double the scaling factor.
            w13_weight_scale = w13_weight_scale * 2.0
            w2_weight_scale = w2_weight_scale * 2.0
        elif fp8_is_fnuz:
            w13_i8 = w13_weight.view(torch.int8)
            w13_i8[w13_i8 == ROCM_FP8_NAN_AS_INT] = 0
            w13_weight = w13_i8.view(torch.float8_e4m3fnuz)
            w2_i8 = w2_weight.view(torch.int8)
            w2_i8[w2_i8 == ROCM_FP8_NAN_AS_INT] = 0
            w2_weight = w2_i8.view(torch.float8_e4m3fnuz)
        else:
            raise AssertionError(f"Unsupported FP8 weight dtype: {w13_weight.dtype}")
    else:
        # Kernel decodes FP8 weights as e4m3fn.
        if fp8_is_fnuz:
            w13_weight = w13_weight.view(torch.int8).view(torch.float8_e4m3fn)
            w2_weight = w2_weight.view(torch.int8).view(torch.float8_e4m3fn)
            w13_weight_scale = w13_weight_scale * 0.5
            w2_weight_scale = w2_weight_scale * 0.5

    e = layer.num_experts
    k = layer.hidden_size
    n = layer.intermediate_size_per_partition
    w13_n = w13_weight.size(1)
    weight_block_size = getattr(layer, "weight_block_size", None)

    # WORKSPACE
    device = layer.w13_weight.device
    # NOTE(rob): we do not need to register the workspace as a param
    # because it is not used as part of the weight reloading process.
    layer.workspace = marlin_make_workspace_new(device, 4)
    perm = torch.empty(0, dtype=torch.int, device=device)

    # WEIGHT
    # Repack weights to marlin format (or MFMA-tiled format on ROCm)

    # Use MFMA-tiled B layout on ROCm CDNA3+ with FP8 MFMA enabled.
    # This layout matches the MFMA 16x16x32 FP8 B operand mapping directly,
    # enabling a single coalesced 8-byte load per lane instead of 4 scattered
    # k_perm loads + byte extraction.  The implicit contract is that the kernel
    # will see use_fp8_mfma=True whenever this path is taken.
    use_mfma_tiled = (
        fp8_mfma_enabled
        and current_platform.is_rocm()
        and current_platform.supports_fp8()
    )

    def repack_weight_mfma_tiled(
        name: str, weight: torch.Tensor
    ) -> torch.Tensor:
        """Repack FP8 [E, N, K] weight to MFMA-tiled int32 format.

        MFMA 16x16x32 FP8 B operand: lane l supplies 8 FP8 bytes at
        B[(l//16)*8..(l//16)*8+7, l%16].  We tile B as
        [K//32, N//16, 4_halves, 16_cols, 8_k_per_half] so that lane l
        loads from a contiguous 8-byte aligned address at offset
        (l//16)*128 + (l%16)*8 within each 512-byte tile.  This gives
        a perfectly coalesced 128-byte load per half-wave (16 lanes).
        """
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k)
        assert size_k % 32 == 0 and size_n % 16 == 0, (
            f"MFMA tiled repack requires K%32==0, N%16==0, "
            f"got K={size_k}, N={size_n}"
        )

        K_tiles = size_k // 32
        N_tiles = size_n // 16

        w = weight.view(torch.uint8)  # [E, N, K]
        # Reshape to [E, N_tiles, 16_cols, K_tiles, 4_halves, 8_k_per_half]
        w = w.reshape(e, N_tiles, 16, K_tiles, 4, 8)
        # N-first layout: [E, N_tiles, K_tiles, 4_halves, 16_cols, 8_k]
        # This gives stride=512 between consecutive K iterations (contiguous),
        # vs K-first which had stride=N/16*512 (huge, causes L1 cache misses).
        w = w.permute(0, 1, 3, 4, 2, 5).contiguous()
        # View as int32: last dim 8 uint8 → 2 int32
        w = w.view(torch.int32)  # [E, N_tiles, K_tiles, 4, 16, 2]
        # Reshape to [E, size_k//16, size_n*4] — same outer shape as Marlin
        # packed format so that marlin_moe_intermediate_size() still works
        # (it reads w2.size(1) * 16 to get the intermediate size).
        w = w.reshape(e, size_k // 16, -1)
        return w

    def repack_weight(name: str, weight: torch.Tensor) -> torch.Tensor:
        tensor_list = []
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        assert weight.shape == (e, size_n, size_k)

        for i in range(e):
            qweight = pack_fp8_to_int32(weight[i], size_k_first=False)
            qweight = qweight.T.contiguous()

            marlin_qweight = ops.gptq_marlin_repack(
                b_q_weight=qweight, perm=perm, size_k=size_k, size_n=size_n, num_bits=8
            )
            tensor_list.append(marlin_qweight)

        return torch.cat([x.unsqueeze(0) for x in tensor_list], 0)

    if use_mfma_tiled:
        logger.info_once(
            "Marlin MoE: using MFMA-tiled B weight layout for FP8.",
            scope="global",
        )
        w13_weight = repack_weight_mfma_tiled("w13", w13_weight)
        w2_weight = repack_weight_mfma_tiled("w2", w2_weight)
    else:
        w13_weight = repack_weight("w13", w13_weight)
        w2_weight = repack_weight("w2", w2_weight)

    # WEIGHT SCALES
    # Permute scales
    group_size = -1 if weight_block_size is None else weight_block_size[1]

    def permute_scales(scales: torch.Tensor, name: str) -> torch.Tensor:
        scales = scales.to(layer.orig_dtype)
        tensor_list = []
        if "w13" in name:
            size_n, size_k = w13_n, k
        else:
            size_n, size_k = k, n

        # marlin kernel only support channel-wise and group-wise quantization
        # we need to convert the scales
        if weight_block_size is None:
            if scales.nelement() == e:
                # tensor-wise quantization -> channel-wise quantization
                # (e, 1, 1) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, 1).repeat_interleave(size_n, 2)
            elif scales.nelement() > e and scales.nelement() != e * size_n:
                assert (e * size_n) % scales.nelement() == 0
                s_size = scales.nelement() // e
                # tensor-wise quantization (for gate-up proj)
                #     -> channel-wise quantization
                # (e, 1, s_size) =>(repeat)=> (e, 1, size_n)
                scales = scales.view(e, 1, s_size)
                scales = scales.repeat_interleave(size_n // s_size, 2)
            else:
                # channel-wise quantization
                # (e, 1, size_n)
                scales = scales.view(e, 1, size_n)
        else:
            # block-wise quantization -> group-wise quantization
            # (e, size_k // block_size[1], ceil(size_n / block_size[0]))
            #  =>(repeat)=> (e, size_k // block_size[1], size_n)
            scales = scales.permute(0, 2, 1)
            block_n = weight_block_size[0]
            scales = scales.repeat_interleave(block_n, 2)
            # size_n may not divisible by block_size[0]
            scales = scales[..., :size_n].contiguous()

        if use_mfma_tiled:
            # MFMA-tiled B format uses linear scale layout: no Marlin
            # permutation needed. The kernel uses direct s[group * N + col]
            # access instead of k_scale_perm_inv lookup.
            # scales is already [E, num_groups, size_n] from above.
            pass
        else:
            for i in range(e):
                marlin_scales = marlin_permute_scales(
                    s=scales[i], size_k=size_k, size_n=size_n,
                    group_size=group_size
                )
                tensor_list.append(marlin_scales)

            scales = torch.cat([x.unsqueeze(0) for x in tensor_list], 0)
        if torch.version.hip is None and input_dtype != torch.float8_e4m3fn:
            scales = fp8_fused_exponent_bias_into_scales(scales)
        return scales

    w13_weight_scale = permute_scales(w13_weight_scale, "w13")
    w2_weight_scale = permute_scales(w2_weight_scale, "w2")

    return w13_weight, w2_weight, w13_weight_scale, w2_weight_scale


def pack_fp8_to_int32(
    fp8_tensor: torch.Tensor, size_k_first: bool = True
) -> torch.Tensor:
    """
    Repack FP8 weights to gptq format (packed int32 elements)
    """
    allowed_dtypes = [torch.float8_e4m3fn]
    if hasattr(torch, "float8_e4m3fnuz"):
        allowed_dtypes.append(torch.float8_e4m3fnuz)
    assert fp8_tensor.dtype in allowed_dtypes
    assert fp8_tensor.ndim == 2

    fp8_tensor = fp8_tensor.T if size_k_first else fp8_tensor
    fp8_tensor = fp8_tensor.contiguous()
    # fp8_tensor is contiguous and have shape (N, K) now
    # with `.view(torch.int32)`, it become (N, K // 4)
    int32_tensor = fp8_tensor.view(torch.int32)
    return int32_tensor.T.contiguous() if size_k_first else int32_tensor


def marlin_quant_fp8_torch(weight, group_size, input_dtype=None):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    if is_a_8bit:
        assert input_dtype == torch.float8_e4m3fn

    size_n, size_k = weight.shape
    device = weight.device

    if group_size != -1:
        scales = weight.view(size_n, -1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(group_size, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales
    else:
        scales = weight.view(size_n, 1, group_size).abs().max(-1)[0] / 448
        repeated_scales = scales.repeat_interleave(size_k, 1)
        fp8_weight = (weight / repeated_scales).to(torch.float8_e4m3fn)
        weight_ref = fp8_weight.to(weight.dtype) * repeated_scales

    packed_weight = pack_fp8_to_int32(fp8_weight, False).T.contiguous()
    perm = torch.empty(0, dtype=torch.int, device=device)
    marlin_qweight = ops.gptq_marlin_repack(
        b_q_weight=packed_weight,
        perm=perm,
        size_k=size_k,
        size_n=size_n,
        num_bits=8,
        is_a_8bit=is_a_8bit,
    )

    marlin_scales = marlin_permute_scales(
        s=scales.T,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    if torch.version.hip is None:
        marlin_scales = fp8_fused_exponent_bias_into_scales(marlin_scales)

    return weight_ref.T, marlin_qweight, marlin_scales
