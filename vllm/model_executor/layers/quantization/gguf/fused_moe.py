# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from functools import partial

import torch

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from . import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    _gguf_moe_weight_loader,
    _gguf_moe_weight_type_loader,
)
from .utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES, logger


def _moe_vec_row_limit(default: int, env: str, cuda_default: int = 64) -> int:
    override = os.environ.get(env)
    if override:
        return int(override)
    if not current_platform.is_rocm():
        return cuda_default
    return default


def _qc_mm_min_tokens() -> int:
    """Token threshold where the Metal tiled MoE GEMM replaces the GEMV for
    w13 (llama.cpp's exact GEMV/GEMM crossover: n_tokens >= 32). The decode
    verify batch is 6 tokens, so the tile never engages at decode."""
    return int(os.environ.get("VLLM_QC_MOE_MM_MIN_TOKENS", "32"))


def _metal_q2k_sum_rows_supported(rows: int, dtype: torch.dtype) -> bool:
    """The folded kernel has no output-row tail; its unfused parent does."""
    if dtype == torch.float16:
        return rows % 32 == 0
    if dtype == torch.bfloat16:
        return rows % 8 == 0
    return False


def _use_dsv4_ampere_fused() -> bool:
    """Allow controlled A/B tests against the existing GGUF MoE path."""
    value = os.environ.get("VLLM_GGUF_DSV4_AMPERE_FUSED", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _use_dsv4_ampere_q2k_repack() -> bool:
    value = os.environ.get("VLLM_GGUF_DSV4_REPACK_Q2K", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _use_dsv4_ampere_q4k() -> bool:
    """Fused Q4_K gate/up+SwiGLU+Q8 and Q8xQ4_K weighted down for the hybrid
    artifact's Q4_K tail layers (37-42) at decode widths."""
    value = os.environ.get("VLLM_GGUF_DSV4_AMPERE_Q4K", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _dsv4_q4k_row_limit() -> int:
    """Widest batch the fused Q4_K route takes; wider batches keep the
    generic MMQ tile route, which amortizes expert reload across rows."""
    value = os.environ.get("VLLM_GGUF_DSV4_Q4K_ROWS", "64")
    try:
        return max(0, min(256, int(value)))
    except ValueError:
        return 64


def _use_dsv4_ampere_mxfp4() -> bool:
    """Fused MXFP4 gate/up+SwiGLU+Q8 and Q8xMXFP4 down for decode widths."""
    value = os.environ.get("VLLM_GGUF_DSV4_AMPERE_MXFP4", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _dsv4_w1_wide_rows() -> int:
    """Routed-row threshold for the 8-wide fused IQ2 W1 layout; must mirror
    iq2_wide_route_threshold() in dsv4_moe_ampere.cuh (same env var)."""
    value = os.environ.get("VLLM_GGUF_DSV4_W1_WIDE_ROWS", "256")
    try:
        parsed = int(value)
    except ValueError:
        return 256
    return parsed if parsed > 0 else 256


def _dsv4_mxfp4_row_limit() -> int:
    """Widest batch the fused MXFP4 route takes; wider goes to generic MMQ.

    DSV4 routing is near-uncorrelated, so the MMQ tile kernels get little
    expert reuse and the per-route warp-GEMV stays ahead well past decode
    widths. Measured on A100 TP4 (dsv4-4-mxfp4, c8 exact harness) -- see
    perf/optimization_status.md.
    """
    value = os.environ.get("VLLM_GGUF_DSV4_MXFP4_ROWS", "64")
    try:
        return max(0, min(256, int(value)))
    except ValueError:
        return 64


def _use_dsv4_mxfp4_seg() -> bool:
    """Segmented (permutation-based) wide MXFP4 route: device-side route
    grouping, tensor-core W1 with fused SwiGLU+Q8_1 epilogue, tensor-core
    W2, deterministic reduce. Replaces the moe_align padded-metadata MMQ
    route for DSV4 shapes; static grids make it CUDA-graph-safe."""
    value = os.environ.get("VLLM_GGUF_DSV4_MXFP4_SEG", "1")
    return not value.startswith("0")


def _use_dsv4_iq2_seg() -> bool:
    """Segmented tensor-core wide route for the hybrid (IQ2_XXS, Q2_K)
    expert pair; below the token gate the tuned fused per-route pipeline
    keeps the batch."""
    value = os.environ.get("VLLM_GGUF_DSV4_IQ2_SEG", "1")
    return not value.startswith("0")


def _dsv4_iq2_seg_tokens() -> int:
    """Measured crossover vs the 8-wide fused pipeline (A100 TP4 shapes,
    per-layer op pair): fused wins 0.93/1.75 ms at 48 tokens through
    4.65/6.05 at 512; seg wins 6.85/8.77 at 1024 and 8.50/17.00 at 2048.
    768 splits the bracket: prefill chunks ride the tiles, decode/verify
    widths keep the fused route."""
    value = os.environ.get("VLLM_GGUF_DSV4_IQ2_SEG_TOKENS", "768")
    try:
        return max(0, int(value))
    except ValueError:
        return 768


def _use_dsv4_ampere_mxfp4_repack() -> bool:
    value = os.environ.get("VLLM_GGUF_DSV4_REPACK_MXFP4", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _use_dsv4_ampere_iq2_repack() -> bool:
    value = os.environ.get("VLLM_GGUF_DSV4_REPACK_IQ2", "1")
    return value.lower() not in {"0", "false", "off", "no"}


def _qc_metal_soa_repack(
    qweight: torch.Tensor, block_bytes: int, planes: tuple[tuple[int, int], ...]
) -> torch.Tensor:
    """Byte-neutral AoS -> per-expert SoA plane permutation for the Metal
    multi-row MoE kernels (A100 precedent: ggml_dsv4_repack_q2_k).

    ``planes`` lists (offset, size) slices of each ``block_bytes`` superblock;
    within every expert the slices are concatenated plane-by-plane, largest
    (the aligned code plane) first. Same shape/bytes out; the caller must
    copy the result back into the original allocation (never keep raw and
    repacked expert stacks alive together). Chunked over experts so the
    transient stays at ~1/8 of the tensor plus the full-size scratch."""
    experts, rows, row_bytes = qweight.shape
    blocks = row_bytes // block_bytes
    assert sum(size for _, size in planes) == block_bytes
    out = torch.empty_like(qweight)
    chunk = 32
    for e0 in range(0, experts, chunk):
        blk = qweight[e0 : e0 + chunk].view(-1, rows, blocks, block_bytes)
        parts = [
            blk[..., off : off + size].reshape(blk.shape[0], -1) for off, size in planes
        ]
        out[e0 : e0 + chunk].view(blk.shape[0], -1).copy_(torch.cat(parts, dim=1))
    return out


# No iq2_xxs repack helper: both the per-row plane split and the paired-row
# A100-style layout measured slower than AoS on Apple (see the note in
# process_weights_after_loading and optimization_status 2026-08-13).


def _metal_weighted_sum(
    out: torch.Tensor, topk_weights: torch.Tensor, out_hidden: torch.Tensor
) -> bool:
    """Metal one-dispatch weighted reduce mirroring the eager chain's
    numerics bitwise (fp32 products, sequential expert-slot sum)."""
    if not (
        out.dtype in (torch.float16, torch.bfloat16)
        and out_hidden.dtype == out.dtype
        and topk_weights.dtype == torch.float32
        and out.is_contiguous()
        and topk_weights.is_contiguous()
        and out_hidden.is_contiguous()
        and out.shape[1] <= 8
    ):
        return False
    from vllm.quixicore import quixicore_ops

    if not (quixicore_ops.is_available() and quixicore_ops.has("moe_weighted_sum")):
        return False
    quixicore_ops.moe_weighted_sum(out, topk_weights, out_hidden)
    return True


def _use_quixi_weighted_sum(
    out: torch.Tensor, topk_weights: torch.Tensor, out_hidden: torch.Tensor
) -> bool:
    """One-launch weighted reduce instead of mul_ + moe_sum (CUDA only)."""
    if current_platform.is_rocm():
        return False
    if not (
        out.dtype == torch.bfloat16
        and out_hidden.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and out.is_contiguous()
        and out_hidden.is_contiguous()
    ):
        return False
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available()


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    swiglu_limit: float | None,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    expert_map: torch.Tensor | None,
    w1_repacked: bool = False,
    w2_repacked: bool = False,
    quant_input: torch.Tensor | None = None,
    defer_down: bool = False,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(inp: torch.Tensor):
        d = inp.shape[-1] // 2
        output_shape = inp.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)
        apply_moe_activation(
            activation_enum,
            out,
            inp,
            clamp_limit=swiglu_limit,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
        )
        return out

    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    mmq_ok = qweight_type in MMQ_QUANT_TYPES and qweight_type2 in MMQ_QUANT_TYPES
    vec_ok = qweight_type in MMVQ_QUANT_TYPES and qweight_type2 in MMVQ_QUANT_TYPES
    if current_platform.is_metal():
        # Metal's grouped path reads the selected raw GGUF expert blocks with
        # one device-resident GEMV dispatch. It is vector-only for now, but it
        # avoids the per-route host synchronization that made startup and
        # decode unusable.
        mmq_ok = False
    if mmq_ok or vec_ok:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        global_num_experts = expert_map.shape[0] if expert_map is not None else E
        local_topk_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
        top_k = topk_ids.shape[1]
        # Q4_K experts (the hybrid artifact's tail layers), decode widths:
        # fused gate/up+SwiGLU+Q8_1 with a weighted Q8xQ4_K down sum (see
        # dsv4_q4k_moe_ampere.cuh). Replaces moe_vec w1 + SwiGLU + requant +
        # four moe_align launches + MMQ w2 + weighted moe_sum with two
        # launches. Prefill and wider batches keep the generic route below.
        if (
            qweight_type == 12
            and qweight_type2 == 12
            and activation_enum == MoEActivation.SILU
            and activation_situ_beta is None
            and activation_situ_linear_beta is None
            and not current_platform.is_rocm()
            and not current_platform.is_metal()
            and not defer_down
            and _use_dsv4_ampere_q4k()
            and num_tokens <= _dsv4_q4k_row_limit()
            and top_k in (6, 8)
            and x.shape[1] % 256 == 0
            and (N // 2) % 256 == 0
        ):
            return ops.ggml_dsv4_moe_a8_q4k(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                local_topk_ids.contiguous(),
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                quant_input,
            )
        # MXFP4 experts, decode widths: fused gate/up+SwiGLU+Q8_1 with a
        # weighted Q8xMXFP4 down accumulation (see dsv4_mxfp4_moe_ampere.cuh).
        # Prefill and wider batches keep the generic MMQ/MMVQ route below.
        if (
            qweight_type == 39
            and qweight_type2 == 39
            and activation_enum == MoEActivation.SILU
            and activation_situ_beta is None
            and activation_situ_linear_beta is None
            and not current_platform.is_rocm()
            and not current_platform.is_metal()
            and _use_dsv4_ampere_mxfp4()
            and num_tokens <= _dsv4_mxfp4_row_limit()
            and top_k in (6, 8)
            and x.shape[1] % 32 == 0
            and (N // 2) % 32 == 0
        ):
            direct_ids = local_topk_ids.contiguous()
            return ops.ggml_dsv4_moe_a8_mxfp4(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                direct_ids,
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                w1_repacked,
                w2_repacked,
                quant_input,
            )
        # MXFP4 experts, wide batches: segmented tensor-core pipeline with
        # the fused SwiGLU+Q8_1 W1 epilogue (dsv4_mxfp4_seg_ampere.cuh).
        # Narrower batches stay on the fused per-route decode path above.
        if (
            qweight_type == 39
            and qweight_type2 == 39
            and activation_enum == MoEActivation.SILU
            and activation_situ_beta is None
            and activation_situ_linear_beta is None
            and not current_platform.is_rocm()
            and not current_platform.is_metal()
            and _use_dsv4_mxfp4_seg()
            and x.shape[1] % 256 == 0
            and (N // 2) % 256 == 0
            and w2.shape[1] % 128 == 0
            and E <= 256
        ):
            return ops.ggml_dsv4_moe_a8_mxfp4_seg(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                local_topk_ids.contiguous(),
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                w1_repacked,
                w2_repacked,
            )
        # ggml_moe_a8_vec has no weight reuse across tokens -- it reloads the
        # expert row for every (token, k) pair -- so its cost is linear in rows
        # while ggml_moe_a8 is flat.  w2 sees num_tokens*top_k rows, so it
        # crosses over 8x sooner than w1; gating both on one request count (the
        # old `x.shape[0] > 64`) leaves w2 on the GEMV kernel until it is 3.5x
        # slower than the GEMM. Crossovers measured on MI300X at GLM-5.2 TP2
        # shapes are between 32 and 64 input rows for w1 and between 128 and
        # 256 routed rows for w2 after grouping four adjacent output rows in
        # the Q2_K vector kernel and using a 4x64 Q2_K MMQ tile. Keep
        # separate overrides because other GGUF model shapes can cross over at
        # different points.
        # A100 w1 crossover at spec verify width (4 query tokens/request):
        # vec still wins at 4 tokens (40.2 vs 41.3 ms/step; real routing is
        # near-uncorrelated so MMQ streams as many experts as vec) but loses
        # from 32 tokens up (720 vs 604 us kernel-level, -10% ms/step e2e at
        # bs=8). Once the MMQ tile stopped computing its padding columns it
        # also won at 16 tokens (331 vs 360 us kernel-level, -10.3% ms/step at
        # bs=4), so the limit moved 16 -> 8: bs=1 verify stays on vec, bs>=2
        # verify goes to MMQ.
        w1_vec = vec_ok and (
            not mmq_ok
            or num_tokens <= _moe_vec_row_limit(32, "VLLM_GGUF_MOE_VEC_W1", 8)
        )
        # The MMVQ kernels read raw 17-byte MXFP4 blocks; repacked expert
        # stacks must stay on the repack-aware tile/fused routes.
        if qweight_type == 39 and w1_repacked:
            w1_vec = False
        # On A100 the w2 crossover sits below any batch this serves: forcing the
        # MMQ tile kernel measured neutral at 8 routed rows and +2.9% / +6.7% at
        # 32 / 64 (GLM-5.2 Q2_K, TP8, CUDA graphs), so vec never wins for w2.
        # w1 is left alone -- the same sweep showed MMQ costs it ~3% at batch 1.
        w2_rows = num_tokens * top_k
        w2_vec = vec_ok and (
            not mmq_ok or w2_rows <= _moe_vec_row_limit(128, "VLLM_GGUF_MOE_VEC_W2", 0)
        )
        if qweight_type2 == 39 and w2_repacked:
            w2_vec = False
        dsv4_fused_native = (
            qweight_type == 16
            and qweight_type2 == 10
            and activation_enum == MoEActivation.SILU
            and activation_situ_beta is None
            and activation_situ_linear_beta is None
            and not current_platform.is_rocm()
            and not current_platform.is_metal()
            and _use_dsv4_ampere_fused()
        )
        if dsv4_fused_native:
            w1_vec = False
            w2_vec = False

        # Decode consumes routes in their original [token, top_k] order. The
        # direct IQ2_XXS W1 kernel and repacked Q2_K down kernel do not read
        # the sorted/aligned metadata, so avoid four alignment launches per
        # layer on the production batch-1 path.
        w1_decode_value = os.environ.get("VLLM_GGUF_DSV4_W1_DECODE")
        direct_dsv4_decode = (
            dsv4_fused_native
            and w2_repacked
            and num_tokens <= 8
            and top_k in (6, 8)
            and (N // 2) % 32 == 0
            and x.shape[1] % 256 == 0
            and (w1_decode_value is None or not w1_decode_value.startswith("0"))
        )
        output_owned_w2 = (
            dsv4_fused_native
            and w2_repacked
            and x.shape[1] == 4096
            and w2.shape[1] in (1024, 2048)
            and w2.shape[1] * (4096 // w2.shape[1]) == 4096
            and w2.shape[2] == 672
        )
        if direct_dsv4_decode and not output_owned_w2:
            direct_ids = local_topk_ids.contiguous()
            return ops.ggml_dsv4_moe_a8(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                direct_ids,
                direct_ids,
                direct_ids,
                direct_ids,
                direct_ids,
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                w1_repacked,
                True,
                quant_input,
                defer_down,
            )

        # Hybrid pair, wide batches: segmented tensor-core pipeline (IQ2 W1
        # with the fused SwiGLU+Q8_1 epilogue, Q2_K W2 with min-term mma).
        # Requires the load-time repacked layouts, which are the A100
        # production state.
        if (
            dsv4_fused_native
            and _use_dsv4_iq2_seg()
            and w1_repacked
            and w2_repacked
            and not output_owned_w2
            and not defer_down
            and num_tokens > _dsv4_iq2_seg_tokens()
            and top_k in (6, 8)
            and x.shape[1] % 256 == 0
            and (N // 2) % 256 == 0
            and w2.shape[1] % 128 == 0
            and E <= 256
        ):
            return ops.ggml_dsv4_moe_a8_iq2_seg(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                local_topk_ids.contiguous(),
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
            )

        sorted_token_ids = num_tokens_post_padded = None
        w1_expert_ids = w2_expert_ids = None
        if not (w1_vec and w2_vec):
            # w1 and w2 need not be the same quant -- DeepSeek-V4 ships IQ2_XXS
            # gate/up with Q2_K down -- and each tile kernel has its own mmq_x.
            # A kernel reads expert_ids[blockIdx.y] with blockIdx.y counting its
            # own tiles, so one shared array is only correct when the widths
            # match; otherwise every w2 tile picks the wrong expert and the
            # output is quietly wrong.
            #
            # So align the rows to a width both agree on (their LCM, which each
            # mmq_x divides) and hand each kernel an expert_ids expanded to its
            # own tile count. The alignment is a property of the layout, the
            # tile width a property of the kernel; conflating them is what tied
            # a type's mmq_x to whatever it was paired with.
            w1_block = 0 if w1_vec else ops.ggml_moe_get_block_size(qweight_type)
            w2_block = 0 if w2_vec else ops.ggml_moe_get_block_size(qweight_type2)
            if dsv4_fused_native:
                # The paired Ampere IQ2 kernel is eight routes wide. Q2_K down
                # remains four wide and receives an expanded expert map below.
                # Must match iq2_wide_route_threshold() in dsv4_moe_ampere.cuh
                # (same env) or the expert map width disagrees with the kernel.
                w1_block = 8 if w2_rows >= _dsv4_w1_wide_rows() else 4
            if w1_block <= 0:
                w1_vec = True
            if w2_block <= 0:
                w2_vec = True
            mmq_blocks = [b for b in (w1_block, w2_block) if b > 0]
            if mmq_blocks:
                align = math.lcm(*mmq_blocks)
                (
                    sorted_token_ids,
                    expert_ids,
                    num_tokens_post_padded,
                ) = moe_align_block_size(
                    topk_ids,
                    align,
                    global_num_experts,
                    expert_map=expert_map,
                )
                # One entry per align columns -> one per mmq_x columns.
                w1_expert_ids = (
                    None
                    if w1_vec
                    else expert_ids
                    if align == w1_block
                    else expert_ids.repeat_interleave(align // w1_block)
                )
                w2_expert_ids = (
                    None
                    if w2_vec
                    else expert_ids
                    if align == w2_block
                    else expert_ids.repeat_interleave(align // w2_block)
                )

        if output_owned_w2:
            if direct_dsv4_decode:
                direct_ids = local_topk_ids.contiguous()
                sorted_for_w1 = direct_ids
                experts_for_w1 = direct_ids
                count_for_w1 = direct_ids
            else:
                assert sorted_token_ids is not None
                assert w1_expert_ids is not None
                assert num_tokens_post_padded is not None
                sorted_for_w1 = sorted_token_ids
                experts_for_w1 = w1_expert_ids
                count_for_w1 = num_tokens_post_padded
            local_quant_mid = ops.ggml_dsv4_moe_w1_a8(
                x,
                w1,
                topk_weights.contiguous(),
                local_topk_ids.contiguous(),
                sorted_for_w1,
                experts_for_w1,
                count_for_w1,
                N // 2,
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                w1_repacked,
                quant_input,
            )
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                tensor_model_parallel_all_gather,
            )

            full_quant_mid = tensor_model_parallel_all_gather(local_quant_mid, dim=-1)
            local_output = ops.ggml_dsv4_moe_down_output_owned(
                w2,
                full_quant_mid,
                local_topk_ids.contiguous(),
                num_tokens,
                top_k,
            )
            partial_output = x.new_zeros(x.shape)
            row_start = get_tensor_model_parallel_rank() * local_output.shape[1]
            partial_output[:, row_start : row_start + local_output.shape[1]].copy_(
                local_output
            )
            return partial_output

        if (
            dsv4_fused_native
            and sorted_token_ids is not None
            and w1_expert_ids is not None
            and w2_expert_ids is not None
        ):
            return ops.ggml_dsv4_moe_a8(
                x,
                w1,
                w2,
                topk_weights.contiguous(),
                local_topk_ids.contiguous(),
                sorted_token_ids,
                w1_expert_ids,
                w2_expert_ids,
                num_tokens_post_padded,
                N // 2,
                w2.shape[1],
                top_k,
                num_tokens,
                0.0 if swiglu_limit is None else swiglu_limit,
                w1_repacked,
                w2_repacked,
                quant_input,
                defer_down,
            )

        # Both kernels emit rows in flat (token, k) order, so either can feed
        # the other.
        # Metal iq2_xxs decode: the multi-row MoE kernel fuses the SwiGLU
        # epilogue (bit-exact vs the two-step path) — one dispatch instead
        # of gate|up + act, and half the intermediate write traffic.
        use_fused_act = (
            current_platform.is_metal()
            and qweight_type == 16  # IQ2_XXS
            and expert_map is None
            and activation_enum == MoEActivation.SILU
            and activation_situ_beta is None
            and activation_situ_linear_beta is None
        )
        # Metal SoA-repacked expert stacks (see process_weights_after_loading)
        # are only readable by the SoA kernel twins; thread the layout flag.
        metal_soa2 = current_platform.is_metal() and w2_repacked
        # Metal prefill widths: the tiled MoE GEMM (llama.cpp mul_mm_id port)
        # replaces the per-slot w13 GEMV once enough tokens share each
        # expert's weight tile. AoS iq2_xxs only (the resident w13 layout);
        # output is the same flat (token, slot) row order, so act() and the
        # down path are unchanged. No fused pair+SwiGLU tile: measured
        # negative twice (optimization_status 2026-08-13/14).
        use_mm_w1 = (
            current_platform.is_metal()
            and qweight_type == 16  # IQ2_XXS
            and expert_map is None
            and x.dtype == torch.float16
            and w1.shape[0] <= 256
            and top_k in (2, 4, 6, 8)
            and x.shape[1] % 256 == 0
            and N % 64 == 0
            and num_tokens >= _qc_mm_min_tokens()
        )
        took_mm_w1 = w1_vec and use_mm_w1
        if took_mm_w1:
            logger.info_once(
                "quixicore(metal): tiled MoE prefill GEMM active (w13 + w2)"
            )
            out = ops.ggml_moe_mm_id(
                x,
                w1,
                local_topk_ids,
                top_k,
                qweight_type,
                N,
                num_tokens,
            )
        elif w1_vec and use_fused_act:
            out = ops.ggml_moe_a8_vec_swiglu(
                x,
                w1,
                local_topk_ids,
                top_k,
                qweight_type,
                N,
                num_tokens,
                clamp_limit=swiglu_limit,
            )
        elif w1_vec:
            out = ops.ggml_moe_a8_vec(
                x,
                w1,
                local_topk_ids,
                top_k,
                qweight_type,
                N,
                num_tokens,
                expert_parallel=expert_map is not None,
            )
        else:
            out = ops.ggml_moe_a8(
                x,
                w1,
                sorted_token_ids,
                w1_expert_ids,
                num_tokens_post_padded,
                qweight_type,
                N,
                top_k,
                num_tokens,
                mxfp4_repacked=(qweight_type == 39 and w1_repacked),
            )
        # Apply the activation exactly once: the mm branch never fuses it
        # (even when the fused-act GEMV would have been eligible), while the
        # fused-act GEMV already did.
        if took_mm_w1 or not (w1_vec and use_fused_act):
            out = act(out)
        # Metal prefill widths, down projection: the tiled q2_K GEMM over the
        # per-slot activations (B row = slot id; SoA planes supported), then
        # the existing bit-matching weighted reduce. Decode stays on the
        # sum-folded GEMV below.
        use_mm_w2 = (
            current_platform.is_metal()
            and w2_vec
            and qweight_type2 == 10  # Q2_K
            and expert_map is None
            and out.dtype == torch.float16
            and w2.shape[0] <= 256
            and top_k in (2, 4, 6, 8)
            and out.shape[1] % 256 == 0
            and w2.shape[1] % 64 == 0
            and out_hidden_states.is_contiguous()
            and out_hidden_states.dtype == out.dtype
            and out_hidden_states.shape == (num_tokens, w2.shape[1])
            and num_tokens >= _qc_mm_min_tokens()
        )
        if use_mm_w2:
            slots = ops.ggml_moe_mm_id(
                out,
                w2,
                local_topk_ids,
                top_k,
                qweight_type2,
                w2.shape[1],
                num_tokens,
                soa=metal_soa2,
            )
            slots = slots.reshape(num_tokens, top_k, w2.shape[1])
            if not _metal_weighted_sum(
                slots, topk_weights.contiguous(), out_hidden_states
            ):
                reduced = (slots.float() * topk_weights.unsqueeze(-1)).sum(dim=1)
                out_hidden_states.copy_(reduced.to(out_hidden_states.dtype))
            return out_hidden_states
        # Metal q2_K decode: fold the down GEMV, the (tokens, topk, N)
        # intermediate, and the weighted expert-slot sum into one kernel
        # writing out_hidden_states directly (rounding points match the
        # unfused chain; see qgemv_moe_mr_q2_K_sum).
        use_sum6 = (
            current_platform.is_metal()
            and w2_vec
            and qweight_type2 == 10  # Q2_K
            and expert_map is None
            and top_k <= 8
            and out_hidden_states.is_contiguous()
            and out_hidden_states.dtype == out.dtype
            and out_hidden_states.shape == (num_tokens, w2.shape[1])
            and _metal_q2k_sum_rows_supported(w2.shape[1], out.dtype)
        )
        if use_sum6:
            ops.ggml_moe_a8_vec_sum(
                out,
                w2,
                local_topk_ids,
                topk_weights.contiguous(),
                top_k,
                qweight_type2,
                w2.shape[1],
                num_tokens,
                out_hidden_states,
                soa=metal_soa2,
            )
            return out_hidden_states
        if w2_vec:
            out = ops.ggml_moe_a8_vec(
                out,
                w2,
                local_topk_ids,
                1,
                qweight_type2,
                w2.shape[1],
                w2_rows,
                soa=metal_soa2,
            )
        else:
            out = ops.ggml_moe_a8(
                out,
                w2,
                sorted_token_ids,
                w2_expert_ids,
                num_tokens_post_padded,
                qweight_type2,
                w2.shape[1],
                1,
                w2_rows,
                mxfp4_repacked=(qweight_type2 == 39 and w2_repacked),
            )
        out = out.reshape(num_tokens, top_k, w2.shape[1])
        if current_platform.is_metal():
            if not _metal_weighted_sum(out, topk_weights, out_hidden_states):
                reduced = (out.float() * topk_weights.unsqueeze(-1)).sum(dim=1)
                out_hidden_states.copy_(reduced.to(out_hidden_states.dtype))
        elif _use_quixi_weighted_sum(out, topk_weights, out_hidden_states):
            from vllm.quixicore import quixicore_ops

            quixicore_ops.moe_weighted_sum(
                out, topk_weights.contiguous(), out_hidden_states
            )
        else:
            out = out.mul_(topk_weights.view(num_tokens, top_k, 1))
            ops.moe_sum(out, out_hidden_states)
    else:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        local_topk_ids = expert_map[topk_ids] if expert_map is not None else topk_ids
        if current_platform.is_metal():
            # Iterating MPS scalar tensors performs a device/host sync for each
            # `ii < 0` test and again when the tensor is used as a Python
            # index.  DSV4 has six routes in 43 layers, making those tiny
            # synchronizations dominate the actual matvecs.  Transfer the
            # 6*token integer routing table once per layer and keep route
            # weights on-device.
            local_topk_ids_host = local_topk_ids.to("cpu").tolist()
        else:
            local_topk_ids_host = local_topk_ids
        for tok, idx in enumerate(local_topk_ids_host):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for slot, ii in enumerate(idx):
                if ii < 0:
                    continue
                ww = topk_weights[tok, slot]
                out = fused_mul_mat_gguf_op(inp, w1[ii], qweight_type)
                out = act(out)
                current_state = fused_mul_mat_gguf_op(out, w2[ii], qweight_type2).mul_(
                    ww
                )
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            if current_hidden_state is None:
                out_hidden_states[tok].zero_()
            else:
                out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
    swiglu_limit: float | None,
    activation_situ_beta: float | None,
    activation_situ_linear_beta: float | None,
    expert_map: torch.Tensor | None,
    w1_repacked: bool = False,
    w2_repacked: bool = False,
    quant_input: torch.Tensor | None = None,
    defer_down: bool = False,
) -> torch.Tensor:
    del (
        w1,
        w2,
        topk_weights,
        topk_ids,
        qweight_type,
        qweight_type2,
        activation,
        swiglu_limit,
        activation_situ_beta,
        activation_situ_linear_beta,
        expert_map,
        w1_repacked,
        w2_repacked,
        quant_input,
        defer_down,
    )
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf
except AttributeError as error:
    raise error


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(
        self,
        quant_config,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del params_dtype
        base_weight_loader = extra_weight_attrs.pop("weight_loader")
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "weight_loader": partial(
                    _gguf_moe_weight_loader, layer, base_weight_loader
                ),
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight_type,
            {
                "weight_loader": _gguf_moe_weight_type_loader,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": 1,
                "ignore_warning": True,
            },
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        del layer
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        layer._dsv4_w1_repacked = False
        layer._dsv4_w2_repacked = False
        layer._dsv4_w2_output_sharded = getattr(layer, "_dsv4_w2_output_sharded", False)
        if current_platform.is_metal():
            # Load-time SoA repack for the Metal multi-row Q2_K MoE kernel:
            # per-expert [qs | scales | d,dmin] planes turn the 84-byte AoS
            # stride's three interleaved unaligned streams into dense aligned
            # planes (bit-exact; byte-neutral in size). IQ2_XXS stays AoS on
            # purpose: both the per-row plane split and the A100-style
            # gate/up pairing measured SLOWER here — Apple's LSU does not
            # penalize the unaligned narrow loads the A100 repack was built
            # to avoid, and the block scale rides free in the code stream
            # (optimization_status 2026-08-13).
            w2 = layer.w2_qweight
            if (
                layer.w2_qweight_type.weight_type == 10  # Q2_K
                and w2.dim() == 3
                and w2.shape[2] % 84 == 0
                and (w2.shape[1] * w2.shape[2]) % 8 == 0
            ):
                replace_parameter(
                    layer,
                    "w2_qweight",
                    _qc_metal_soa_repack(w2, 84, ((16, 64), (0, 16), (80, 4))),
                    prefer_copy=True,
                )
                layer._dsv4_w2_repacked = True
                # Drain the async permutation and release its transients
                # before anything else allocates: tens of GiB of enqueued
                # repack copies otherwise bleed into later boot phases and
                # leave the MPS allocator in a churned state.
                torch.mps.synchronize()
                torch.mps.empty_cache()
            return
        if current_platform.is_rocm():
            return
        # NOTE: MXFP4 experts stay in the raw AoS layout for now. The fused
        # MXFP4 path only covers decode widths (tokens <= 8); prefill still
        # runs the generic MMQ/MMVQ kernels, which read raw 17-byte blocks.
        # The SoA repack (ggml_dsv4_repack_mxfp4) is wired but must wait until
        # every consumer understands the split layout.
        if (
            layer.w13_qweight_type.weight_type == 39
            and layer.w2_qweight_type.weight_type == 39
            and _use_dsv4_ampere_mxfp4_repack()
            and layer.hidden_size % 256 == 0
            and layer.intermediate_size_per_partition % 256 == 0
        ):
            # Byte-neutral AoS(17) -> SoA(scales | aligned codes) split.
            # Measured 2.0x on the fused per-route GEMV at verify widths
            # (bit-identical outputs) and enables aligned tile loads.
            replace_parameter(
                layer,
                "w13_qweight",
                ops.ggml_dsv4_repack_mxfp4(layer.w13_qweight, layer.hidden_size),
                prefer_copy=True,
            )
            replace_parameter(
                layer,
                "w2_qweight",
                ops.ggml_dsv4_repack_mxfp4(
                    layer.w2_qweight, layer.intermediate_size_per_partition
                ),
                prefer_copy=True,
            )
            layer._dsv4_w1_repacked = True
            layer._dsv4_w2_repacked = True
            return
        if (
            not _use_dsv4_ampere_fused()
            or layer.w13_qweight_type.weight_type != 16
            or layer.w2_qweight_type.weight_type != 10
            # 512/1024 are the TP4/TP2 shards; 256 is the TP8 shard (added
            # 2026-08-10 -- without it every IQ2 layer fell back to the
            # generic route at TP8 and cost it the throughput matrix). 256
            # means single-superblock Q2_K down rows; the channel-owned and
            # cooperative variants gate themselves off at that shape.
            or layer.intermediate_size_per_partition not in (256, 512, 1024)
        ):
            return
        if _use_dsv4_ampere_iq2_repack():
            repacked_w1 = ops.ggml_dsv4_repack_iq2_xxs(
                layer.w13_qweight, layer.hidden_size
            )
            replace_parameter(layer, "w13_qweight", repacked_w1, prefer_copy=True)
            layer._dsv4_w1_repacked = True
        if _use_dsv4_ampere_q2k_repack():
            packed_block_bytes = 84
            w2_intermediate = layer.w2_qweight.shape[2] // packed_block_bytes * 256
            repacked_w2 = ops.ggml_dsv4_repack_q2_k(layer.w2_qweight, w2_intermediate)
            # Both layouts are byte-neutral. Copy back into the original
            # allocations so raw and repacked expert stacks never coexist.
            replace_parameter(layer, "w2_qweight", repacked_w2, prefer_copy=True)
            layer._dsv4_w2_repacked = True

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._apply(
            layer,
            x,
            None,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    def apply_prequant(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        quant_input: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._apply(
            layer,
            x,
            quant_input,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    def _apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        quant_input: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        from . import fused_moe_gguf as fused_moe_gguf_op

        return fused_moe_gguf_op(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
            layer.swiglu_limit,
            self.moe.activation_situ_beta,
            self.moe.activation_situ_linear_beta,
            layer.global_to_local_expert_map,
            getattr(layer, "_dsv4_w1_repacked", False),
            getattr(layer, "_dsv4_w2_repacked", False),
            quant_input,
            getattr(layer, "_dsv4_defer_down", False),
        )
