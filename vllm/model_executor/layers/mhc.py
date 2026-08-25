# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch

# this import will also register the custom ops
# import vllm.model_executor.kernels.mhc  # noqa: F401
import vllm.model_executor.kernels.mhc as mhc_kernels
from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang


def _has_tilelang_mhc() -> bool:
    # Decide against TileLang before probing for it: has_tilelang() performs a
    # trial import, and importing tilelang loads libhip_stub.so, which exports
    # the public HIP ABI (including hipGetDevicePropertiesR0600, backed by the
    # legacy R0000 implementation). That interposes on every HIP call resolved
    # afterwards -- e.g. aiter's ASM modules read warpSize as the clock rate.
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx942

        # TileLang MHC currently produces incorrect results on gfx942. Keep
        # gfx942 on the existing torch/triton fallbacks until that path is fixed.
        if on_gfx942():
            return False
    if not has_tilelang():
        return False
    if current_platform.is_cuda():
        return True
    return current_platform.is_rocm()


HAS_TILELANG_MHC = _has_tilelang_mhc()
HAS_AITER_MHC = is_aiter_found_and_supported()


def _has_quixicore_mhc() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        if torch.cuda.get_device_capability()[0] != 8:
            return False
        from vllm.quixicore import quixicore_ops

        return all(
            quixicore_ops.has(name)
            for name in (
                "dsv4_mhc_pre",
                "dsv4_mhc_fused_post_pre",
                "dsv4_mhc_post",
                "dsv4_hc_head",
            )
        )
    except (ImportError, RuntimeError):
        return False


HAS_QUIXICORE_MHC = _has_quixicore_mhc()
# The native fallback launches independent split-K work per token and supports
# the full SlimServe prefill quantum. Keeping this at decode width forced F16
# GGUF weights through the FP32 torch fallback during memory profiling/prefill.
_QUIXICORE_MHC_MAX_TOKENS = 2048


def _has_quixicore_mhc_metal() -> bool:
    if not current_platform.is_metal():
        return False
    # VLLM_METAL_MHC=0 restores the eager torch path (the fused kernels
    # replace an eager decomposition of thousands of tiny MPS ops per step).
    if os.environ.get("VLLM_METAL_MHC", "1").lower() not in ("1", "true", "on"):
        return False
    try:
        from vllm.quixicore import quixicore_ops

        return all(
            quixicore_ops.has(name)
            for name in (
                "dsv4_mhc_pre",
                "dsv4_mhc_fused_post_pre",
                "dsv4_mhc_post",
                "dsv4_hc_head",
            )
        )
    except (ImportError, RuntimeError):
        return False


HAS_QUIXICORE_MHC_METAL = _has_quixicore_mhc_metal()


# The Metal kernels re-read `fn` per token (one threadgroup per token) and
# scale by threadgroup count, so the cap defaults to the full prefill
# quantum. VLLM_QC_MHC_METAL_MAX_TOKENS overrides (32 restores decode-only
# behavior).
def _mhc_metal_max_tokens() -> int:
    value = os.environ.get("VLLM_QC_MHC_METAL_MAX_TOKENS", "2048")
    try:
        return int(value)
    except ValueError:
        return 2048


_QUIXICORE_MHC_METAL_MAX_TOKENS = _mhc_metal_max_tokens()


def _use_quixicore_mhc_metal(tensor: torch.Tensor) -> bool:
    return (
        HAS_QUIXICORE_MHC_METAL
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and tensor.shape[-2] == 4
        and tensor.numel() // (4 * tensor.shape[-1]) <= _QUIXICORE_MHC_METAL_MAX_TOKENS
    )


def _use_quixicore_mhc(tensor: torch.Tensor) -> bool:
    return (
        HAS_QUIXICORE_MHC
        and tensor.shape[-2] == 4
        and tensor.numel() // (4 * tensor.shape[-1]) <= _QUIXICORE_MHC_MAX_TOKENS
    )


def use_quixicore_mhc(tensor: torch.Tensor) -> bool:
    """Whether this concrete activation shape takes the native Ampere path."""
    return _use_quixicore_mhc(tensor)


# --8<-- [start:mhc_pre]
@CustomOp.register("mhc_pre")
class MHCPreOp(CustomOp):
    """MHC pre block.

    Computes mix logits from RMS-normalized HC residual streams, then
    returns post_mix, comb_mix, and
    layer_input = sum_i pre_mix_i * residual_i.
    """

    # --8<-- [end:mhc_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _use_quixicore_mhc(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            residual_flat = residual.view(-1, 4, hidden_size)
            post, comb, layer_input = quixicore_ops.dsv4_mhc_pre(
                residual_flat,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                norm_weight,
                norm_eps,
            )
            return (
                post.view(*outer_shape, 4, 1),
                comb.view(*outer_shape, 4, 4),
                layer_input.view(*outer_shape, hidden_size),
            )
        if not HAS_TILELANG_MHC:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        return torch.ops.vllm.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_size = residual.shape[-1]
        if HAS_AITER_MHC and hidden_size % 256 == 0:
            return torch.ops.vllm.mhc_pre_aiter(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
        elif HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_pre_tilelang(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )
        else:
            return self.forward_native(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                norm_weight,
                norm_eps,
            )

    def forward_native(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_kernels.mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    def forward_xpu(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._xpu_C.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    def forward_mps(
        self,
        residual: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if norm_weight is None and _use_quixicore_mhc_metal(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            post, comb, layer_input = quixicore_ops.dsv4_mhc_pre(
                residual.view(-1, 4, hidden_size).contiguous(),
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
            )
            return (
                post.view(*outer_shape, 4, 1),
                comb.view(*outer_shape, 4, 4),
                layer_input.view(*outer_shape, hidden_size),
            )
        return self.forward_native(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )


# --8<-- [start:mhc_post]
@CustomOp.register("mhc_post")
class MHCPostOp(CustomOp):
    """MHC post block.

    Combines the layer output with the HC residual streams:
    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i.
    """

    # --8<-- [end:mhc_post]

    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if _use_quixicore_mhc(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            output = quixicore_ops.dsv4_mhc_post(
                x.view(-1, hidden_size),
                residual.view(-1, 4, hidden_size),
                post_layer_mix.view(-1, 4).contiguous(),
                comb_res_mix.view(-1, 4, 4).contiguous(),
            )
            return output.view(*outer_shape, 4, hidden_size)
        if not HAS_TILELANG_MHC:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)
        return torch.ops.vllm.mhc_post_tilelang(
            x, residual, post_layer_mix, comb_res_mix
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        hidden_size = residual.shape[-1]
        if HAS_AITER_MHC and hidden_size % 256 == 0:
            return torch.ops.vllm.mhc_post_aiter(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
            )
        if HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_post_tilelang(
                x, residual, post_layer_mix, comb_res_mix
            )
        else:
            return self.forward_native(x, residual, post_layer_mix, comb_res_mix)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return mhc_kernels.mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops._xpu_C.mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    def forward_mps(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
    ) -> torch.Tensor:
        if _use_quixicore_mhc_metal(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            output = quixicore_ops.dsv4_mhc_post(
                x.reshape(-1, hidden_size).contiguous(),
                residual.view(-1, 4, hidden_size).contiguous(),
                post_layer_mix.reshape(-1, 4).float().contiguous(),
                comb_res_mix.reshape(-1, 4, 4).float().contiguous(),
            )
            return output.view(*outer_shape, 4, hidden_size)
        return self.forward_native(x, residual, post_layer_mix, comb_res_mix)


# --8<-- [start:hc_head]
@CustomOp.register("hc_head")
class HCHeadOp(CustomOp):
    """HC head reduction for DeepSeek V4.

    Computes gates from the RMS-normalized flattened HC residual and
    returns out = sum_i gate_i * residual_i, collapsing hc_mult streams
    to one.
    """

    # --8<-- [end:hc_head]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        if _use_quixicore_mhc(hidden_states):
            from vllm.quixicore import quixicore_ops

            out = quixicore_ops.dsv4_hc_head(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        elif HAS_TILELANG_MHC:
            out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        else:
            out = mhc_kernels.hc_head_fused_torch(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        return out.view(*outer_shape, hidden_size)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)

        if HAS_TILELANG_MHC:
            out = torch.ops.vllm.hc_head_fused_kernel_tilelang(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
        else:
            num_tokens = hs_flat.shape[0]
            out = torch.empty(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            torch.ops.vllm.hc_head_triton(
                hs_flat,
                hc_fn,
                hc_scale,
                hc_base,
                out,
                hidden_size,
                rms_norm_eps,
                hc_eps,
                hc_mult,
            )

        return out.view(*outer_shape, hidden_size)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        out = mhc_kernels.hc_head_fused_torch(
            hs_flat,
            hc_fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
        )
        return out.view(*outer_shape, hidden_size)

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        hc_mult, hidden_size = hidden_states.shape[-2:]
        outer_shape = hidden_states.shape[:-2]
        hs_flat = hidden_states.view(-1, hc_mult, hidden_size)
        num_tokens = hs_flat.shape[0]

        out = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
        )
        torch.ops._xpu_C.hc_head_fused(
            hs_flat, hc_fn, hc_scale, hc_base, out, rms_norm_eps, hc_eps
        )
        return out.view(*outer_shape, hidden_size)

    def forward_mps(
        self,
        hidden_states: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_norm_eps: float,
        hc_eps: float,
    ) -> torch.Tensor:
        if _use_quixicore_mhc_metal(hidden_states):
            from vllm.quixicore import quixicore_ops

            hc_mult, hidden_size = hidden_states.shape[-2:]
            outer_shape = hidden_states.shape[:-2]
            out = quixicore_ops.dsv4_hc_head(
                hidden_states.view(-1, hc_mult, hidden_size).contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                rms_norm_eps,
                hc_eps,
            )
            return out.view(*outer_shape, hidden_size)
        return self.forward_native(
            hidden_states, hc_fn, hc_scale, hc_base, rms_norm_eps, hc_eps
        )


# --8<-- [start:mhc_fused_post_pre]
@CustomOp.register("mhc_fused_post_pre")
class MHCFusedPostPreOp(CustomOp):
    """Fused MHC post block followed by the next MHC pre block.

    Equivalent to applying MHCPostOp and then MHCPreOp to the updated
    residual streams, returning residual_cur, post_mix_cur, comb_mix_cur,
    and layer_input_cur.
    """

    # --8<-- [end:mhc_fused_post_pre]
    @classmethod
    def enabled(cls) -> bool:
        return True

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if _use_quixicore_mhc(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            residual_cur, post_cur, comb_cur, layer_input_cur = (
                quixicore_ops.dsv4_mhc_fused_post_pre(
                    x.view(-1, hidden_size),
                    residual.view(-1, 4, hidden_size),
                    post_layer_mix.view(-1, 4).contiguous(),
                    comb_res_mix.view(-1, 4, 4).contiguous(),
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                    norm_weight,
                    norm_eps,
                )
            )
            return (
                residual_cur.view(*outer_shape, 4, hidden_size),
                post_cur.view(*outer_shape, 4, 1),
                comb_cur.view(*outer_shape, 4, 4),
                layer_input_cur.view(*outer_shape, hidden_size),
            )
        if not HAS_TILELANG_MHC:
            return self.forward_native(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )
        return torch.ops.vllm.mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if HAS_TILELANG_MHC:
            return torch.ops.vllm.mhc_fused_post_pre_tilelang(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                tile_n,
                norm_weight,
                norm_eps,
            )
        return self.forward_native(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Decompose into post + pre (no fused kernel available).
        residual_cur = mhc_kernels.mhc_post_torch(
            x, residual, post_layer_mix, comb_res_mix
        )
        post_mix_cur, comb_mix_cur, layer_input_cur = mhc_kernels.mhc_pre_torch(
            residual_cur,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
        return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._xpu_C.mhc_fused_post_pre(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    def forward_mps(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_layer_mix: torch.Tensor,
        comb_res_mix: torch.Tensor,
        fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 1,
        tile_n: int = 1,
        norm_weight: torch.Tensor | None = None,
        norm_eps: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if norm_weight is None and _use_quixicore_mhc_metal(residual):
            from vllm.quixicore import quixicore_ops

            outer_shape = residual.shape[:-2]
            hidden_size = residual.shape[-1]
            residual_cur, post_cur, comb_cur, layer_input_cur = (
                quixicore_ops.dsv4_mhc_fused_post_pre(
                    x.reshape(-1, hidden_size).contiguous(),
                    residual.view(-1, 4, hidden_size).contiguous(),
                    post_layer_mix.reshape(-1, 4).float().contiguous(),
                    comb_res_mix.reshape(-1, 4, 4).float().contiguous(),
                    fn,
                    hc_scale,
                    hc_base,
                    rms_eps,
                    hc_pre_eps,
                    hc_sinkhorn_eps,
                    hc_post_mult_value,
                    sinkhorn_repeat,
                )
            )
            return (
                residual_cur.view(*outer_shape, 4, hidden_size),
                post_cur.view(*outer_shape, 4, 1),
                comb_cur.view(*outer_shape, 4, 4),
                layer_input_cur.view(*outer_shape, hidden_size),
            )
        return self.forward_native(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            tile_n,
            norm_weight,
            norm_eps,
        )
