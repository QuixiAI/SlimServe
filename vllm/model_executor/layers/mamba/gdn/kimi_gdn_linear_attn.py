# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn.parameter import Parameter

import vllm.model_executor.layers.mamba.ops.kda_gate_projection  # noqa: F401
from vllm import envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.metal_phaseprof import phase as _qc_phase

from ...linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from ..mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from ..ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from ..ops.gather_initial_states import gather_initial_states

# Empirical lower bound for the KDA gate to avoid numerical underflow.
_KDA_GATE_LOGBOUND_MIN = -5.0


def _apply_kda_output_norm(
    norm: FusedRMSNormGated, output: torch.Tensor, gate: torch.Tensor
) -> None:
    # This runs INSIDE the opaque kda_attention op. CustomOp's default
    # forward_native dispatch assumes enclosing torch.compile can fuse its
    # PyTorch operations, but that compiler cannot see this body. Use the
    # CUDA kernel explicitly instead of launching casts, reductions, sigmoid
    # and multiplies separately at every KDA layer. Its output aliases a
    # contiguous input; copy_ is then a no-op (and handles noncontiguous input).
    if current_platform.is_cuda():
        output.copy_(norm.forward_cuda(output, gate))
    else:
        output.copy_(norm(output, gate))


def _materialize_kda_gate_and_beta(
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_input = raw_g.float()
    if dt_bias is not None:
        gate_input = gate_input + dt_bias.float().view(
            1, 1, raw_g.shape[2], raw_g.shape[3]
        )
    decay = a_log.float().exp().view(1, 1, -1, 1)
    if lower_bound is not None:
        gate = lower_bound * torch.sigmoid(decay * gate_input)
    else:
        gate = -decay * F.softplus(gate_input)
    return gate, raw_beta.float().sigmoid()


def _use_recurrent_kda_prefill() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


_KDA_FUSED: bool | None = None


logger = init_logger(__name__)
_KDA_DEBUG = os.environ.get("VLLM_QC_KDA_DEBUG") == "1"
_kda_branch_counts: dict[str, int] = {}


def _kda_branch_log(tag: str, num_tokens: int, m) -> None:
    """Campaign diagnostic (VLLM_QC_KDA_DEBUG=1): which KDA core branch each
    layer call takes, with the batch composition. Logs the first 3 hits per
    branch and every 1000th after."""
    n = _kda_branch_counts.get(tag, 0) + 1
    _kda_branch_counts[tag] = n
    if n <= 3 or n % 1000 == 0:
        logger.info(
            "[kda-branch] %s n=%d T=%d spec=%s prefills=%s decodes=%s spec_dec=%s",
            tag, n, num_tokens,
            m.spec_sequence_masks is not None,
            getattr(m, "num_prefills", None), getattr(m, "num_decodes", None),
            getattr(m, "num_spec_decodes", None),
        )


_KDA_SPEC: bool | None = None


def _kda_metal_spec_available() -> bool:
    """The Metal speculative-verify KDA kernels (conv rewind mode +
    kda_recur_spec) are present; ``VLLM_METAL_KDA_SPEC=0`` pins the
    torch-native spec path."""
    global _KDA_SPEC
    if _KDA_SPEC is None:
        _KDA_SPEC = False
        if _kda_metal_fused_available() and os.environ.get(
            "VLLM_METAL_KDA_SPEC", "1"
        ) != "0":
            try:
                from vllm.quixicore import quixicore_ops

                _KDA_SPEC = quixicore_ops.has_kernel("kda_recur_spec_d128")
            except Exception:
                _KDA_SPEC = False
    return _KDA_SPEC


def _kda_metal_fused_available() -> bool:
    """The Metal ``kda_step`` kernels (conv + gate + per-channel delta rule
    + gated norm) are present and not disabled. ``VLLM_METAL_KDA_FUSED=0``
    pins the torch-native reference path for A/B and parity work."""
    global _KDA_FUSED
    if _KDA_FUSED is None:
        if os.environ.get("VLLM_METAL_KDA_FUSED", "1") == "0":
            _KDA_FUSED = False
        else:
            try:
                from vllm.platforms import current_platform
                from vllm.quixicore import quixicore_ops

                _KDA_FUSED = (
                    current_platform.is_metal()
                    and quixicore_ops.has("kda_step")
                    and quixicore_ops.has_kernel("kda_recur_d128")
                )
            except Exception:
                _KDA_FUSED = False
    return _KDA_FUSED


def _kda_native_path_default() -> bool:
    """Whether the torch-native KDA core (kda_mps_fallback.py) serves this
    platform by default. Mirrors the Qwen GDN layer's VLLM_METAL_GDN
    switch: Metal/MPS and CPU (no Triton) take the native path;
    CUDA/ROCm keep their Triton kernels byte-for-byte.

    VLLM_METAL_KDA=0 pins the Triton route everywhere (a kill switch: on a
    Triton-less platform the layer then fails at the kernel import, as it
    did before the native path existed); VLLM_METAL_KDA=native forces the
    native core on every platform (the CUDA/ROCm A/B oracle).
    """
    flag = os.environ.get("VLLM_METAL_KDA", "1").strip().lower()
    if flag == "0":
        return False
    if flag == "native":
        return True
    return not (current_platform.is_cuda() or current_platform.is_rocm())


def _get_kda_kernels() -> tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    if current_platform.is_rocm():
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            chunk_kda_with_fused_gate as amd_chunk_kda,
        )
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            fused_recurrent_kda as amd_recurrent_kda,
        )
        from vllm.models.kimi_k3.amd.ops.third_party.kda import (
            fused_recurrent_kda_packed_decode as amd_packed_decode,
        )

        return amd_chunk_kda, amd_recurrent_kda, amd_packed_decode

    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate as nvidia_chunk_kda,
    )
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        fused_recurrent_kda as nvidia_recurrent_kda,
    )
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode as nvidia_packed_decode,
    )

    return nvidia_chunk_kda, nvidia_recurrent_kda, nvidia_packed_decode


def a_log_weight_loader(
    shard_axis: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load KDA A_log stored as either old 4D or current 1D weights."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if loaded_weight.dim() == 4:
            assert loaded_weight.shape[:2] == (1, 1), (
                f"Expected old A_log shape (1, 1, H, 1), got {loaded_weight.shape}"
            )
            assert loaded_weight.shape[-1] == 1, (
                f"Expected old A_log last dim to be 1, got {loaded_weight.shape}"
            )
            loaded_weight = loaded_weight.view(loaded_weight.shape[2])

        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)

    return loader


def _make_fused_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
) -> Callable[..., None]:
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)

    return weight_loader


_METAL_GATE_DUAL = (
    current_platform.is_metal()
    and os.environ.get("VLLM_METAL_KDA_GATE_DUAL", "0") == "1"
)


def _metal_gate_dual_ok(layer, f_a: torch.Tensor) -> bool:
    """The two K=128 gate projections (f_b, g_b: GGUF q8_0 [8192, 128]) can
    ride one dual dispatch: Metal, decode widths (<= 4 rows), both layers
    q8_0. Opt-in per profile (VLLM_METAL_KDA_GATE_DUAL=1, glm53f-q2-1)."""
    if f_a.dim() != 2 or not (1 <= f_a.shape[0] <= 4):
        return False
    for proj in (layer.f_b_proj, layer.g_b_proj):
        qt = getattr(proj, "qweight_type", None)
        if qt is None or getattr(qt, "weight_type", None) != 8:
            return False
        if getattr(proj, "qweight", None) is None:
            return False
    return True


def _metal_gate_dual(layer, f_a: torch.Tensor, g_a: torch.Tensor):
    if not (_METAL_GATE_DUAL and _metal_gate_dual_ok(layer, f_a)):
        return None
    from vllm.quixicore.ops import quixicore_ops

    wf, wg = layer.f_b_proj.qweight, layer.g_b_proj.qweight
    return quixicore_ops.ggml_mul_mat_vec_a8_dual(
        wf, f_a.contiguous(), wg, g_a.contiguous(), 8, wf.shape[0], wg.shape[0]
    )


class _KimiGDNMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with selected outputs replicated across TP ranks.

    The replicated shard is represented as ``size * tp_size`` so the merged
    parameter reserves ``size`` local rows on every rank. Loading that shard
    from rank zero then gives every rank the complete checkpoint weight.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_id: int | tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = (
            (replicated_shard_id,)
            if isinstance(replicated_shard_id, int)
            else replicated_shard_id
        )
        output_sizes = output_sizes.copy()
        for shard_id in self.replicated_shard_ids:
            output_sizes[shard_id] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)
        # Metal GGUF: adjacent same-type shards of this hetero-quant
        # projection ride one GEMV and write column slices of one output
        # (gguf/linear.py); scoped to this layer, not to GGUF linears at large.
        self.qc_metal_fused_shards = True

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@PluggableLayer.register("kimi_gated_delta_net_attention")
class KimiGatedDeltaNetAttention(GatedDeltaNetAttention):
    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
        fuse_gate_a: bool = False,
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        self.use_full_rank_gate = kda_config.get("use_full_rank_gate", False)
        # GLM's low-rank gate consumes the same hidden input as Q/K/V and
        # f_a. Load its unchanged BF16 rows into the existing projection to
        # avoid a separate narrow GEMV at every KDA layer. Other models keep
        # their existing module/loader contract unless explicitly opted in.
        self.fuse_gate_a = fuse_gate_a and not self.use_full_rank_gate

        if self.use_full_rank_gate:
            # Keep f_a before the narrow beta shard, then pad each TP-local row
            # to select the aligned BF16 GEMM path. The padding also avoids an
            # Inductor correctness issue seen with the row-strided G view.
            qkvg_output_sizes = [self.projection_size] * 4
            in_proj_output_sizes = qkvg_output_sizes + [
                self.head_dim,
                self.num_heads,
            ]
            local_output_size = (
                4 * self.local_projection_size + self.head_dim + self.local_num_heads
            )
            self.in_proj_padding = -local_output_size % 16
            if self.in_proj_padding:
                in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        else:
            in_proj_output_sizes = [self.projection_size] * 3 + [
                self.num_heads,
                self.head_dim,
            ]
            self.in_proj_padding = 0
            if self.fuse_gate_a:
                in_proj_output_sizes.append(self.head_dim)
        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=(4, 5) if self.fuse_gate_a else 4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache let decode run a single conv update.
        # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_fused_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                ),
                # Checkpoints ship this convolution either as three tensors
                # (q/k/v_conv1d, stacked here by shard id) or pre-fused as one
                # tensor of 3 x projection_size rows. A loader needs the count
                # to split the pre-fused form back into its shards.
                "fused_conv1d_shards": 3,
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )
        self.use_safe_gate = self.gate_lower_bound is not None
        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        # "native" is the torch-native core (kda_mps_fallback.py): the
        # default off CUDA/ROCm, selectable anywhere for A/B.
        self.use_native_kda = backend == "native" or (
            backend == "auto" and _kda_native_path_default()
        )
        if backend == "auto":
            backend = "native" if self.use_native_kda else "triton"
        assert backend in ("triton", "native"), (
            "The shared Kimi GDN layer only supports the 'triton' and "
            f"'native' KDA prefill backends, got {backend!r}."
        )
        if not self.use_full_rank_gate:
            if not self.fuse_gate_a:
                self.g_a_proj = ReplicatedLinear(
                    self.hidden_size,
                    self.head_dim,
                    bias=False,
                    quant_config=self.quant_config,
                    prefix=f"{prefix}.g_a_proj",
                )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=False,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.use_paired_gate_projection = (
            self.fuse_gate_a
            and current_platform.is_cuda()
            and not envs.VLLM_BATCH_INVARIANT
            and self.head_dim == 128
            and isinstance(self.f_b_proj.quant_method, UnquantizedLinearMethod)
            and isinstance(self.g_b_proj.quant_method, UnquantizedLinearMethod)
            and self.f_b_proj.weight.dtype == torch.bfloat16
            and self.g_b_proj.weight.dtype == torch.bfloat16
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = mixed_qkv.shape[0]
        qkv = mixed_qkv.view(seq_len, 3, self.local_num_heads, self.head_dim)
        # Materialize all three row-strided inputs with one token-major to
        # QKV-major permutation. Each unbound tensor is then contiguous.
        qkv = qkv.permute(1, 0, 2, 3).contiguous().unsqueeze(1)
        return qkv.unbind(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor | None,
        projected_qkvgfab: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """``output`` None returns the o_proj result directly (one fewer
        copy per layer); otherwise it is written in place."""
        num_tokens = hidden_states.size(0)
        if projected_qkvgfab is None:
            with _qc_phase("kda_in_proj"):
                projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        else:
            assert projected_qkvgfab.shape[0] == num_tokens
            assert projected_qkvgfab.dtype == hidden_states.dtype
        g1 = None
        if self.use_full_rank_gate:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_projection_size,
                self.head_dim,
                self.local_num_heads,
            ]
            if self.in_proj_padding:
                split_sizes.append(self.in_proj_padding)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, g_proj_states, f_a, beta = projected[:4]
        else:
            split_sizes = [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
            ]
            if self.fuse_gate_a:
                split_sizes.append(self.head_dim)
            projected = projected_qkvgfab.split(split_sizes, dim=-1)
            mixed_qkv, beta, f_a = projected[:3]
            g_a = projected[3] if self.fuse_gate_a else self.g_a_proj(hidden_states)[0]
            if self.use_paired_gate_projection:
                with _qc_phase("kda_gate_proj"):
                    g1, g_proj_states = torch.ops.vllm.kda_gate_pair(
                        f_a, g_a, self.f_b_proj.weight, self.g_b_proj.weight
                    )
            else:
                with _qc_phase("kda_gate_proj"):
                    # Metal (opt-in): f_b and g_b in one dispatch.
                    dual = _metal_gate_dual(self, f_a, g_a)
                if dual is not None:
                    g1, g_proj_states = dual
                else:
                    g_proj_states = self.g_b_proj(g_a)[0]

        if not self.use_paired_gate_projection and g1 is None:
            with _qc_phase("kda_gate_proj"):
                g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)

        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Opaque to torch.compile: the core reads the forward context
        # (attention metadata, state indices) that Dynamo would otherwise
        # bake in from the trace-time run. "vllm::kda_attention" is a
        # splitting op (CompilationConfig._attention_ops), matching the
        # unified-attention treatment.
        with _qc_phase("kda_core"):
            torch.ops.vllm.kda_attention(
                mixed_qkv, g1, g2, beta, core_attn_out, self.prefix
            )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        with _qc_phase("kda_o_proj"):
            if output is None:
                return self.o_proj(core_attn_out)[0]
            output[:] = self.o_proj(core_attn_out)[0]
        return None

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if self.use_native_kda:
            # Metal/MPS/CPU (or VLLM_METAL_KDA=native): torch-native core,
            # no Triton anywhere below this point.
            self._forward_native(
                mixed_qkv, g1, g2, beta, core_attn_out, attn_metadata_raw
            )
            return

        if attn_metadata_raw is None:
            return

        # Vendor-specific KDA kernels: AMD/ROCm and NVIDIA keep their own copies
        # under kimi_k3/{amd,nvidia}/ops so each can diverge independently.
        (
            chunk_kda_with_fused_gate,
            fused_recurrent_kda,
            fused_recurrent_kda_packed_decode,
        ) = _get_kda_kernels()

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        m = attn_metadata_narrowed
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_sequence_masks = m.spec_sequence_masks
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        constant_caches = self.kv_cache

        conv_state, recurrent_state = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Split tokens into the multi-query spec-decode part and the remaining
        # (prefill / plain decode) part.
        if spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

        # ---------- spec-decode multi-query path ----------
        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)

            # Sibling beta and, for full-rank gates, output-gate views remain
            # live, so write the convolution output separately.
            spec_conv_out = torch.empty(
                mixed_qkv_spec.shape,
                dtype=mixed_qkv_spec.dtype,
                device=mixed_qkv_spec.device,
            )
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            # Spec-only batches write directly into core_attn_out.
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
            core_attn_out_spec, _ = fused_recurrent_kda(
                q=q_spec,
                k=k_spec,
                v=v_spec,
                raw_g=g1_spec,
                raw_beta=beta_spec,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                cu_seqlens=spec_cu_seqlens,
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                out=spec_out,
            )

        # ---------- non-spec path (prefill or plain decode) ----------
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                # Packed prefill conv would require copying V solely to make
                # it dense for KDA. Separate calls accept the strided inputs
                # and produce dense Q/K/V without that extra traffic.
                # TODO: Use packed conv once every KDA prefill backend accepts
                # row-strided Q/K/V directly.
                def _prefill_conv(
                    x: torch.Tensor,
                    state: torch.Tensor,
                    weight: torch.Tensor,
                ) -> torch.Tensor:
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )

                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                if _use_recurrent_kda_prefill():
                    from vllm.third_party.flash_linear_attention.ops.kda import (
                        fused_recurrent_kda as recurrent_kda_prefill,
                    )

                    new_state_indices = non_spec_state_indices_tensor[
                        ~has_initial_state
                    ]
                    recurrent_state.index_fill_(0, new_state_indices.long(), 0)
                    gate, recurrent_beta = _materialize_kda_gate_and_beta(
                        g1_ns,
                        beta_ns,
                        self.A_log,
                        self.dt_bias,
                        self.gate_lower_bound,
                    )
                    state_indices = non_spec_state_indices_tensor[:, None].expand(
                        -1, q_ns.shape[1]
                    )
                    core_attn_out_non_spec, _ = recurrent_kda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=gate,
                        beta=recurrent_beta,
                        initial_state=recurrent_state,
                        inplace_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                        ssm_state_indices=state_indices,
                    )
                else:
                    initial_state = gather_initial_states(
                        recurrent_state,
                        non_spec_state_indices_tensor,
                        has_initial_state,
                    )
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                    recurrent_state[non_spec_state_indices_tensor] = (
                        last_recurrent_state
                    )

            else:
                # pure-decode non-spec batch
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                # Sibling beta and, for full-rank gates, output-gate views
                # remain live, so write the conv output separately.
                packed_conv_out = torch.empty(
                    mixed_qkv_ns.shape,
                    dtype=mixed_qkv_ns.dtype,
                    device=mixed_qkv_ns.device,
                )
                mixed_qkv_ns = causal_conv1d_update(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                    out=packed_conv_out,
                )
                core_attn_out_non_spec, _ = fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                )

        # ---------- merge spec and non-spec outputs ----------
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            # Mixed batches require indexed placement in the original order.
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0, :num_actual_tokens]
        elif core_attn_out_non_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_spec is not None
        _apply_kda_output_norm(self.o_norm, core_attn_out, g2)

    def _kda_metal_fused_step(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        slot_mapping: torch.Tensor,
        g2: torch.Tensor,
        out: torch.Tensor | None = None,
        slot_table: torch.Tensor | None = None,
        num_accepted: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One ``kda_step`` launch set over the non-spec rows (or, with
        ``slot_table`` + ``num_accepted``, the speculative-verify rows);
        returns the o_norm'd output ``[T, H*Dv]``."""
        from vllm.quixicore import quixicore_ops

        T = mixed_qkv.size(0)
        cache = self.__dict__.setdefault("_kda_fused_cache", {})
        conv_w = cache.get("conv_w")
        if conv_w is None:
            conv_w = (
                self.conv1d.weight.view(
                    self.conv1d.weight.size(0), self.conv1d.weight.size(2)
                )
                .float()
                .contiguous()
            )
            cache["conv_w"] = conv_w
            cache["A_log"] = self.A_log.float().reshape(-1).contiguous()
            cache["dt_bias"] = (
                self.dt_bias.float().reshape(-1).contiguous()
                if self.dt_bias is not None
                else None
            )
            cache["norm_w"] = (
                self.o_norm.weight.to(mixed_qkv.dtype).reshape(-1).contiguous()
            )
        # conv_state arrives as the (dim, L)-per-slot view the caller already
        # oriented (SD pools are transposed in _forward_native); the kernel
        # takes the two inner strides, so either physical layout works.
        return quixicore_ops.kda_step(
            mixed_qkv,
            g1.reshape(T, -1),
            beta.reshape(T, -1),
            conv_w,
            conv_state,
            recurrent_state,
            cu_seqlens,
            slot_mapping,
            cache["A_log"],
            cache["dt_bias"],
            0.0 if self.gate_lower_bound is None else float(self.gate_lower_bound),
            True,
            cache["norm_w"],
            g2.reshape(T, -1),
            float(self.o_norm.eps),
            float(self.head_dim) ** -0.5,
            out=out,
            slot_table=slot_table,
            num_accepted=num_accepted,
        )

    def _forward_native(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata_raw: Any,
    ) -> None:
        """Torch-native KDA core (Metal/MPS/CPU), filling ``core_attn_out``
        in place; the counterpart of the Triton body of ``_forward`` with
        the same batch split (spec verify / varlen prefill / plain decode),
        the same state indexing contract and the same merge order. See
        kda_mps_fallback.py for the numerics contract.
        """
        if attn_metadata_raw is None:
            # Profile/warmup run: no Triton autotuner to prime and no state
            # to touch; hand o_proj a finite container.
            core_attn_out.zero_()
            return

        from vllm.model_executor.layers.mamba.gdn.kda_mps_fallback import (
            KdaVarlenPlan,
            kda_conv_prefill_native,
            kda_conv_spec_update_native,
            kda_conv_update_native,
            kda_recurrent_decode_native,
            kda_recurrent_prefill_native,
            kda_recurrent_spec_native,
        )

        assert isinstance(attn_metadata_raw, dict)
        m = attn_metadata_raw[self.prefix]
        assert isinstance(m, GDNAttentionMetadata)
        num_actual_tokens = m.num_actual_tokens
        if num_actual_tokens < core_attn_out.shape[1]:
            # Padded rows: keep them finite (their o_proj output is dropped).
            core_attn_out[:, num_actual_tokens:].zero_()
        if num_actual_tokens == 0:
            return
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state = self.kv_cache
        # conv_state must be (..., dim, width-1[+num_spec]) for the conv
        # fallbacks. DS layout stores it that way; SD needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        conv_bias = self.conv1d.bias
        num_heads, head_dim = self.local_num_heads, self.head_dim
        proj = self.local_projection_size

        def split_heads(
            packed: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return tuple(
                x.reshape(1, -1, num_heads, head_dim)
                for x in packed.split(proj, dim=-1)
            )

        spec_sequence_masks = m.spec_sequence_masks
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        if spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec, g1_spec, beta_spec = mixed_qkv, g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

        fused_ok = (
            mixed_qkv.device.type == "mps"
            and head_dim == 128
            and getattr(self, "_kda_metal_fused_step", None) is not None
        )

        def _spec_fused_cache():
            """Spec-verify kernel inputs over the spec rows (cached on the
            shared metadata object: every KDA layer reuses it)."""
            num_spec = m.num_spec_decodes
            cache = getattr(m, "_kda_spec_cache", None)
            if cache is None:
                from types import SimpleNamespace

                table = m.spec_state_indices_tensor[:num_spec]
                if table.dtype != torch.int32:
                    table = table.to(torch.int32)
                table = table.contiguous()
                accepted = m.num_accepted_tokens[:num_spec]
                if accepted.dtype != torch.int32:
                    accepted = accepted.to(torch.int32)
                cu = m.spec_query_start_loc[: num_spec + 1]
                if cu.dtype != torch.int32:
                    cu = cu.to(torch.int32)
                cache = SimpleNamespace(
                    table=table,
                    conv_slots=table[:, 0].contiguous(),
                    num_accepted=accepted.contiguous(),
                    cu=cu.contiguous(),
                )
                m._kda_spec_cache = cache  # type: ignore[attr-defined]
            return cache

        def _ns_fused_args() -> tuple[torch.Tensor, torch.Tensor]:
            """(cu_seqlens, slot_mapping) for the fused step over the
            non-spec rows; zeroes the pool rows of fresh prefills first."""
            assert mixed_qkv_ns is not None
            non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
            assert non_spec_state_indices_tensor is not None
            T_ns = mixed_qkv_ns.size(0)
            if m.num_prefills > 0:
                assert m.non_spec_query_start_loc is not None
                num_ns = m.num_prefills + m.num_decodes
                cu = m.non_spec_query_start_loc[: num_ns + 1]
                slots = non_spec_state_indices_tensor[:num_ns]
                his = m.has_initial_state
                # Same contract as the Triton prefill path: a fresh prefill
                # must not resume from a stale pool row.
                assert his is not None, "prefill requires has_initial_state"
                # Fresh prefills start from S = 0 / empty ring: zero their
                # pool rows once, then every request loads its state.
                fresh = (~his[:num_ns].to(torch.bool)) & (slots > 0)
                # No host sync (`fresh.any()` cost a 23 ms pipeline
                # drain per layer per prefill step): scale the rows of
                # every non-spec request by keep = 0/1 instead.
                rows = slots.clamp(min=0).to(torch.long)
                keep = (~fresh).to(recurrent_state.dtype)
                recurrent_state[rows] = recurrent_state[rows] * keep.view(
                    -1, 1, 1, 1
                )
                conv_state[rows] = conv_state[rows] * keep.to(
                    conv_state.dtype
                ).view(-1, 1, 1)
            else:
                num_ns = T_ns
                # Decode rows: one shared [0..T] cu per step (the metadata
                # object is shared by every KDA layer; 34 aranges otherwise).
                cu = getattr(m, "_kda_decode_cu", None)
                if cu is None or cu.numel() != T_ns + 1:
                    cu = torch.arange(
                        T_ns + 1, dtype=torch.int32, device=mixed_qkv_ns.device
                    )
                    m._kda_decode_cu = cu  # type: ignore[attr-defined]
                slots = non_spec_state_indices_tensor[:T_ns]
            if cu.dtype != torch.int32:
                cu = cu.to(torch.int32)
            if slots.dtype != torch.int32:
                slots = slots.to(torch.int32)
            return cu, slots

        # ---------- spec-decode multi-query path ----------
        if (
            spec_sequence_masks is not None
            and m.num_prefills == 0
            and m.num_decodes == 0
            and fused_ok
            and _kda_metal_spec_available()
        ):
            if _KDA_DEBUG:
                _kda_branch_log("spec_fused", mixed_qkv.shape[0], m)
            # Metal fused verify step (conv rewind + checkpointing delta rule
            # + gated norm): every row is a spec row, so the kernel writes
            # the o_norm'd result straight into the attention slab. Mixed
            # spec/prefill batches stay on the torch path below.
            cache = _spec_fused_cache()
            g2_rows = g2.reshape(-1, num_heads * head_dim)[:num_actual_tokens]
            out_rows = core_attn_out[0, :num_actual_tokens].view(
                num_actual_tokens, num_heads * head_dim
            )
            if out_rows.is_contiguous():
                self._kda_metal_fused_step(
                    mixed_qkv, g1, beta, conv_state, recurrent_state, cache.cu,
                    cache.conv_slots, g2_rows, out_rows,
                    slot_table=cache.table, num_accepted=cache.num_accepted,
                )
            else:
                fused = self._kda_metal_fused_step(
                    mixed_qkv, g1, beta, conv_state, recurrent_state, cache.cu,
                    cache.conv_slots, g2_rows,
                    slot_table=cache.table, num_accepted=cache.num_accepted,
                )
                core_attn_out[0, :num_actual_tokens] = fused.view(
                    num_actual_tokens, num_heads, head_dim
                )
            return
        # ---------- mixed spec + non-spec batch: both subsets fused ----------
        if (
            spec_sequence_masks is not None
            and mixed_qkv_ns is not None
            and fused_ok
            and _kda_metal_spec_available()
            and _kda_metal_fused_available()
        ):
            if _KDA_DEBUG:
                _kda_branch_log("mixed_fused", mixed_qkv.shape[0], m)
            # A request arriving while others decode makes every step of its
            # prefill a mixed batch. Each subset takes its own fused kernel
            # over its own rows (the verify step over the spec rows, the
            # conv+recurrence step over the prefill/decode rows); both write
            # o_norm'd rows, scattered back by token index, and their pool
            # slots are disjoint sequences. Before this, a mixed batch sent
            # BOTH subsets to the torch reference, whose prefill is a
            # per-token loop: 34 layers x a 4288-token chunk per step, the
            # whole concurrent-prefill collapse (2026-09-17).
            assert g1_spec is not None and beta_spec is not None
            assert g1_ns is not None and beta_ns is not None
            cache = _spec_fused_cache()
            cu, slots = _ns_fused_args()
            g2_flat = g2.reshape(-1, num_heads * head_dim)[:num_actual_tokens]
            out_spec = self._kda_metal_fused_step(
                mixed_qkv_spec, g1_spec, beta_spec, conv_state, recurrent_state,
                cache.cu, cache.conv_slots,
                g2_flat.index_select(0, spec_token_indx),
                slot_table=cache.table, num_accepted=cache.num_accepted,
            )
            out_ns = self._kda_metal_fused_step(
                mixed_qkv_ns, g1_ns, beta_ns, conv_state, recurrent_state,
                cu.contiguous(), slots.contiguous(),
                g2_flat.index_select(0, non_spec_token_indx),
            )
            merged = torch.empty(
                (num_actual_tokens, num_heads * head_dim),
                dtype=core_attn_out.dtype,
                device=core_attn_out.device,
            )
            merged.index_copy_(0, spec_token_indx, out_spec.to(merged.dtype))
            merged.index_copy_(0, non_spec_token_indx, out_ns.to(merged.dtype))
            core_attn_out[0, :num_actual_tokens] = merged.view(
                num_actual_tokens, num_heads, head_dim
            )
            return
        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            spec_state_indices_tensor = m.spec_state_indices_tensor
            spec_query_start_loc = m.spec_query_start_loc
            num_accepted_tokens = m.num_accepted_tokens
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            assert num_accepted_tokens is not None
            num_spec = m.num_spec_decodes
            spec_rows = spec_state_indices_tensor[:num_spec]
            spec_max_query_len = spec_state_indices_tensor.size(-1)
            spec_cu_seqlens = spec_query_start_loc[: num_spec + 1]
            conv_out_spec = kda_conv_spec_update_native(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                "silu",
                spec_rows[:, 0],
                num_accepted_tokens,
                spec_cu_seqlens,
                spec_max_query_len,
            )
            q_spec, k_spec, v_spec = split_heads(conv_out_spec)
            if _KDA_DEBUG:
                _kda_branch_log("spec_torch", mixed_qkv.shape[0], m)
            core_attn_out_spec = kda_recurrent_spec_native(
                q_spec,
                k_spec,
                v_spec,
                g1_spec,
                beta_spec,
                self.A_log,
                self.dt_bias,
                self.gate_lower_bound,
                recurrent_state,
                spec_rows,
                num_accepted_tokens,
                spec_cu_seqlens,
                spec_max_query_len,
            )

        # ---------- non-spec path (prefill or plain decode) ----------
        core_attn_out_non_spec = None
        if (
            mixed_qkv_ns is not None
            and spec_sequence_masks is None
            and fused_ok
            and _kda_metal_fused_available()
        ):
            if _KDA_DEBUG:
                _kda_branch_log("nonspec_fused", mixed_qkv.shape[0], m)
            # Metal fused step (conv + gate + per-channel delta rule + gated
            # norm in one command buffer). Covers plain decode and varlen
            # prefill; the pool is written in place. Output is already
            # o_norm'd, so the trailing forward_native is skipped.
            assert g1_ns is not None and beta_ns is not None
            T_ns = mixed_qkv_ns.size(0)
            cu, slots = _ns_fused_args()
            g2_ns = g2.reshape(-1, num_heads * head_dim)[:num_actual_tokens]
            # The kernel writes the o_norm'd rows straight into the
            # attention output slab (no per-layer copy).
            out_rows = core_attn_out[0, :num_actual_tokens].view(
                num_actual_tokens, num_heads * head_dim
            )
            if T_ns == num_actual_tokens and out_rows.is_contiguous():
                self._kda_metal_fused_step(
                    mixed_qkv_ns, g1_ns, beta_ns, conv_state, recurrent_state,
                    cu.contiguous(), slots.contiguous(), g2_ns, out_rows,
                )
            else:
                fused = self._kda_metal_fused_step(
                    mixed_qkv_ns, g1_ns, beta_ns, conv_state, recurrent_state,
                    cu.contiguous(), slots.contiguous(), g2_ns,
                )
                core_attn_out[0, :num_actual_tokens] = fused.view(
                    T_ns, num_heads, head_dim
                )
            return
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
            assert non_spec_state_indices_tensor is not None
            if m.num_prefills > 0:
                # Decodes ride along as length-1 varlen rows, as on CUDA.
                assert m.non_spec_query_start_loc is not None
                plan = getattr(m, "_kda_native_plan", None)
                if plan is None:
                    # One host copy of the varlen layout per step (the
                    # metadata object is shared by every KDA layer; prefill
                    # steps never hit the steady-metadata reuse path).
                    num_ns = m.num_prefills + m.num_decodes
                    plan = KdaVarlenPlan.build(
                        m.non_spec_query_start_loc[: num_ns + 1],
                        non_spec_state_indices_tensor,
                        m.has_initial_state,
                    )
                    m._kda_native_plan = plan  # type: ignore[attr-defined]
                conv_out_ns = kda_conv_prefill_native(
                    mixed_qkv_ns, conv_state, conv_weights, conv_bias, "silu", plan
                )
                q_ns, k_ns, v_ns = split_heads(conv_out_ns)
                if _KDA_DEBUG:
                    _kda_branch_log("nonspec_torch_prefill", mixed_qkv.shape[0], m)
                core_attn_out_non_spec = kda_recurrent_prefill_native(
                    q_ns,
                    k_ns,
                    v_ns,
                    g1_ns,
                    beta_ns,
                    self.A_log,
                    self.dt_bias,
                    self.gate_lower_bound,
                    recurrent_state,
                    plan,
                )
            else:
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                conv_out_ns = kda_conv_update_native(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    conv_bias,
                    "silu",
                    decode_conv_indices,
                )
                if _KDA_DEBUG:
                    _kda_branch_log("nonspec_torch_decode", mixed_qkv.shape[0], m)
                core_attn_out_non_spec = kda_recurrent_decode_native(
                    conv_out_ns,
                    g1_ns,
                    beta_ns,
                    self.A_log,
                    self.dt_bias,
                    self.gate_lower_bound,
                    recurrent_state,
                    decode_conv_indices,
                )

        # ---------- merge spec and non-spec outputs ----------
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0]
        elif core_attn_out_non_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[0]
        else:
            assert core_attn_out_spec is not None
            core_attn_out[0, :num_actual_tokens] = core_attn_out_spec[0]
        # Output gate: rmsnorm(o) * sigmoid(g2) per head. forward_native is the
        # decomposed reference (no residual/prenorm here), no Triton.
        core_attn_out.copy_(self.o_norm.forward_native(core_attn_out, g2))


def kda_attention(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._forward(
        mixed_qkv=mixed_qkv, g1=g1, g2=g2, beta=beta, core_attn_out=core_attn_out
    )


def kda_attention_fake(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_attention",
    op_func=kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kda_attention_fake,
)
