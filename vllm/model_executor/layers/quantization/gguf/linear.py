# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from . import ops
from .params import (
    GGUFUninitializedWeightParameter,
    GGUFUninitializedWeightTypeParameter,
    GGUFWeightParameter,
    _gguf_ordered_shard_ids,
    _materialize_gguf_weight_parameter,
    _materialize_gguf_weight_type_parameter,
    _resolve_gguf_weight_loader,
    _resolve_gguf_weight_type_loader,
)
from .utils import (
    DEQUANT_TYPES,
    IMATRIX_QUANT_TYPES,
    MMQ_QUANT_TYPES,
    MMVQ_QUANT_TYPES,
    UNQUANTIZED_TYPES,
)


def _cublas_dequant_enabled() -> bool:
    # Metal has no generic GGUF dequant kernel, so the dequant-then-dense route
    # does not exist there; it reads the quantized blocks directly instead.
    if current_platform.is_rocm() or current_platform.is_metal():
        return False
    return os.environ.get("VLLM_GGUF_CUBLAS", "1").lower() not in ("0", "false")


def _mmq_shape_ok(x: torch.Tensor, qweight: torch.Tensor) -> bool:
    """Whether the tile GEMM can take this shape.

    Only Metal constrains it: that kernel derives its grid by integer division,
    so a partial tile is skipped rather than computed. Everywhere else the tile
    kernels handle the remainder themselves.
    """
    if not current_platform.is_metal():
        return True
    return x.shape[0] % 32 == 0 and qweight.shape[0] % 32 == 0


def _cublas_min_batch(rows: int) -> int:
    """Smallest batch where dequant-to-bf16 + cuBLAS beats mmq_v2 for q8_0.

    Measured on idle A100 across N in {1536..12288}, K in {2048..8192}:
    mmq_v2 wins through 64 tokens, the routes cross at 96 for N <= 6144
    (1.1-1.4x for cuBLAS, growing to 1.5-1.8x at 256), while N = 12288 stays
    a tie until 160 because mmq_v2's tile wave still fits the SMs there.
    """
    override = os.environ.get("VLLM_GGUF_CUBLAS_MIN_BATCH")
    if override:
        return int(override)
    return 160 if rows >= 8192 else 96


_q8_0_scratch: dict[tuple[torch.device, int, int, torch.dtype], torch.Tensor] = {}


def _q8_0_dequant_scratch(
    qweight: torch.Tensor, rows: int, cols: int, dtype: torch.dtype
) -> torch.Tensor:
    """Reused dequant buffer, keyed by shape.

    Allocated once on first use and held forever, so every later call --
    including CUDA graph replays -- sees a fixed pointer. A first use inside
    graph capture allocates from the capture pool, which is also safe because
    the buffer is never freed.
    """
    key = (qweight.device, rows, cols, dtype)
    buf = _q8_0_scratch.get(key)
    if buf is None:
        buf = torch.empty(rows, cols, dtype=dtype, device=qweight.device)
        _q8_0_scratch[key] = buf
    return buf


def _mmvq_batch_limit(rows: int, qweight_type: int) -> int:
    """Largest batch for which the GEMV kernel still beats the GEMM kernel.

    mmq's runtime is flat in batch (weight-stationary) while mmvq's is linear,
    so the crossover is mmq_time / mmvq_time_per_token.  Measured on MI300X at
    GLM-5.2 TP2 shapes it lands at 28-340 for ordinary layers and ~5 for the
    vocab projection, because mmq cannot fill 304 CUs until the output is wide.
    Q8_0 projections with 6,144 output rows remain faster on the multi-column
    vector kernel through batch 64. The historical 2/6 sent every layer to a
    kernel 5-20x slower from batch 4 up.
    """
    override = os.environ.get("VLLM_GGUF_MMVQ_MAX_BATCH")
    if override:
        return int(override)
    if not current_platform.is_rocm():
        return 8 if rows > 5120 else 16
    if rows >= 32768:
        return 4
    if qweight_type == int(WeightType.Q8_0) and rows <= 6144:
        return 64
    if rows < 4096:
        return 64
    return 16


def _fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = _mmvq_batch_limit(qweight.shape[0], qweight_type)
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif (
        qweight_type == WeightType.Q8_0
        and x.shape[0] >= _cublas_min_batch(qweight.shape[0])
        and _cublas_dequant_enabled()
    ):
        weight = _q8_0_dequant_scratch(qweight, qweight.shape[0], x.shape[1], x.dtype)
        ops.ggml_dequantize_into(
            qweight, qweight_type, weight.shape[0], weight.shape[1], weight
        )
        y = x @ weight.T
    elif qweight_type in MMQ_QUANT_TYPES and _mmq_shape_ok(x, qweight):
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in MMVQ_QUANT_TYPES and current_platform.is_metal():
        # Ragged batch on Metal: the tile kernel cannot take this shape and
        # there is no dequant fallback, so widen the vector path rather than
        # failing. Slower than a tile pass, still correct.
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf
except AttributeError as error:
    raise error


@register_weight_loader_v2_supported_method
class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(self, quant_config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        fallback_weight_loader = extra_weight_attrs.pop("weight_loader", None)
        weight_loader = _resolve_gguf_weight_loader(layer, fallback_weight_loader)
        assert weight_loader is not None

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedWeightParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "weight_loader": weight_loader,
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
                "tp_rank": layer.tp_rank,
                "tp_size": layer.tp_size,
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        weight_loader_type = _resolve_gguf_weight_type_loader(
            layer, fallback_weight_loader
        )
        assert weight_loader_type is not None
        qweight_type = GGUFUninitializedWeightTypeParameter(requires_grad=False)
        set_weight_attrs(
            qweight_type,
            {
                "weight_loader": weight_loader_type,
                "weight_type": 0,
                "shard_weight_type": {},
                "num_elements": len(output_partition_sizes),
                "ignore_warning": True,
                "tp_rank": layer.tp_rank,
                "tp_size": layer.tp_size,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._materialize_gguf_parameters(layer)
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)

    def _materialize_gguf_parameters(self, layer: torch.nn.Module) -> None:
        self._materialize_qweight(layer)
        self._materialize_qweight_type(layer)

    def _materialize_qweight(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_parameter(layer, "qweight")

    def _materialize_qweight_type(self, layer: torch.nn.Module) -> None:
        _materialize_gguf_weight_type_parameter(layer, "qweight_type")

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[int | str, tuple[int, int, int]]()
            ordered_shard_ids = _gguf_ordered_shard_ids(shard_id)
            current_offset = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = current_offset
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                current_offset = end
            padded_param = GGUFWeightParameter(
                data=padded_data,
                weight_loader=qweight.weight_loader,
                input_dim=qweight.input_dim,
                output_dim=qweight.output_dim,
                tensor_shape=qweight.tensor_shape,
            )
            padded_param.data_container = []
            padded_param.shard_id = ordered_shard_ids
            padded_param.shard_id_map = dict(qweight.shard_id_map)
            if hasattr(qweight, "ignore_warning"):
                padded_param.ignore_warning = qweight.ignore_warning
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            qweight.data_container.clear()
            qweight.shard_id.clear()
            qweight.shard_id_map.clear()
            if qweight.data.numel() > 0:
                qweight.data = torch.empty(
                    0, dtype=qweight.dtype, device=qweight.device
                )
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from . import fused_mul_mat_gguf as fused_mul_mat_gguf_op

        shard_id = layer.qweight.shard_id
        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            fallback_wtype = layer.qweight_type.weight_type
            shard_weight_types = [
                layer.qweight_type.shard_weight_type.get(idx, fallback_wtype)
                for idx in shard_id
            ]
            if len(set(shard_weight_types)) == 1:
                out = fused_mul_mat_gguf_op(x, qweight, shard_weight_types[0])
                if bias is not None:
                    out.add_(bias)
                return out
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type.get(
                    idx, fallback_wtype
                )
                result.append(
                    fused_mul_mat_gguf_op(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf_op(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out
