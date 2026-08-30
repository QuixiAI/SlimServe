# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

from gguf import GGMLQuantizationType as WeightType


def _linear_module():
    return importlib.import_module(
        "vllm.model_executor.layers.quantization.gguf.linear"
    )


def test_rocm_imatrix_routes_follow_measured_crossovers(monkeypatch) -> None:
    linear = _linear_module()
    monkeypatch.setattr(
        linear, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )

    wide_vector_types = (
        WeightType.IQ1_S,
        WeightType.IQ1_M,
        WeightType.IQ2_XXS,
        WeightType.IQ2_S,
        WeightType.IQ3_XXS,
        WeightType.IQ4_XS,
    )
    for weight_type in wide_vector_types:
        assert linear._imatrix_mmvq_batch_limit(17408, weight_type) == 32

    assert linear._imatrix_mmvq_batch_limit(17408, WeightType.IQ2_XS) == 16
    assert linear._imatrix_mmvq_batch_limit(5120, WeightType.IQ3_S) == 8


def test_imatrix_route_override_applies_to_all_formats(monkeypatch) -> None:
    linear = _linear_module()
    monkeypatch.setenv("VLLM_GGUF_MMVQ_MAX_BATCH", "24")
    monkeypatch.setattr(
        linear, "current_platform", SimpleNamespace(is_rocm=lambda: True)
    )

    assert linear._imatrix_mmvq_batch_limit(17408, WeightType.IQ3_S) == 24


def test_unmeasured_platform_keeps_legacy_imatrix_route(monkeypatch) -> None:
    linear = _linear_module()
    monkeypatch.setattr(
        linear, "current_platform", SimpleNamespace(is_rocm=lambda: False)
    )

    assert linear._imatrix_mmvq_batch_limit(17408, WeightType.IQ2_XXS) == 8
    assert linear._imatrix_mmvq_batch_limit(5120, WeightType.IQ2_XXS) == 16
