# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.attention import (
    _largest_kernel_block_within,
)
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionBackend,
)
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_groups,
    group_and_unify_kv_cache_specs,
    unify_kv_cache_spec_page_size,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    TQFullAttentionSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
from vllm.v1.worker.utils import select_common_block_size

pytestmark = pytest.mark.filterwarnings("ignore:.*deprecated.*:FutureWarning")


def test_turboquant_indexes_pages_by_block_stride() -> None:
    assert TurboQuantAttentionBackend.indexes_kv_by_block_stride()


def test_turboquant_kernel_uses_the_physical_cache_block() -> None:
    assert select_common_block_size(512, [TurboQuantAttentionBackend]) == 512


def test_turboquant_draft_uses_largest_fitting_primary_block_divisor() -> None:
    target_page = 1024 * 1152

    block_size = _largest_kernel_block_within(
        TurboQuantAttentionBackend,
        per_token_bytes=1600,
        page_budget=target_page,
        fallback=1024,
    )

    assert block_size == 512


def test_turboquant_draft_page_can_pad_to_hybrid_target() -> None:
    target = FullAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=288,
        head_size_v=288,
        dtype=torch.bfloat16,
    )
    draft = TQFullAttentionSpec(
        block_size=512,
        num_kv_heads=16,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
        tq_slot_size=100,
        tq_cache_dtype="turboquant_k8v4",
        indexes_kv_by_block_stride=True,
    )

    unified = unify_kv_cache_spec_page_size({"target": target, "draft": draft})

    assert unified["target"].page_size_bytes == target.page_size_bytes
    assert unified["draft"].page_size_bytes == target.page_size_bytes
    assert unified["draft"].block_size == 512


def test_deepseek_mla_tuple_planner_keeps_mixed_turboquant_cache() -> None:
    common = {
        "block_size": 256,
        "num_kv_heads": 1,
        "head_size": 512,
        "dtype": torch.uint8,
        "cache_dtype_str": "fp8_ds_mla",
        "model_version": "deepseek_v4",
    }
    specs = {
        "full": MLAAttentionSpec(**common),
        "sliding": SlidingWindowMLASpec(sliding_window=1024, **common),
        "draft": TQFullAttentionSpec(
            block_size=256,
            num_kv_heads=64,
            head_size=64,
            head_size_v=64,
            dtype=torch.bfloat16,
            tq_slot_size=100,
            tq_cache_dtype="turboquant_k8v4",
            indexes_kv_by_block_stride=True,
        ),
    }

    assert group_and_unify_kv_cache_specs(specs) is not None

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        speculative_config=None,
    )
    groups = get_kv_cache_groups(config, specs)

    assert {name for group in groups for name in group.layer_names} == set(specs)


def test_padded_turboquant_view_fits_its_backing_storage() -> None:
    num_blocks = 2
    page_size = 1024 * 1152
    spec = TQFullAttentionSpec(
        block_size=512,
        num_kv_heads=16,
        head_size=64,
        head_size_v=64,
        dtype=torch.uint8,
        tq_slot_size=100,
        tq_cache_dtype="turboquant_k8v4",
        page_size_padded=page_size,
        indexes_kv_by_block_stride=True,
    )
    shape = TurboQuantAttentionBackend.get_kv_cache_shape(
        num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        spec.tq_cache_dtype,
    )
    raw = torch.empty(num_blocks * page_size, dtype=torch.int8)

    cache = _reshape_attention_kv_cache(
        raw,
        spec,
        shape,
        TurboQuantAttentionBackend.get_kv_cache_stride_order(),
        num_blocks,
        packing=None,
    )

    assert cache.shape == (num_blocks, spec.block_size, 16, 100)
    assert cache.stride(0) == page_size
