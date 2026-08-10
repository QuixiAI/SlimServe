# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.turboquant_native import _select_num_kv_splits
from vllm.v1.core.kv_cache_utils import group_and_unify_kv_cache_specs
from vllm.v1.kv_cache_interface import (
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    TQSlidingWindowSpec,
)


def test_native_turboquant_splits_bound_dspark_scratch():
    assert _select_num_kv_splits(1, 64, 16, 128) == 16
    assert _select_num_kv_splits(8, 64, 16, 128) == 2
    assert _select_num_kv_splits(32, 64, 16, 128) == 1
    assert _select_num_kv_splits(640, 64, 16, 128) == 1
    assert _select_num_kv_splits(640, 64, 32, 0) == 32


def test_dsv4_hybrid_grouping_retains_turboquant_draft_layer():
    specs = {
        "main": MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.uint8,
            cache_dtype_str="fp8_ds_mla",
            model_version="deepseek_v4",
        ),
        "swa": SlidingWindowMLASpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
            sliding_window=128,
            model_version="deepseek_v4",
        ),
        "draft": TQSlidingWindowSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=512,
            head_size_v=512,
            dtype=torch.uint8,
            sliding_window=128,
            tq_slot_size=772,
            tq_cache_dtype="turboquant_k8v4",
        ),
    }

    grouped = group_and_unify_kv_cache_specs(specs)

    assert grouped is not None
    grouped_names = {name for group in grouped for name in group.kv_cache_specs}
    assert grouped_names == set(specs)
    draft_spec = next(
        group.kv_cache_specs["draft"]
        for group in grouped
        if "draft" in group.kv_cache_specs
    )
    assert isinstance(draft_spec, TQSlidingWindowSpec)
    assert draft_spec.tq_cache_dtype == "turboquant_k8v4"
