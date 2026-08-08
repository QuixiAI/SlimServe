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
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
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
        tq_slot_size=108,
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
            tq_slot_size=108,
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
        tq_slot_size=108,
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

    assert cache.shape == (num_blocks, spec.block_size, 16, 108)
    assert cache.stride(0) == page_size


def test_turboquant_impl_preserves_block_token_head_layout() -> None:
    impl = object.__new__(TurboQuantAttentionImpl)
    impl.num_heads = 4
    impl.num_kv_heads = 1
    impl.head_size = 8
    impl._ensure_on_device = lambda layer, device: None

    cache = torch.empty(3, 16, 1, 12, dtype=torch.uint8)
    key = torch.empty(1, 1, 8)
    value = torch.empty_like(key)
    slot_mapping = torch.tensor([16], dtype=torch.int64)
    seen: list[torch.Tensor] = []

    def record_store(key, value, kv_cache, slot_mapping, layer) -> None:
        seen.append(kv_cache)

    impl._store_kv = record_store

    layer = SimpleNamespace()
    impl.do_kv_cache_update(layer, key, value, cache, slot_mapping)

    assert len(seen) == 1
    assert seen[0] is cache
    assert seen[0].shape == (3, 16, 1, 12)

    layer._tq_Pi = torch.eye(8)
    layer._tq_PiT = torch.eye(8)
    layer._tq_centroids = torch.zeros(1)

    def record_decode(query, kv_cache, *args):
        seen.append(kv_cache)
        return torch.zeros_like(query)

    impl._decode_attention = record_decode
    metadata = TurboQuantMetadata(
        seq_lens=torch.tensor([1], dtype=torch.int32),
        slot_mapping=slot_mapping,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=1,
        is_prefill=False,
    )
    query = torch.empty(1, 4, 8)
    impl.forward(layer, query, key, value, cache, metadata)

    assert seen[-1] is cache
    assert seen[-1].shape == (3, 16, 1, 12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device kernel")
def test_turboquant_gpu_crosses_padded_cache_pages() -> None:
    device = torch.device("cuda")
    num_blocks = 4
    block_size = 16
    num_kv_heads = 1
    num_query_heads = 4
    head_dim = 64
    shape = TurboQuantAttentionBackend.get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        "turboquant_k8v4",
    )
    slot_size = shape[-1]
    natural_page_size = block_size * num_kv_heads * slot_size
    physical_page_size = natural_page_size + 128
    raw = torch.full(
        (num_blocks * physical_page_size,),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    cache = torch.as_strided(
        raw,
        size=shape,
        stride=(physical_page_size, num_kv_heads * slot_size, slot_size, 1),
    )

    slots = torch.tensor(
        [block_size - 1, block_size, 3 * block_size - 1], device=device
    )
    key = torch.randn(3, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    identity = torch.eye(head_dim, dtype=torch.float32, device=device)
    triton_turboquant_store(
        key,
        value,
        cache,
        slots,
        identity,
        torch.empty(0, dtype=torch.float32, device=device),
        mse_bits=8,
        key_packed_size=head_dim,
        value_quant_bits=4,
        key_fp8=True,
    )

    padding = raw.view(num_blocks, physical_page_size)[:, natural_page_size:]
    assert torch.all(padding == 0xA5)

    output = triton_turboquant_decode_attention(
        query=torch.randn(
            1,
            num_query_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        ),
        kv_cache=cache,
        block_table=torch.arange(num_blocks, dtype=torch.int32, device=device).view(
            1, -1
        ),
        seq_lens=torch.tensor([3 * block_size], dtype=torch.int32, device=device),
        Pi=identity,
        centroids=torch.zeros(1, dtype=torch.float32, device=device),
        scale=head_dim**-0.5,
        mse_bits=8,
        key_packed_size=head_dim,
        value_quant_bits=4,
        key_fp8=True,
        sliding_window=block_size + 1,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
