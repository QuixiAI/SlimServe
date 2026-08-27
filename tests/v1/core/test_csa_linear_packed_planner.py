# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CSA+linear (Qwen4Exp) groups must plan through the fork's packed slab.

Upstream vLLM plans this cache format with a dedicated stride-layout builder;
this fork's packed allocator already implements the same block-ownership
overlap contract, so `_get_csa_linear_tensor_layout` recognition must route
`get_kv_cache_config_from_groups` into `_get_kv_cache_config_packed`.
"""

import torch

from vllm.v1.core.kv_cache_utils import (
    _get_csa_linear_tensor_layout,
    _get_packed_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)


def _make_groups():
    block_size = 64
    main = {
        f"layers.{i}.attn": FullAttentionSpec(
            block_size=block_size, num_kv_heads=1, head_size=256, dtype=torch.bfloat16
        )
        for i in (3, 7)
    }
    compressed = {
        f"layers.{i}.indexer": MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            compress_ratio=4,
        )
        for i in (3, 7)
    }
    rings = {
        f"layers.{i}.ring": CircularBufferSpec(
            block_size=8,
            num_kv_heads=1,
            head_size=128,
            head_size_v=0,
            dtype=torch.bfloat16,
            page_size_padded=compressed["layers.3.indexer"].page_size_bytes,
        )
        for i in (3, 7)
    }
    main_page = main["layers.3.attn"].page_size_bytes
    mamba = MambaSpec(
        shapes=((16, 16),),
        dtypes=(torch.float32,),
        block_size=block_size,
        page_size_padded=main_page,
    )
    sparse_group = KVCacheGroupSpec(
        list(main) + list(compressed),
        UniformTypeKVCacheSpecs(
            block_size=block_size, kv_cache_specs={**main, **compressed}
        ),
    )
    ring_group = KVCacheGroupSpec(
        list(rings),
        UniformTypeKVCacheSpecs(block_size=8, kv_cache_specs=rings),
    )
    mamba_group = KVCacheGroupSpec(["layers.0.linear_attn"], mamba)
    return [sparse_group, ring_group, mamba_group]


def test_csa_layout_is_recognized():
    layout = _get_csa_linear_tensor_layout(_make_groups())
    assert layout is not None
    assert len(layout.main_kv_names) == 2
    assert len(layout.compressed_names) == 2
    assert len(layout.compressor_state_names) == 2
    assert len(layout.mamba_groups) == 1


def test_csa_groups_fit_the_packed_slab():
    groups = _make_groups()
    block_stride, layers_by_offset = _get_packed_kv_cache_layout(groups)
    sparse = groups[0].kv_cache_spec
    assert block_stride == sum(
        spec.page_size_bytes for spec in sparse.kv_cache_specs.values()
    )
    # Every layer's page must end within the slab's block stride.
    for byte_offset, layer_names in layers_by_offset.items():
        for name in layer_names:
            for group in groups:
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    page = spec.kv_cache_specs.get(name)
                    if page is not None:
                        assert byte_offset + page.page_size_bytes <= block_stride
                elif name in group.layer_names:
                    assert byte_offset + spec.page_size_bytes <= block_stride


def test_non_csa_groups_are_not_recognized():
    groups = _make_groups()[:1]
    assert _get_csa_linear_tensor_layout(groups) is None
