"""Multi-pool packed slab (E5b): each page-size class gets its own block
pool at its natural page size; the attention-class pool is sized for
`kv_pool_deep_requests` max-length requests and the rest maximizes
chat-context concurrency. Uses the CSA+linear group fixture."""

from types import SimpleNamespace

import vllm.v1.core.kv_cache_utils as kvu
from tests.v1.core.test_csa_linear_packed_planner import _make_groups
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec


def _cfg(deep):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"kv_pool_deep_requests": deep}
        ),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, mamba_cache_mode="align"),
        model_config=SimpleNamespace(max_model_len=262144),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, prefill_context_parallel_size=1,
            tensor_parallel_size=1, pipeline_parallel_size=1,
        ),
        speculative_config=None,
    )


def _plan(monkeypatch, deep, available):
    monkeypatch.setattr(kvu, "may_override_num_blocks", lambda cfg, n: n)
    monkeypatch.setattr(kvu, "host_resident_kv_layers", lambda cfg, groups: [])
    groups = _make_groups()
    num_blocks, tensors, pools = kvu._get_kv_cache_config_packed(_cfg(deep), groups, available)
    return groups, num_blocks, tensors, pools


def test_single_pool_when_unset(monkeypatch):
    groups, num_blocks, tensors, pools = _plan(monkeypatch, 0, 1 << 30)
    assert pools == ([], [])
    assert all(t.pool == 0 for t in tensors)


def test_pools_follow_page_classes_and_keep_deep_capacity(monkeypatch):
    groups, num_blocks, tensors, pools = _plan(monkeypatch, 1.0, 1 << 30)
    pool_sizes, group_pool = pools
    assert len(pool_sizes) >= 2 and pool_sizes[0] == num_blocks
    # attention-class groups are pool 0; mamba groups are not
    for g, pl in zip(groups, group_pool):
        if isinstance(g.kv_cache_spec, MambaSpec):
            assert pl > 0
    strides = sorted({t.block_stride for t in tensors}, reverse=True)
    assert strides[0] == tensors[0].block_stride
    for t in tensors:
        assert t.size == t.block_stride * pool_sizes[t.pool]
    total = sum(pool_sizes[p] * st for p, st in enumerate(strides))
    assert total <= (1 << 30) + strides[0]
    cfg = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups,
        pool_num_blocks=pool_sizes, group_pool=group_pool,
    )
    assert cfg.num_pools == len(pool_sizes)
    assert cfg.blocks_in_pool(1) == pool_sizes[1]
