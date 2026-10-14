# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as indexer


def config(**changes):
    values = dict(
        additional_config={
            "glm5_next_indexer_row_shard": True,
            "glm5_next_compact_indexer_cache": True,
        },
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, data_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_opt_in_and_config_boundary(monkeypatch):
    platform = Mock()
    platform.is_cuda.return_value = platform.is_device_capability.return_value = True
    monkeypatch.setattr(indexer, "current_platform", platform)
    assert indexer._row_shard_option(config(), 32)
    assert not indexer._row_shard_option(SimpleNamespace(additional_config={}), 16)
    for invalid in (1, "true", None):
        with pytest.raises(ValueError, match="boolean"):
            indexer._row_shard_option(config(additional_config={
                "glm5_next_indexer_row_shard": invalid
            }), 32)
    # Speculative decoding is an ordinary row layout to the sharded path
    # (rows = requests x next_n, per-row request map): accepted since
    # 2026-09-10 so MTP and row sharding coexist.
    assert indexer._row_shard_option(
        config(speculative_config=object()), 32
    ) is True
    # TP4 (8 rows per rank at R=32) and DP replicas (rows shard over the
    # replica's own TP group) are accepted since 2026-09-11.
    for good in (
        config(parallel_config=SimpleNamespace(
            tensor_parallel_size=4, data_parallel_size=1, pipeline_parallel_size=1
        )),
        config(parallel_config=SimpleNamespace(
            tensor_parallel_size=4, data_parallel_size=2, pipeline_parallel_size=1
        )),
    ):
        assert indexer._row_shard_option(good, 32) is True
    for bad in (
        config(additional_config={"glm5_next_indexer_row_shard": True}),
        config(parallel_config=SimpleNamespace(
            tensor_parallel_size=2, data_parallel_size=1, pipeline_parallel_size=1
        )),
        config(parallel_config=SimpleNamespace(
            tensor_parallel_size=8, data_parallel_size=1, pipeline_parallel_size=2
        )),
    ):
        with pytest.raises(ValueError, match="requires SM80"):
            indexer._row_shard_option(bad, 32)
    with pytest.raises(ValueError, match="requires SM80"):
        indexer._row_shard_option(config(), 16)
    platform.is_device_capability.return_value = False
    with pytest.raises(ValueError, match="requires SM80"):
        indexer._row_shard_option(config(), 32)


@pytest.mark.parametrize("rows", [0, 1, 8, 9, 15, 16, 17, 31, 32, 33, 64])
@pytest.mark.parametrize("enabled", [False, True])
def test_capture_stable_dispatch(rows, enabled):
    assert indexer._row_shard_dispatch(enabled, rows, 0, 1) == (
        enabled and rows in (16, 32)
    )
    assert not indexer._row_shard_dispatch(enabled, rows, 1, 1)
    # next_n > 1 (speculative rows) shards on the same row-count rule.
    assert indexer._row_shard_dispatch(enabled, rows, 0, 2) == (
        enabled and rows in (16, 32)
    )


@pytest.mark.parametrize("rows", [1, 8, 16, 32, 64])
@pytest.mark.parametrize("enabled", [False, True])
def test_opaque_decode_forwards_policy(monkeypatch, rows, enabled):
    md = SimpleNamespace(
        slot_mapping=torch.arange(rows), num_prefills=0, num_decodes=rows,
        num_decode_tokens=rows,
        decode=SimpleNamespace(seq_lens=torch.ones(rows, dtype=torch.int32) * 1000,
                               block_table=torch.zeros(rows, 2, dtype=torch.int32)),
    )
    monkeypatch.setattr(indexer, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(indexer, "get_forward_context", lambda: SimpleNamespace(
        attn_metadata={"cache": md}
    ))
    monkeypatch.setattr(indexer, "update_pool_cache", Mock())
    select = Mock()
    monkeypatch.setattr(indexer, "_pooled_select", select)
    indexer.glm5_next_pooled_indexer(
        torch.empty(rows, 32, 128), torch.empty(rows, 256), torch.empty(rows, 32),
        torch.empty(4, 128), "cache", torch.empty(2, 576, 64),
        torch.empty(rows, 2080, dtype=torch.int32), torch.empty(rows, 512),
        512, 512, 4, 128**-0.5, row_shard_decode=enabled,
    )
    assert select.call_count == 1
    assert select.call_args.kwargs["row_shard"] == (enabled and rows in (16, 32))
