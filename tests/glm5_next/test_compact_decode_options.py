# SPDX-License-Identifier: Apache-2.0
"""Independent opt-ins and narrow dispatch for compact-cache optimizations."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as indexer
from vllm.model_executor.layers import glm5_next_pool_score as score


def test_decode_options_are_independent_and_require_compact_state():
    adaptive = "glm5_next_adaptive_pool_score"
    singleton = "glm5_next_singleton_pool_update"
    base = {"glm5_next_compact_indexer_cache": True}
    for a, s in ((False, False), (False, True), (True, False), (True, True)):
        config = SimpleNamespace(additional_config=base | {adaptive: a, singleton: s})
        assert indexer._compact_decode_options(config, 32) == (a, s)
    for flag in (adaptive, singleton):
        with pytest.raises(ValueError, match="requires"):
            indexer._compact_decode_options(
                SimpleNamespace(additional_config={flag: True}), 32
            )
        for invalid in (1, "false", None):
            with pytest.raises(ValueError, match="boolean"):
                indexer._compact_decode_options(
                    SimpleNamespace(additional_config=base | {flag: invalid}), 32
                )
    with pytest.raises(ValueError, match="32 heads"):
        indexer._compact_decode_options(
            SimpleNamespace(additional_config=base | {adaptive: True}), 16
        )
    assert indexer._compact_decode_options(
        SimpleNamespace(additional_config={}), 16
    ) == (False, False)


@pytest.mark.parametrize(
    "rows,enabled,which",
    [
        (0, True, None),
        (1, False, "baseline"),
        (1, True, "adaptive"),
        (64, True, "adaptive"),
        (65, True, "baseline"),
    ],
)
def test_adaptive_dispatch_stays_in_qualified_decode_shapes(
    monkeypatch, rows, enabled, which
):
    baseline, adaptive = Mock(), Mock()
    monkeypatch.setattr(indexer, "cached_pool_logits", baseline)
    monkeypatch.setattr(score, "adaptive_pool_logits", adaptive)
    monkeypatch.setattr(indexer.ops, "top_k_per_row_prefill", Mock())

    class NoopKernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(indexer, "_expand_topk_kernel", NoopKernel())
    indexer._pooled_select(
        torch.empty(rows, 32, 128),
        torch.empty(rows, 32),
        torch.empty(4, 128),
        torch.empty(1, 64, 64),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.full((rows,), 4, dtype=torch.int32),
        torch.empty(rows, 16),
        16,
        64,
        128**-0.5,
        1,
        torch.empty(rows, 32, dtype=torch.int32),
        4,
        adaptive_score=enabled,
    )
    assert baseline.call_count == (which == "baseline")
    assert adaptive.call_count == (which == "adaptive")


def test_old_opaque_callers_keep_false_defaults():
    arguments = torch.ops.vllm.glm5_next_pooled_indexer.default._schema.arguments
    assert [arg.name for arg in arguments[-3:]] == [
        "adaptive_score",
        "singleton_fused_update",
        "row_shard_decode",
    ]
    assert all(arg.default_value is False for arg in arguments[-3:])
