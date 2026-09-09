# SPDX-License-Identifier: Apache-2.0
import inspect

import pytest
import torch

from slimserve import canonical_indexer as ci


def test_disabled_is_original_callable(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER", raising=False)
    function = lambda: None
    assert ci.maybe_ordered_topk(function) is function


@pytest.mark.parametrize("missing", ["MODEL_JOURNAL", "CANONICAL_MOE", "INDEX_JOURNAL"])
def test_flag_prerequisites(monkeypatch, missing):
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER", "1")
    for name in ("MODEL_JOURNAL", "CANONICAL_MOE", "INDEX_JOURNAL"):
        monkeypatch.setenv("SLIMSERVE_GLM53_" + name, "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_" + missing)
    with pytest.raises(ValueError, match="requires"):
        ci.enabled()


def test_bad_flag_and_cpu_tensor_rejected(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER", "yes")
    with pytest.raises(ValueError, match="0 or 1"):
        ci.enabled()
    with pytest.raises(ValueError, match="CUDA int32"):
        ci.canonicalize(torch.zeros(2, 512, dtype=torch.int32))


def test_wrapper_preserves_arguments_result_and_orders_after_selection(monkeypatch):
    for name in (
        "CANONICAL_INDEX_ORDER",
        "MODEL_JOURNAL",
        "CANONICAL_MOE",
        "INDEX_JOURNAL",
    ):
        monkeypatch.setenv("SLIMSERVE_GLM53_" + name, "1")
    calls = []

    def native(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        raw_topk_indices,
        num_rows,
        stride0,
        stride1,
        topk_tokens,
    ):
        calls.append("native")
        raw_topk_indices.fill_(3)
        return logits

    def order(indices):
        assert (indices == 3).all()
        calls.append("ordered")

    monkeypatch.setattr(ci, "canonicalize", order)
    wrapped = ci.maybe_ordered_topk(native)
    assert inspect.signature(wrapped) == inspect.signature(native)
    logits = torch.zeros(2, 600)
    indices = torch.zeros(2, 512, dtype=torch.int32)
    assert wrapped(logits, None, None, indices, 2, 600, 1, 512) is logits
    assert calls == ["native", "ordered"]
    with pytest.raises(ValueError, match="geometry"):
        wrapped(logits, None, None, indices, 3, 600, 1, 512)
