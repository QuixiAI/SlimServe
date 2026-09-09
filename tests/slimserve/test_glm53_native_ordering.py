# SPDX-License-Identifier: Apache-2.0
"""Native-only policy does not enable observers, sorting fallbacks or stale graphs."""

from dataclasses import replace

import pytest
import torch

from slimserve import canonical_indexer, canonical_moe, cli
from slimserve import glm53_ordering as order
from slimserve.hardware import Machine
from slimserve.registry import resolve
from tests.slimserve.test_stable_align_wiring import invoke
from vllm.model_executor.layers.fused_moe.experts import marlin_moe
from vllm.quixicore.ops import quixicore_ops


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    for key in order.LEGACY_FLAGS + (order.FLAG, "SLIMSERVE_GLM_ROUTE_ALIGN"):
        monkeypatch.delenv(key, raising=False)
    for suffix in ("MODEL_JOURNAL", "INDEX_JOURNAL", "SCORE_JOURNAL", "MOE_JOURNAL"):
        monkeypatch.delenv("SLIMSERVE_GLM53_" + suffix, raising=False)


def test_policy_is_off_by_default_and_does_not_mutate_environment(monkeypatch):
    import os

    assert not order.enabled()
    monkeypatch.setenv(order.FLAG, "1")
    before = dict(os.environ)
    assert all(
        call()
        for call in (
            canonical_moe.enabled,
            canonical_moe.stable_route_enabled,
            canonical_moe.stable_align_enabled,
            canonical_indexer.enabled,
            canonical_indexer.ties_enabled,
            canonical_indexer.fused_enabled,
        )
    )
    assert dict(os.environ) == before


@pytest.mark.parametrize("value", ["", "yes", "-1", "2"])
def test_invalid_policy_rejected(monkeypatch, value):
    monkeypatch.setenv(order.FLAG, value)
    with pytest.raises(ValueError, match="0 or 1"):
        order.enabled()


@pytest.mark.parametrize("legacy", order.LEGACY_FLAGS)
def test_mixed_control_policies_rejected(monkeypatch, legacy):
    monkeypatch.setenv(order.FLAG, "1")
    monkeypatch.setenv(legacy, "1")
    with pytest.raises(ValueError, match="cannot combine"):
        order.enabled()


def test_disabling_fused_router_rejected(monkeypatch):
    monkeypatch.setenv(order.FLAG, "1")
    monkeypatch.setenv("SLIMSERVE_GLM_ROUTE_ALIGN", "0")
    with pytest.raises(ValueError, match="requires fused small-M"):
        order.enabled()


@pytest.mark.parametrize(
    "field",
    ["profile", "platform", "gpus", "quant", "recipe", "tp", "ep", "spec", "moe"],
)
def test_plan_scope_fails_closed(monkeypatch, field):
    plan = resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    monkeypatch.setenv(order.FLAG, "1")
    order.validate_plan(plan)
    changes = {
        "profile": dict(profile_id="other"),
        "platform": dict(platform="a100"),
        "gpus": dict(gpus=8),
        "quant": dict(quant=replace(plan.quant, name="FP8")),
        "recipe": dict(weight_recipe=None),
        "spec": dict(speculative=True),
        "tp": dict(engine={**plan.engine, "tensor_parallel_size": 2}),
        "ep": dict(engine={**plan.engine, "enable_expert_parallel": True}),
        "moe": dict(engine={**plan.engine, "moe_backend": "other"}),
    }
    with pytest.raises(ValueError, match="requires glm53"):
        order.validate_plan(replace(plan, **changes[field]))


def test_cli_scope_and_recipe_unchanged(monkeypatch):
    machine = Machine("rtx6000", "RTX PRO 6000 Blackwell", 4)
    monkeypatch.setattr(cli.hardware, "detect", lambda: machine)
    monkeypatch.setenv(order.FLAG, "1")
    captured = []
    monkeypatch.setattr(cli, "_show", captured.append)
    monkeypatch.setattr(cli.fetch, "ensure", lambda *a, **k: pytest.fail("download"))
    assert cli.main(["glm53-nvfp4-4", "--dry-run"]) == 0
    assert captured == [resolve("glm53-nvfp4-4", "rtx6000", 4, None)]
    monkeypatch.setattr(cli.hardware, "detect", lambda: Machine("a100", "A100", 4))
    assert cli.main(["glm53-nvfp4-4", "--dry-run"]) == 2


def test_cache_factor_reaches_actual_graph_capabilities(monkeypatch):
    before = quixicore_ops.graph_factors()
    assert order.cache_factor() in before
    monkeypatch.setenv(order.FLAG, "1")
    after = quixicore_ops.graph_factors()
    assert order.cache_factor() in after
    assert set(after) - set(before) == {"glm53_native_order_v1=1"}
    assert set(before) - set(after) == {"glm53_native_order_v1=0"}


def test_native_pool_selector_never_calls_sort_control(monkeypatch):
    monkeypatch.setenv(order.FLAG, "1")
    calls = []
    monkeypatch.setattr(
        canonical_indexer, "canonicalize", lambda *a: pytest.fail("sort")
    )
    monkeypatch.setattr(
        canonical_indexer, "_native_tie_selector", lambda: pytest.fail("unfused")
    )
    monkeypatch.setattr(
        canonical_indexer, "_native_fused_selector", lambda: lambda *a: calls.append(a)
    )
    function = canonical_indexer.maybe_ordered_topk(lambda *a: pytest.fail("atomic"))
    ids = torch.empty((17, 512), dtype=torch.int32)
    function(None, None, None, ids, 17, 600, 1, 512)
    assert len(calls) == 1 and calls[0][3] is ids


@pytest.mark.parametrize("tokens", [1, 16, 8193])
def test_native_policy_refuses_sorting_fallback(monkeypatch, tokens):
    monkeypatch.setattr(marlin_moe, "_NATIVE_ORDER", True)
    with pytest.raises(ValueError, match="no sorting fallback"):
        invoke(monkeypatch, tokens)


@pytest.mark.parametrize("tokens", [17, 32, 33, 640, 8192])
def test_native_policy_uses_qualified_alignment(monkeypatch, tokens):
    monkeypatch.setattr(marlin_moe, "_NATIVE_ORDER", True)
    assert invoke(monkeypatch, tokens) == ["native"]


def test_native_policy_keeps_small_fused_alignment(monkeypatch):
    monkeypatch.setattr(marlin_moe, "_NATIVE_ORDER", True)
    assert invoke(monkeypatch, 16, reused=True) == []


def test_explicit_index_observer_records_effective_native_policy(monkeypatch, tmp_path):
    import json

    from slimserve import index_journal
    from tests.slimserve.test_index_journal import config

    monkeypatch.setenv(order.FLAG, "1")
    assert not index_journal.enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEX_JOURNAL", "test-config")
    with pytest.raises(ValueError, match="requires bounded model journal"):
        index_journal.enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    assert index_journal.enabled()
    journal = index_journal.IndexJournal(config(tmp_path))
    journal.close()
    header = json.loads(journal.path.read_text().splitlines()[0])
    assert header["selection_order"] == "canonical-pool-id"
    assert header["selection_ties"] == "smaller-pool-id"
    assert header["selection_order_implementation"] == "native-bitonic"
