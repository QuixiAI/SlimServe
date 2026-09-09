# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace

import pytest

from slimserve.reduction_receipts import graph_snapshot, install


@pytest.fixture
def graphs(tmp_path):
    module_path = tmp_path / "graph.py"
    module_path.write_text("# graph\n")
    kernel_path = tmp_path / "norm.py"
    kernel_path.write_text("# source\n")
    config = SimpleNamespace(
        kwargs={"XBLOCK": 1, "R0_BLOCK": 1024},
        num_warps=8,
        num_stages=1,
        num_ctas=1,
        maxnreg=None,
    )
    tuner = SimpleNamespace(
        filename=str(kernel_path),
        indductor_unused="unrelated",
        inductor_meta=dict(
            kernel_name="triton_red_fused_rms_norm_0",
            num_reduction=1,
            deterministic=True,
            batch_invariant=False,
            are_deterministic_algorithms_enabled=False,
        ),
        deterministic_mode=True,
        _could_rblock_scale=False,
        launchers=[SimpleNamespace(cache_hash="binary", config=config)],
    )
    module = SimpleNamespace(
        __file__=str(module_path), call=lambda: None, triton_red_fused_rms_norm_0=tuner
    )
    return [module], tuner, tmp_path


def test_inspects_aliases_without_mutating_objects(graphs):
    modules, tuner, root = graphs
    modules[0].triton_red_fused_rms_norm_alias = tuner
    before = copy.deepcopy(vars(tuner))
    result = graph_snapshot(modules + modules, root)
    assert not result["violations"]
    assert len(result["graphs"]) == 1 and len(result["bindings"]) == 2
    assert len({r["object_id"] for r in result["bindings"]}) == 1
    assert vars(tuner) == before
    assert result["bindings"][0]["selected"][0]["config"]["num_ctas"] == 1


@pytest.mark.parametrize(
    "change",
    [
        "metadata",
        "runtime",
        "binary",
        "dynamic",
        "batch",
        "global",
        "uninspectable",
        "source",
    ],
)
def test_incomplete_or_wrong_policy_is_not_silently_excluded(graphs, change):
    modules, tuner, root = graphs
    if change == "metadata":
        tuner.inductor_meta["deterministic"] = False
    elif change == "runtime":
        tuner.deterministic_mode = False
    elif change == "binary":
        tuner.launchers = []
    elif change == "dynamic":
        tuner._could_rblock_scale = True
    elif change == "batch":
        tuner.inductor_meta["batch_invariant"] = True
    elif change == "global":
        tuner.inductor_meta["are_deterministic_algorithms_enabled"] = True
    elif change == "uninspectable":
        del tuner.inductor_meta
    else:
        tuner.filename += ".missing"
    assert graph_snapshot(modules, root)["violations"]


def test_no_graphs_cannot_pass():
    assert graph_snapshot([], "/tmp")["violations"]


def test_default_path_never_inspects_runner(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_NATIVE_ORDER", raising=False)
    install(object())
