# SPDX-License-Identifier: Apache-2.0
import json
from dataclasses import dataclass

import numpy as np
import pytest

from benchmarks.analyze_routing_journal import analyze
from slimserve.routing_journal import RoutingJournal, diagnostic_plan


def journal(tmp_path, **kwargs):
    return RoutingJournal(
        tmp_path,
        model="test",
        num_layers=3,
        first_moe_layer=1,
        num_experts=8,
        top_k=2,
        **kwargs,
    )


def inputs():
    return dict(
        data=np.array([[[0, 0], [1, 2], [3, 4]], [[0, 0], [5, 6], [1, 7]]]),
        slots=np.array([91, 22]),
        req_ids=["second", "first"],
        scheduled={"first": 1, "second": 1},
        computed={"first": 1002, "second": 1003},
        prompt_lengths={"first": 1000, "second": 1000},
    )


def rows(j):
    return [json.loads(line) for line in j.path.read_text().splitlines()]


def test_preserves_actual_step_order_and_excludes_dense_layers(tmp_path):
    j = journal(tmp_path)
    j.record(**inputs())
    j.close()
    row = rows(j)[1]
    assert row["request_ids"] == ["second", "first"]
    assert row["computed_tokens"] == [1003, 1002]
    assert row["slots"] == [91, 22]
    assert row["routes"] == [[[1, 2], [3, 4]], [[5, 6], [1, 7]]]


@pytest.mark.parametrize("kind", ["prefill", "one_token_prefill", "too_large"])
def test_skips_are_explicit(tmp_path, kind):
    j = journal(tmp_path, max_tokens=1 if kind == "too_large" else 16)
    args = inputs()
    if kind == "prefill":
        args["scheduled"]["first"] = 64
    if kind == "one_token_prefill":
        args["computed"]["first"] = 1000
    j.record(**args)
    j.close()
    assert rows(j)[1]["kind"] == "skip"
    assert j.recorded == 0


@pytest.mark.parametrize("kind", ["shape", "slots", "range", "duplicates", "dtype"])
def test_bad_capture_is_retained_and_rejected(tmp_path, kind):
    j = journal(tmp_path)
    args = inputs()
    if kind == "shape":
        args["data"] = args["data"][:1]
    elif kind == "slots":
        args["slots"] = args["slots"][:1]
    elif kind == "range":
        args["data"][0, 1, 0] = 8
    elif kind == "duplicates":
        args["data"][0, 1, 0] = 2
    else:
        args["data"] = args["data"].astype(float)
    with pytest.raises(ValueError):
        j.record(**args)
    j.close()
    assert rows(j)[1]["kind"] == "invalid"


def test_record_bound_does_not_overwrite_evidence(tmp_path):
    j = journal(tmp_path, max_records=1)
    j.record(**inputs())
    j.record(**inputs())
    j.close()
    assert [r["kind"] for r in rows(j)] == ["header", "decode", "limit"]
    with pytest.raises(FileExistsError):
        journal(tmp_path)


def test_non_decode_steps_are_also_bounded(tmp_path):
    j = journal(tmp_path, max_records=1, max_tokens=1)
    for _ in range(8):
        j.record(**inputs())
    j.close()
    assert [r["kind"] for r in rows(j)] == ["header"] + ["skip"] * 4 + ["step_limit"]


def test_analysis_counts_unique_experts_in_the_same_step(tmp_path):
    j = journal(tmp_path)
    args = inputs()
    # One shared expert in layer 1, two in layer 2: 3 + 2 unique experts.
    args["data"][1, 1] = [2, 5]
    args["data"][1, 2] = [3, 4]
    j.record(**args)
    j.close()
    result = analyze(j.path, per_expert_bytes=100)
    row = result["summary"][2]
    assert row["sum_layer_unique_experts"]["mean"] == 5
    assert row["max_tokens_per_expert"] == 2
    assert row["marlin_m8_route_utilization"]["mean"] == 8 / 40
    assert row["unique_weight_footprint_bytes_not_dram"]["mean"] == 500


def test_analysis_rejects_failed_capture(tmp_path):
    j = journal(tmp_path)
    args = inputs()
    args["data"].fill(0)
    with pytest.raises(ValueError):
        j.record(**args)
    j.close()
    with pytest.raises(ValueError, match="invalid routing capture"):
        analyze(j.path)


def test_analysis_rejects_unknown_record_after_valid_decode(tmp_path):
    j = journal(tmp_path)
    j.record(**inputs())
    j._write({"kind": "decdoe"})
    j.close()
    with pytest.raises(ValueError, match="unsupported schema-1.*decdoe"):
        analyze(j.path)


@dataclass
class Plan:
    profile_id: str
    platform: str
    speculative: bool
    engine: dict


def test_capture_plan_is_explicit_and_preserves_the_original(tmp_path):
    original = Plan("glm53-nvfp4-4", "rtx6000", False, {"additional_config": {"a": 1}})
    changed = diagnostic_plan(original, tmp_path)
    assert changed.engine["enable_return_routed_experts"] is True
    assert changed.engine["async_scheduling"] is False
    assert changed.engine["additional_config"]["a"] == 1
    assert original.engine == {"additional_config": {"a": 1}}
    with pytest.raises(ValueError, match="no-spec"):
        diagnostic_plan(Plan(original.profile_id, "rtx6000", True, {}), tmp_path)
    with pytest.raises(ValueError, match="RTX6000"):
        diagnostic_plan(Plan(original.profile_id, "a100", False, {}), tmp_path)


def test_real_profile_serializes_explicit_capture(tmp_path):
    from slimserve import cli, registry
    from slimserve.engine import serve_argv

    args = cli._parser().parse_args(
        ["glm53-nvfp4-4", "--route-profile-dir", str(tmp_path)]
    )
    original = registry.resolve("glm53-nvfp4-4", "rtx6000", 4, None)
    plan = diagnostic_plan(original, args.route_profile_dir)
    argv = serve_argv(plan, "127.0.0.1", 8000)
    assert "--enable-return-routed-experts" in argv
    assert "--no-async-scheduling" in argv
    additional = json.loads(argv[argv.index("--additional-config") + 1])
    assert additional["slimserve_routing_journal"] == str(tmp_path)
    assert "enable_return_routed_experts" not in original.engine
