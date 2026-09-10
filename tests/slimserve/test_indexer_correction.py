# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest
import torch

from benchmarks.kernels import check_glm53_indexer_correction as probe
from benchmarks.kernels import glm53_indexer_correction as kernel


def tensors(rows=3):
    packed = torch.zeros(rows, 2336, dtype=torch.bfloat16)
    weights = [torch.ones(n, dtype=torch.bfloat16) for n in (512, 1536, 128, 128)]
    outputs = [torch.empty(rows, n, dtype=torch.bfloat16) for n in (512, 1536)]
    outputs.append(torch.empty(rows, 256, dtype=torch.bfloat16)[:, :128])
    selected = torch.empty(rows, 128, dtype=torch.uint8)
    return (packed, *weights, *outputs, rows, rows, rows), selected


def test_jit_import_is_gpu_inert_and_fixed_precision_policy():
    assert not torch.cuda.is_initialized()
    assert kernel.correct_indexer.arg_names == [
        "Packed",
        "Gamma",
        "Bias",
        "Output",
        "Selected",
        "N",
    ]
    assert kernel.EXPONENT == -12
    assert kernel.OPTIONS == {
        "num_warps": 1,
        "num_stages": 1,
        "enable_fp_fusion": False,
    }
    source = kernel.correct_indexer.src
    assert "0.000244140625 * scale" in source
    assert "row * 2336 + 2048 + col" in source
    assert "row * 256 + col" in source
    assert "tl.sum(centered * centered, 0)" in source
    assert "value.to(tl.bfloat16), selected" in source
    assert "tl.full((), 1e-6, tl.float64)" in source
    assert ".to(tl.float64)" in source


def test_adapter_preserves_exact_abi_stream_and_order():
    args, selected = tensors()
    calls = []

    def combo(*a, **kw):
        calls.append(("combo", a, kw))
        return "original result"

    def correction(*a, **kw):
        calls.append(("correction", a, kw))

    adapter = kernel.IndexerOnlyCorrection(combo, correction, selected)
    assert adapter(*args, stream=123) == "original result"
    assert calls[0] == ("combo", args, dict(stream=123))
    expected = (args[0], args[3], args[4], args[7], selected, 3)
    assert all(a is b for a, b in zip(calls[1][1][:-1], expected[:-1]))
    assert calls[1][1][-1] == 3 and calls[1][2] == dict(stream=123)
    assert adapter.combo_calls == adapter.correction_calls == 1


@pytest.mark.parametrize(
    "bad", ["rows", "bool_rows", "short_abi", "stride", "packed", "dtype", "selection"]
)
def test_adapter_rejects_invalid_layout_before_original_launch(bad):
    args, selected = tensors()
    args = list(args)
    if bad == "rows":
        args[-1] = 2
    elif bad == "bool_rows":
        args[-3:] = [True] * 3
    elif bad == "short_abi":
        args.pop()
    elif bad == "stride":
        args[7] = args[7].contiguous()
    elif bad == "packed":
        args[0] = args[0][:, :2335]
    elif bad == "dtype":
        args[3] = args[3].float()
    else:
        selected = selected[:, :127]

    def forbidden(*a, **kw):
        raise AssertionError("launch forbidden")

    with pytest.raises(ValueError):
        kernel.IndexerOnlyCorrection(forbidden, forbidden, selected)(*args, stream=0)


def test_failed_original_does_not_launch_correction():
    args, selected = tensors()

    def fail(*a, **kw):
        raise RuntimeError("original failed")

    def forbidden(*a, **kw):
        raise AssertionError("correction forbidden")

    adapter = kernel.IndexerOnlyCorrection(fail, forbidden, selected)
    with pytest.raises(RuntimeError, match="original failed"):
        adapter(*args, stream=0)
    assert adapter.combo_calls == adapter.correction_calls == 0


def phase_fixture():
    baseline = [torch.ones(3, w, dtype=torch.bfloat16) for w in (512, 1536, 128)]
    baseline[2][1, 60] = 4.0605664253234863e-7
    reference = baseline[2].clone()
    reference[1, 60] = 3.986060619354248e-7
    candidate = [t.clone() for t in baseline]
    candidate[2] = reference.clone()
    bias = torch.full((128,), -0.57421875, dtype=torch.bfloat16)
    selected = probe.precision.cancellation_mask(baseline[2], bias, -12).byte()
    return baseline, candidate, selected, reference, bias


def test_phase_metrics_count_actual_selection_and_require_exact_neighbors():
    baseline, candidate, selected, reference, bias = phase_fixture()
    metrics = probe.phase_metrics(baseline, candidate, selected, reference, bias)
    assert (
        metrics["selected_elements"]
        == metrics["selected_rows"]
        == metrics["changed_elements"]
        == 1
    )
    assert metrics["missed_original_failures"] == 0
    assert metrics["corrected_oracle"]["max_bf16_ulp"] == 0
    selected[1, 60] = 0
    with pytest.raises(ValueError, match="detector disagrees"):
        probe.phase_metrics(baseline, candidate, selected, reference, bias)


@pytest.mark.parametrize("slot,column", [(0, 0), (1, 0), (2, 59)])
def test_any_non_target_or_unselected_change_rejected(slot, column):
    args = phase_fixture()
    args[1][slot][0, column] = 2
    with pytest.raises(ValueError, match="Q/KV changed|unselected indexer"):
        probe.phase_metrics(*args)


def record_fixture():
    baseline, candidate, selected, reference, bias = phase_fixture()
    phase = probe.phase_metrics(baseline, candidate, selected, reference, bias)
    case = dict(rank=0, rows=3, seed=530901, magnitude=0.125)
    historical = dict(
        inputs=["input0", "input1"], outputs={"combo": [phase["baseline_sha256"]] * 2}
    )
    record = dict(
        **case,
        input_sha256=historical["inputs"],
        phases=[phase, copy.deepcopy(phase)],
        replay_guards_mutation_passed=True,
    )
    return record, case, historical


@pytest.mark.parametrize(
    "key,value",
    [
        ("missed_original_failures", 1),
        ("non_target_and_unselected_exact", False),
        ("detector_exact", False),
        ("changed_elements", 2),
    ],
)
def test_audit_rejects_missing_coverage_or_preservation(key, value):
    record, case, historical = record_fixture()
    probe.audit_record(record, case, historical)
    record["phases"][0][key] = value
    with pytest.raises(ValueError):
        probe.audit_record(record, case, historical)


def test_audit_keeps_one_ulp_floor_and_historical_output_checks():
    record, case, historical = record_fixture()
    record["phases"][0]["corrected_oracle"]["max_bf16_ulp"] = 2
    with pytest.raises(ValueError, match="one-ULP"):
        probe.audit_record(record, case, historical)
    record, case, historical = record_fixture()
    record["phases"][0]["candidate_sha256"][0] = "changed"
    with pytest.raises(ValueError, match="Q/KV hashes"):
        probe.audit_record(record, case, historical)


def test_matrix_and_timing_are_fixed_before_gpu():
    cases = probe.previous.matrix()
    assert len(cases) == 120
    assert [(r["rank"], r["rows"]) for r in cases[::30]] == [(r, 1) for r in range(4)]
    assert probe.TIMING == {
        "rank": 0,
        "seed": 530901,
        "magnitude": 1.0,
        "repeats": 3,
        "graph_calls": 32,
        "warmup": 10,
    }
    assert probe.TIMING_ROWS == (1, 16, 640, 7616)


def test_real_preparation_does_not_run_expired_historical_freeze(tmp_path):
    if not probe.SCREEN.exists():
        pytest.skip("local campaign evidence absent")
    path = tmp_path / "manifest.json"
    probe.prepare(path)
    manifest = json.loads(path.read_text())
    assert len(manifest["cases"]) == 120 and len(manifest["records"]) == 4
    assert manifest["exponent"] == -12
    probe.verify(manifest)
    with pytest.raises(ValueError, match="preserve prior"):
        probe.prepare(path)


def test_existing_attempt_never_restarted(tmp_path):
    output = tmp_path / "attempt"
    output.mkdir()
    with pytest.raises(ValueError, match="no retry"):
        probe.run(tmp_path / "absent-manifest", output)
