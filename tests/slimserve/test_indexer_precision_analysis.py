# SPDX-License-Identifier: Apache-2.0
import copy
from itertools import product

import pytest
import torch

from benchmarks import analyze_glm53_indexer_precision as probe


def closed_evidence():
    checks = []
    for rank, rows, seed, magnitude in product(
        range(4), probe.ROWS, probe.SEEDS, probe.MAGNITUDES
    ):
        failed = rows == 7616 and (seed, magnitude) != (530901, 8.0)
        checks.append(
            dict(
                rank=rank,
                rows=rows,
                seed=seed,
                magnitude=magnitude,
                passed=not failed,
                repeat_graph_guards_mutation_pass=True,
                oracle={
                    arm: [[{"max_bf16_ulp": 2 if failed else 0}]]
                    for arm in ("combo", "split")
                },
                inputs=["first", "second"],
            )
        )
    summary = dict(
        status="failed",
        manifest_sha256=probe.PINS["manifest"][1],
        weights={},
        checks=checks,
    )
    manifest = dict(
        rows=list(probe.ROWS),
        seeds=list(probe.SEEDS),
        magnitudes=list(probe.MAGNITUDES),
        expected_pairs=120,
        weight_sha256={},
    )
    audit = dict(
        status="complete",
        numerical_pass=False,
        manifest_sha256=probe.PINS["manifest"][1],
        summary_sha256=probe.PINS["summary"][1],
        pairs=120,
        failed_cases=[
            {k: r[k] for k in ("rank", "rows", "seed", "magnitude")}
            for r in checks
            if not r["passed"]
        ],
    )
    supplement = dict(
        status="complete",
        manifest_sha256=probe.PINS["manifest"][1],
        summary_sha256=probe.PINS["summary"][1],
        audit_sha256=probe.PINS["audit"][1],
        failed_pairs=20,
        all_four_rank_records_exact=True,
    )
    return dict(summary=summary, audit=audit, manifest=manifest, supplement=supplement)


def test_closed_evidence_deduplicates_only_identical_ranks():
    documents = closed_evidence()
    cases = probe.joined_cases(**documents)
    assert len(cases) == 30
    assert sum(not r["passed"] for r in cases) == 5
    assert all(r["rank"] == 0 for r in cases)
    documents["summary"]["checks"][30]["inputs"][0] = "changed"
    with pytest.raises(ValueError, match="rank numerical drift"):
        probe.joined_cases(**documents)


@pytest.mark.parametrize(
    "document,field,value",
    [
        ("summary", "status", "complete"),
        ("audit", "status", "running"),
        ("audit", "numerical_pass", True),
        ("audit", "manifest_sha256", "changed"),
        ("supplement", "summary_sha256", "changed"),
        ("supplement", "audit_sha256", "changed"),
        ("supplement", "failed_pairs", 19),
        ("supplement", "all_four_rank_records_exact", False),
        ("manifest", "expected_pairs", 119),
        ("manifest", "weight_sha256", {"new": "weight"}),
        ("manifest", "rows", [1]),
    ],
)
def test_changed_joins_and_failed_gate_rejected(document, field, value):
    documents = closed_evidence()
    documents[document][field] = value
    with pytest.raises(ValueError):
        probe.joined_cases(**documents)


@pytest.mark.parametrize("change", ["missing", "reorder", "pass_flag", "guards"])
def test_incomplete_matrix_or_bad_flags_rejected(change):
    documents = closed_evidence()
    checks = documents["summary"]["checks"]
    if change == "missing":
        checks.pop()
    elif change == "reorder":
        checks[0], checks[1] = checks[1], checks[0]
    elif change == "pass_flag":
        checks[0]["passed"] = False
    else:
        checks[0]["repeat_graph_guards_mutation_pass"] = False
    with pytest.raises(ValueError):
        probe.joined_cases(**documents)


def test_pinned_file_digest_checked_before_parsing(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_text('{"status": "complete"}')
    digest = probe.sha(path)
    assert probe.load_checked(path, digest)["status"] == "complete"
    path.write_text("broken JSON")
    with pytest.raises(ValueError, match="receipt changed"):
        probe.load_checked(path, digest)


def test_arithmetic_boundaries_and_chunked_control_preserve_inputs():
    packed = probe.packed_inputs(129, 530901, 0.125)
    x = packed[:, 2048:2176]
    weight = torch.linspace(0.25, 1.75, 128).bfloat16()
    bias = torch.linspace(-1, 1, 128).bfloat16()
    originals = [t.clone() for t in (packed, weight, bias)]
    values = probe.arithmetic_models(x, weight, bias)
    assert set(values) == set(probe.MODELS)
    assert values["fp32_separate"].dtype == torch.float32
    assert values["fp32_fused_affine"].dtype == torch.float32
    assert values["fp64_control"].dtype == torch.float64
    assert torch.equal(
        values["fp32_fused_affine"], values["fp32_norm_fp64_affine"].float()
    )
    d = x.double()
    centered = d - d.mean(-1, keepdim=True)
    norm = centered / (centered.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    torch.testing.assert_close(
        values["fp64_control"],
        norm * weight.double() + bias.double(),
        rtol=1e-14,
        atol=1e-14,
    )
    outputs = probe.modeled_outputs(x, weight, bias)
    assert torch.equal(outputs["fp64_control"], probe.layernorm_oracle(x, weight, bias))
    for name in values:
        assert torch.equal(outputs[name], values[name].bfloat16())
    assert all(torch.equal(a, b) for a, b in zip((packed, weight, bias), originals))


def test_constant_rows_return_bias_without_nonfinite_values():
    x = torch.full((3, 128), 2, dtype=torch.bfloat16)
    weight = torch.ones(128, dtype=torch.bfloat16)
    bias = torch.linspace(-1, 1, 128).bfloat16()
    for value in probe.modeled_outputs(x, weight, bias).values():
        assert torch.equal(value, bias.expand_as(x))


@pytest.mark.parametrize("bad", ["dtype", "width", "weight", "nan"])
def test_invalid_arithmetic_inputs_rejected(bad):
    x = torch.ones(2, 128, dtype=torch.bfloat16)
    weight = bias = torch.ones(128, dtype=torch.bfloat16)
    if bad == "dtype":
        x = x.float()
    elif bad == "width":
        x = x[:, :127]
    elif bad == "weight":
        weight = weight[:127]
    else:
        x[0, 0] = float("nan")
    with pytest.raises(ValueError):
        probe.arithmetic_models(x, weight, bias)


def test_historical_scalar_is_checked_and_both_arms_retained():
    x = torch.ones(3, 128, dtype=torch.bfloat16)
    weight = bias = torch.ones(128, dtype=torch.bfloat16)
    reference = probe.layernorm_oracle(x, weight, bias)
    example = dict(row=1, column=2, actual=1.015625, reference=1.0, bf16_ulp=2)
    failures = dict(count=1, worst=[example])
    case = {
        "mismatch_examples": {
            arm: [[None, None, copy.deepcopy(failures)]] for arm in ("combo", "split")
        }
    }
    result = probe.retained_examples(case, 0, x, weight, bias, reference)
    assert len(result) == 2 and result[0]["fp64_result"] == 1.0
    case["mismatch_examples"]["split"][0][2]["worst"][0]["reference"] = 2.0
    with pytest.raises(ValueError, match="oracle scalar drift"):
        probe.retained_examples(case, 0, x, weight, bias, reference)
    case["mismatch_examples"]["combo"][0][2]["count"] = 17
    with pytest.raises(ValueError, match="truncated historical failures"):
        probe.retained_examples(case, 0, x, weight, bias, reference)


def test_cpu_analysis_requires_hidden_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="hide GPUs"):
        probe.analyze()


def test_failed_attempt_preserved_and_output_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "analysis.json"
    monkeypatch.setattr("sys.argv", ["analysis", "--output", str(output)])

    def fail():
        raise ValueError("fixture failure")

    monkeypatch.setattr(probe, "analyze", fail)
    with pytest.raises(ValueError, match="fixture failure"):
        probe.main()
    original = output.read_bytes()
    assert b'"status": "failed"' in original
    with pytest.raises(FileExistsError):
        probe.main()
    assert output.read_bytes() == original
