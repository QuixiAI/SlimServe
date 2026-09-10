# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.kernels import check_glm53_attention_overwrite as probe
from benchmarks.kernels.glm53_attention_overwrite import KVOnlyOverwrite


def test_exact_original_abi_stream_order_and_destination():
    calls = []
    args = (*[object() for _ in range(8)], 65, 65, 65)

    def combo(*a, **kw):
        calls.append(("combo", a, kw))
        return "original-result"

    def kv(*a, **kw):
        calls.append(("kv", a, kw))

    wrapped = KVOnlyOverwrite(combo, kv)
    assert wrapped.run(*args, stream=123) == "original-result"
    assert calls == [
        ("combo", args, dict(stream=123)),
        ("kv", (args[0], args[1], args[5], 65, 512), dict(stream=123)),
    ]
    assert wrapped.combo_calls == wrapped.kv_calls == 1


@pytest.mark.parametrize("rows", [(0, 0, 0), (1, 2, 1), (True, True, True)])
def test_bad_rows_fail_before_launch(rows):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not launch")

    wrapped = KVOnlyOverwrite(forbidden, lambda *_: None)
    with pytest.raises(ValueError, match="row counts"):
        wrapped(*([None] * 8), *rows, stream=0)


def test_combo_failure_never_launches_overwrite():
    def failed(*args, **kwargs):
        raise RuntimeError("original failed")

    def forbidden(*args, **kwargs):
        raise AssertionError("must not launch")

    wrapped = KVOnlyOverwrite(failed, forbidden)
    with pytest.raises(RuntimeError, match="original failed"):
        wrapped(*([None] * 8), 1, 1, 1, stream=0)
    assert wrapped.combo_calls == wrapped.kv_calls == 0


def historical_fixture():
    return dict(
        inputs=["input0", "input1"],
        outputs={
            "combo": [["oldkv", "q", "k"]] * 2,
            "split": [["newkv", "q", "newk"]] * 2,
        },
        combo_vs_split=[[dict(bit_mismatches=1)]] * 2,
    )


def record_fixture():
    return dict(
        **probe.matrix()[0],
        input_sha256=["input0", "input1"],
        combo_host_calls=4,
        kv_host_calls=4,
        replay_guards_mutation_passed=True,
        phases=[
            dict(
                baseline_sha256=["oldkv", "q", "k"],
                direct_kv_sha256="newkv",
                candidate_sha256=["newkv", "q", "k"],
                adapter_vs_expected=[
                    dict(elements=w, bit_mismatches=0) for w in (512, 1536, 128)
                ],
                adapter_vs_original=[
                    dict(elements=w, bit_mismatches=int(w == 512))
                    for w in (512, 1536, 128)
                ],
                kv_oracle=dict(max_bf16_ulp=1),
            )
            for _ in range(2)
        ],
    )


def test_auditor_keeps_non_kv_exact_and_kv_oracle_unchanged():
    record, historical = record_fixture(), historical_fixture()
    probe.audit_record(record, 0, historical)
    for slot in (0, 1, 2):
        changed = copy.deepcopy(record)
        changed["phases"][1]["candidate_sha256"][slot] = "wrong"
        with pytest.raises(ValueError, match="adapter output"):
            probe.audit_record(changed, 0, historical)
    changed = copy.deepcopy(record)
    changed["phases"][0]["kv_oracle"]["max_bf16_ulp"] = 2
    with pytest.raises(ValueError, match="KV oracle"):
        probe.audit_record(changed, 0, historical)


def test_failed_record_is_terminal_and_no_retry(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    probe.gate.save_new(
        manifest,
        dict(
            historical_summary="unused",
            historical_summary_sha256="unused",
            sources={},
            records=[],
            original_files={},
        ),
    )
    output = tmp_path / "attempt"
    output.mkdir()
    record = record_fixture()
    record["phases"][0]["kv_oracle"]["max_bf16_ulp"] = 2
    probe.gate.save_new(output / "case-000.json", record)
    probe.gate.save_new(
        output / "summary.json",
        dict(
            status="failed",
            manifest_sha256=probe.sha(manifest),
            gpu_config="fixture",
            binaries=[],
            records=[
                dict(path="case-000.json", sha256=probe.sha(output / "case-000.json"))
            ],
        ),
    )
    monkeypatch.setattr(probe, "verify", lambda _: None)
    monkeypatch.setattr(probe.gate, "gpu_query", lambda: "")
    monkeypatch.setattr(probe.gate, "gpu_config", lambda: "fixture")
    read = probe.contracts.read_pinned
    monkeypatch.setattr(
        probe.contracts,
        "read_pinned",
        lambda p, digest: (
            dict(checks=[historical_fixture()])
            if str(p) == "unused"
            else read(p, digest)
        ),
    )
    probe.audit(manifest, output)
    analysis = json.loads((output / "analysis.json").read_text())
    assert analysis["status"] == "terminal-failure" and len(analysis["failures"]) == 1
    assert not analysis["historical_indexer_oracle_pass"]
    with pytest.raises(ValueError, match="no retry"):
        probe.run(manifest, output)


def test_real_preparation_uses_completed_evidence_without_old_validator(tmp_path):
    if not probe.CONTRACTS.exists():
        pytest.skip("local campaign evidence is not installed")
    path = tmp_path / "manifest.json"
    probe.prepare(path)
    result = json.loads(path.read_text())
    assert len(result["cases"]) == 120 and len(result["records"]) == 8
    probe.verify(result)


def test_preparation_does_not_rebase_archived_graph_hash(tmp_path, monkeypatch):
    if not probe.CONTRACTS.exists():
        pytest.skip("local campaign evidence is not installed")
    mapped = json.loads(probe.CONTRACTS.read_text())
    target = mapped["pairs"][0]["old_graph"]
    original = probe.sha
    monkeypatch.setattr(
        probe, "sha", lambda p: "changed" if str(p) == target else original(p)
    )
    with pytest.raises(ValueError, match="historical compiler/native/evidence changed"):
        probe.prepare(tmp_path / "manifest.json")


@pytest.mark.parametrize("break_replay", [False, True])
def test_real_tensor_adapter_and_graph_orchestration_on_cpu(monkeypatch, break_replay):
    import torch

    # Real tensor math, guards, receipts and run_launches orchestration; only
    # CUDA allocation and recorded graph execution are substituted on CPU.
    for name in ("full", "empty"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self.clone())
    monkeypatch.setattr(torch.Tensor, "cpu", lambda self: self.clone())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=123)
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: object())
    active = []

    class Graph:
        def __init__(self):
            self.calls = []

        def replay(self):
            if not break_replay:
                for call in self.calls:
                    call()

    @contextlib.contextmanager
    def capture(graph, **kwargs):
        active.append(graph)
        yield
        active.pop()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)

    def launch(compute):
        compute()
        if active:
            active[-1].calls.append(compute)

    def combo(x, wkv, wq, wk, bk, kv, q, k, *rows, stream):
        def compute():
            kv.copy_((probe.norms.oracle(x[:, 1536:2048], wkv) + 0.125).bfloat16())
            q.copy_(x[:, :1536])
            k.copy_(x[:, 2048:2176])

        launch(compute)

    def split(x, w, output, *sizes, stream):
        launch(lambda: output.copy_(probe.norms.oracle(x[:, 1536:2048], w)))

    case = probe.matrix()[0]
    weights = [torch.ones(w, dtype=torch.bfloat16) for w in (512, 1536, 128, 128)]
    data = [
        probe.norms.packed_inputs(
            case["rows"], case["seed"] + offset, case["magnitude"]
        )
        for offset in (0, 100)
    ]
    expected = [probe.norms.oracle(x[:, 1536:2048], weights[0]) for x in data]
    baselines = [
        [(kv + 0.125).bfloat16(), x[:, :1536], x[:, 2048:2176]]
        for x, kv in zip(data, expected)
    ]
    historical = dict(
        inputs=[probe.norms.tensor_sha(x) for x in data],
        outputs=dict(
            combo=[[probe.norms.tensor_sha(t) for t in phase] for phase in baselines],
            split=[[probe.norms.tensor_sha(kv)] for kv in expected],
        ),
        combo_vs_split=[
            [probe.norms.compare(old[0], new)] for old, new in zip(baselines, expected)
        ],
    )
    if break_replay:
        with pytest.raises(ValueError, match="changed eager/replay mismatch"):
            probe.execute_case(
                case, dict(combo=combo, split=split), weights, historical
            )
    else:
        result = probe.execute_case(
            case, dict(combo=combo, split=split), weights, historical
        )
        probe.audit_record(result, 0, historical)
        assert all(
            p["adapter_vs_original"][0]["bit_mismatches"] > 0 for p in result["phases"]
        )
