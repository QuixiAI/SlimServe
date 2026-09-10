# SPDX-License-Identifier: Apache-2.0
import contextlib
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from benchmarks.kernels import check_glm53_kda_tail as runner
from benchmarks.kernels import glm53_kda_tail as tail


@pytest.mark.parametrize("stage", ["state", "output"])
@pytest.mark.parametrize("failure", [None, "replay", "mutation"])
def test_runner_rehearsal_on_cpu(stage, failure, tmp_path, monkeypatch):
    import torch

    full, make_inputs = torch.full, tail.make_inputs
    monkeypatch.setattr(tail, "make_inputs", lambda s, c, d: make_inputs(s, c, "cpu"))

    def cpu_full(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", cpu_full)
    active = []

    class Graph:
        def __init__(self):
            self.calls, self.replays = [], 0

        def replay(self):
            self.replays += 1
            if failure == "replay" and self.replays > 1:
                return
            for call in self.calls:
                call()

    @contextlib.contextmanager
    def capture(graph):
        active.append(graph)
        yield
        active.pop()

    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class JIT:
        def __getitem__(self, grid):
            def launch(**kwargs):
                def compute():
                    if stage == "state":
                        kwargs["h"].copy_(kwargs["h0"])
                        kwargs["ht"].copy_(kwargs["h0"])
                        kwargs["v_new"].copy_(kwargs["v"])
                        if failure == "mutation":
                            kwargs["k"].add_(0.25)
                    else:
                        kwargs["o"].copy_(kwargs["v"])
                        if failure == "mutation":
                            kwargs["q"].add_(0.25)

                compute()
                if active:
                    active[-1].calls.append(compute)
                config = {key: kwargs[key] for key in ("num_warps", "num_stages")}
                return SimpleNamespace(
                    metadata=SimpleNamespace(**config),
                    asm={"cubin": json.dumps(config).encode()},
                    hash="fixture",
                    name=tail.NAMES[stage],
                )

            return launch

    (tmp_path / "binaries").mkdir()
    bins = {}
    if failure:
        with pytest.raises(ValueError, match="mismatch|mutation"):
            runner.execute_case(stage, tail.matrix()[0], JIT(), tmp_path, bins)
    else:
        record = runner.execute_case(stage, tail.matrix()[0], JIT(), tmp_path, bins)
        assert runner.exact_passed(record)
        record["index"] = 0
        runner.audit_record(stage, record, 0, bins)
        record["phases"][0]["arms"][0]["outputs"][-1]["dtype"] = "float16"
        with pytest.raises(ValueError, match="shape/dtype"):
            runner.audit_record(stage, record, 0, bins)


def fixture_record(index, bins):
    item = tail.matrix()[index]
    specs = tail.output_spec("state", item["lengths"])
    return dict(
        index=index,
        **item,
        repeat_replay_mutation_guards_passed=True,
        bindings=[
            dict(config=config, cubin_sha256=digest)
            for config, digest in zip(tail.CONFIGS["state"], bins)
        ],
        phases=[
            dict(
                phase=phase,
                arms=[
                    dict(
                        config=config,
                        outputs=[
                            dict(
                                shape=list(shape),
                                dtype=dtype,
                                sha256=f"phase{phase}-slot{slot}",
                            )
                            for slot, (shape, dtype) in enumerate(specs)
                        ],
                        reference=[dict(bit_mismatches=0)] * len(specs),
                        exact=None
                        if item["regime"] == "conditioned"
                        else [dict(bit_mismatches=0)] * len(specs),
                    )
                    for config in tail.CONFIGS["state"]
                ],
                pairwise=[
                    dict(elements=math.prod(shape), bit_mismatches=0)
                    for shape, dtype in specs
                ],
            )
            for phase in (0, 1)
        ],
    )


def test_real_preparation_and_full_predecessor_rehearsal(tmp_path):
    if not runner.gate.INVENTORY.exists():
        pytest.skip("local campaign evidence is not installed")
    state = tmp_path / "state.json"
    runner.prepare("state", state)
    runner.verify(json.loads(state.read_text()))
    artifacts = tmp_path / "state"
    (artifacts / "binaries").mkdir(parents=True)
    bins = {}
    for config in tail.CONFIGS["state"]:
        data = json.dumps(config).encode()
        digest = hashlib.sha256(data).hexdigest()
        (artifacts / "binaries" / f"{digest}.cubin").write_bytes(data)
        bins[digest] = dict(name=tail.NAMES["state"], config=config)
    summary = dict(
        status="complete",
        stage="state",
        records=[],
        binaries=bins,
        manifest_sha256=runner.sha(state),
    )
    for index in range(len(tail.matrix())):
        record = fixture_record(index, bins)
        runner.audit_record("state", record, index, bins)
        path = artifacts / f"pair-{index:03d}.json"
        runner.gate.save_new(path, record)
        summary["records"].append(dict(path=path.name, sha256=runner.sha(path)))
    runner.gate.save_new(artifacts / "summary.json", summary)
    predecessor = artifacts / "analysis.json"
    runner.gate.save_new(
        predecessor,
        dict(
            status="complete",
            stage="state",
            pairs=224,
            gpu_processes_after="",
            manifest=str(state),
            manifest_sha256=runner.sha(state),
            summary_sha256=runner.sha(artifacts / "summary.json"),
        ),
    )
    output = tmp_path / "output.json"
    runner.prepare("output", output, predecessor)
    runner.verify(json.loads(output.read_text()))
    (artifacts / "pair-000.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        runner.prepare("output", tmp_path / "bad.json", predecessor)


def test_exact_failure_is_retained_and_cannot_clear_predecessor(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    runner.gate.save_new(manifest, dict(stage="state", sources={}))
    output = tmp_path / "attempt"
    (output / "binaries").mkdir(parents=True)
    bins = {}
    for config in tail.CONFIGS["state"]:
        data = json.dumps(config).encode()
        digest = hashlib.sha256(data).hexdigest()
        (output / "binaries" / f"{digest}.cubin").write_bytes(data)
        bins[digest] = dict(name=tail.NAMES["state"], config=config)
    record = fixture_record(0, bins)
    record["phases"][0]["arms"][0]["exact"][0]["bit_mismatches"] = 1
    with pytest.raises(ValueError, match="exact algebraic"):
        runner.audit_record("state", record, 0, bins)
    runner.gate.save_new(output / "pair-000.json", record)
    runner.gate.save_new(
        output / "summary.json",
        dict(
            status="failed",
            stage="state",
            gpu_config="fixture",
            manifest_sha256=runner.sha(manifest),
            binaries=bins,
            records=[
                dict(path="pair-000.json", sha256=runner.sha(output / "pair-000.json"))
            ],
        ),
    )
    monkeypatch.setattr(runner, "verify", lambda _: None)
    monkeypatch.setattr(runner.gate, "gpu_query", lambda: "")
    monkeypatch.setattr(runner.gate, "gpu_config", lambda: "fixture")
    runner.audit(manifest, output)
    result = json.loads((output / "analysis.json").read_text())
    assert result["status"] == "terminal-failure"
    assert result["pairs"] == 1 and result["exact_oracles_passed"] is False
    with pytest.raises(ValueError, match="no retry"):
        runner.run(manifest, output)
    monkeypatch.setattr(runner.gate, "preparation", lambda: dict(sources={}))
    monkeypatch.setattr(runner.gate, "read_pinned", lambda *_: {})
    with pytest.raises(ValueError, match="state predecessor incomplete"):
        runner.prepare("output", tmp_path / "output.json", output / "analysis.json")
