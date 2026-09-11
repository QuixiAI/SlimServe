# SPDX-License-Identifier: Apache-2.0
"""Single-process state/output geometry probes; never serving qualification."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels import check_glm53_kda_gate as gate
from benchmarks.kernels import check_glm53_kda_recompute as recompute
from benchmarks.kernels import glm53_kda_tail as tail
from benchmarks.kernels.check_glm53_cached_rmsnorm import tensor_sha


def prepare(stage, path, predecessor=None):
    require(not path.exists(), "manifest already exists")
    require(stage in tail.NAMES, "unknown stage")
    base = gate.preparation()
    sources = base["sources"]
    for name in (__file__, tail.__file__, recompute.__file__):
        sources[str(Path(name).resolve())] = sha(name)
    completed = gate.ROOT / "perf/results/2026-09-10/kda-recompute-v1/analysis.json"
    gate.read_pinned(
        completed, "eb997c612a0347924b5a66b49c11a90ed969ba37e10ae5d6ac8ce4bf494fd6a9"
    )
    sources[str(completed)] = sha(completed)
    if stage == "output":
        require(predecessor is not None, "completed state audit required")
        previous = json.loads(predecessor.read_text())
        require(
            previous["status"] == "complete"
            and previous["stage"] == "state"
            and previous["pairs"] == len(tail.matrix()),
            "state predecessor incomplete",
        )
        require(previous["gpu_processes_after"] == "", "state GPU not released")
        sources[str(predecessor.resolve())] = sha(predecessor)
        prior_summary_path = predecessor.parent / "summary.json"
        prior_summary = gate.read_pinned(prior_summary_path, previous["summary_sha256"])
        require(
            prior_summary["status"] == "complete"
            and prior_summary["stage"] == "state"
            and len(prior_summary["records"]) == len(tail.matrix()),
            "state summary incomplete",
        )
        sources[str(prior_summary_path.resolve())] = previous["summary_sha256"]
        for index, entry in enumerate(prior_summary["records"]):
            require(entry["path"] == f"pair-{index:03d}.json", "wrong predecessor path")
            pair_path = predecessor.parent / entry["path"]
            pair = gate.read_pinned(pair_path, entry["sha256"])
            audit_record("state", pair, index, prior_summary["binaries"])
            sources[str(pair_path.resolve())] = entry["sha256"]
        for digest in prior_summary["binaries"]:
            file = predecessor.parent / "binaries" / f"{digest}.cubin"
            require(sha(file) == digest, "predecessor binary changed")
            sources[str(file.resolve())] = digest
        prior_manifest = gate.read_pinned(
            Path(previous["manifest"]), previous["manifest_sha256"]
        )
        verify(prior_manifest)
        require(
            prior_summary["manifest_sha256"] == previous["manifest_sha256"],
            "predecessor manifest mismatch",
        )
        sources[str(Path(previous["manifest"]).resolve())] = previous["manifest_sha256"]
    else:
        require(predecessor is None, "state has no predecessor")
    inventory = gate.read_pinned(gate.INVENTORY, gate.INVENTORY_SHA)
    rows = [r for r in inventory["comparisons"] if r["kernel"] == tail.NAMES[stage]]
    require(len(rows) == 4, "incomplete source inventory")
    choices = []
    for row in rows:
        for label in ("original", "fresh_shared"):
            item = row[label]
            gate.read_pinned(item["path"], item["sha256"])
            sources[item["path"]] = item["sha256"]
            choices.append(item["selected"])
    blocks = {"BV": 32} if stage == "state" else {"BK": 64, "BV": 64}
    for config in tail.CONFIGS[stage]:
        require(
            any(
                c["kwargs"] == blocks
                and all(c[key] == value for key, value in config.items())
                for c in choices
            ),
            "unrecorded geometry",
        )
    target_source = (
        "vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py"
        if stage == "state"
        else gate.CHUNK
    )
    for commit in ("59ae0c88f", "ce6df61aa"):
        data = subprocess.check_output(
            ["git", "show", f"{commit}:{target_source}"], cwd=gate.ROOT
        )
        require(
            hashlib.sha256(data).hexdigest() == sha(gate.ROOT / target_source),
            "kernel source differs from historical series",
        )
    manifest = dict(
        schema="kda-tail-v1",
        stage=stage,
        cases=tail.matrix(),
        configs=list(tail.CONFIGS[stage]),
        sources=sources,
        git_commit=base["git_commit"],
        target_source=target_source,
        predecessor=str(predecessor.resolve()) if predecessor else None,
    )
    gate.save_new(path, manifest)
    print(f"Prepared {stage}: {len(tail.matrix())} pairs, {len(sources)} receipts")


def verify(manifest):
    require(manifest["schema"] == "kda-tail-v1", "wrong schema")
    stage = manifest["stage"]
    require(stage in tail.NAMES, "unknown stage")
    require(
        manifest["cases"] == tail.matrix()
        and manifest["configs"] == list(tail.CONFIGS[stage]),
        "plan changed",
    )
    for name, digest in manifest["sources"].items():
        require(sha(name) == digest, f"source/evidence changed: {name}")


def output_receipt(value):
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype).removeprefix("torch."),
        sha256=tensor_sha(value.cpu()),
    )


def execute_case(stage, case, jit, output, binaries):
    import torch

    inputs = tail.make_inputs(stage, case, "cuda")
    protected = {name: t.clone() for name, t in inputs.items()}
    guards, outputs, graphs, bindings = [], [], [], []
    for config in tail.CONFIGS[stage]:
        stores, values = [], []
        for shape, dtype in tail.output_spec(stage, case["lengths"]):
            store = torch.full(
                (math.prod(shape) + 64,),
                123,
                dtype=getattr(torch, dtype),
                device="cuda",
            )
            stores.append(store)
            values.append(store[32:-32].view(shape))
        compiled = tail.launch(stage, jit, inputs, values, config)
        require(
            all(
                getattr(compiled.metadata, key) == value
                for key, value in config.items()
            ),
            "compiled geometry changed",
        )
        digest = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
        if digest not in binaries:
            with (output / "binaries" / f"{digest}.cubin").open("xb") as f:
                f.write(compiled.asm["cubin"])
            binaries[digest] = dict(
                hash=compiled.hash, name=compiled.name, config=config
            )
        first = [t.clone() for t in values]
        tail.launch(stage, jit, inputs, values, config)
        require(
            all(torch.equal(x, y) for x, y in zip(first, values)),
            "eager repeat mismatch",
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tail.launch(stage, jit, inputs, values, config)
        graph.replay()
        require(all(torch.equal(x, y) for x, y in zip(first, values)), "graph mismatch")
        guards.append(stores)
        outputs.append(values)
        graphs.append(graph)
        bindings.append(dict(config=config, cubin_sha256=digest))
    record = dict(**case, bindings=bindings, phases=[])
    for phase in range(2):
        if phase:
            tail.mutate(stage, inputs)
            tail.mutate(stage, protected)
            for graph in graphs:
                graph.replay()
            for config, values in zip(tail.CONFIGS[stage], outputs):
                changed = [t.clone() for t in values]
                tail.launch(stage, jit, inputs, values, config)
                require(
                    all(torch.equal(x, y) for x, y in zip(changed, values)),
                    "changed-input replay mismatch",
                )
        require(
            all(torch.equal(t, protected[name]) for name, t in inputs.items()),
            "input mutation",
        )
        references = (
            tail.state_reference if stage == "state" else tail.output_reference
        )(inputs, case["lengths"])
        exact = (
            tail.exact_reference(stage, case, inputs)
            if case["regime"] != "conditioned"
            else None
        )
        row = dict(
            phase=phase,
            arms=[],
            pairwise=[tail.compare(x, y) for x, y in zip(*outputs)],
        )
        for config, values, stores in zip(tail.CONFIGS[stage], outputs, guards):
            require(
                all(
                    torch.all(t[:32] == 123).item() and torch.all(t[-32:] == 123).item()
                    for t in stores
                ),
                "output guard overwritten",
            )
            row["arms"].append(
                dict(
                    config=config,
                    outputs=[output_receipt(t) for t in values],
                    reference=[
                        tail.reference_metrics(x, y) for x, y in zip(values, references)
                    ],
                    exact=[tail.compare(x, y) for x, y in zip(values, exact)]
                    if exact
                    else None,
                )
            )
        record["phases"].append(row)
        del references, exact
    record["repeat_replay_mutation_guards_passed"] = True
    return record


def exact_passed(record):
    return all(
        m["bit_mismatches"] == 0
        for p in record["phases"]
        for arm in p["arms"]
        for m in (arm["exact"] or [])
    )


def run(manifest_path, output):
    require(not output.exists(), "attempt exists; no retry")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    stage = manifest["stage"]
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU0 required")
    require(not gate.gpu_query(), "another GPU workload active")
    output.mkdir(parents=True)
    (output / "binaries").mkdir()
    os.environ.update(
        TRITON_CACHE_DIR=str(output.resolve() / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(output.resolve() / "inductor"),
        TRITON_CACHE_AUTOTUNING="0",
    )
    summary = dict(
        status="running",
        stage=stage,
        records=[],
        binaries={},
        manifest_sha256=sha(manifest_path),
        gpu_config=gate.gpu_config(),
    )
    gate.save_new(
        output / "attempt.json",
        dict(pid=os.getpid(), manifest_sha256=sha(manifest_path)),
    )
    try:
        import torch

        torch.set_num_threads(4)
        require(torch.cuda.get_device_capability() == (12, 0), "SM120 required")
        jit = tail.jit_for(stage)
        for index, case in enumerate(tail.matrix()):
            summary["active_case"] = dict(index=index, **case)
            record = execute_case(stage, case, jit, output, summary["binaries"])
            record["index"] = index
            filename = f"pair-{index:03d}.json"
            gate.save_new(output / filename, record)
            summary["records"].append(
                dict(path=filename, sha256=sha(output / filename))
            )
            require(exact_passed(record), "exact algebraic oracle failed")
            require(
                all(
                    record["phases"][0]["arms"][arm]["outputs"]
                    != record["phases"][1]["arms"][arm]["outputs"]
                    for arm in range(2)
                ),
                "outputs did not change after input mutation",
            )
            if (index + 1) % 16 == 0:
                print(
                    f"{stage}: completed {index + 1}/{len(tail.matrix())} pairs",
                    flush=True,
                )
        verify(manifest)
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        gate.save_new(output / "summary.json", summary)


def audit_record(stage, record, index, binaries, *, allow_exact_failure=False):
    case = tail.matrix()[index]
    require(
        record["index"] == index and {key: record[key] for key in case} == case,
        "wrong case sequence",
    )
    require(
        record["repeat_replay_mutation_guards_passed"] is True, "launch checks failed"
    )
    require(len(record["bindings"]) == 2, "missing binary bindings")
    for binding, config in zip(record["bindings"], tail.CONFIGS[stage]):
        require(
            binding["config"] == config and binding["cubin_sha256"] in binaries,
            "wrong binary binding",
        )
        require(
            binaries[binding["cubin_sha256"]]["name"] == tail.NAMES[stage],
            "wrong kernel",
        )
    specs = tail.output_spec(stage, case["lengths"])
    require([p["phase"] for p in record["phases"]] == [0, 1], "missing phase")
    for phase in record["phases"]:
        require(
            [a["config"] for a in phase["arms"]] == list(tail.CONFIGS[stage]),
            "wrong arms",
        )
        require(len(phase["pairwise"]) == len(specs), "missing output comparison")
        for arm in phase["arms"]:
            require(
                len(arm["outputs"]) == len(arm["reference"]) == len(specs),
                "incomplete outputs",
            )
            require(
                arm["exact"] is None
                if case["regime"] == "conditioned"
                else isinstance(arm["exact"], list) and len(arm["exact"]) == len(specs),
                "wrong exact/reference classification",
            )
            for output, (shape, dtype) in zip(arm["outputs"], specs):
                require(
                    output["shape"] == list(shape) and output["dtype"] == dtype,
                    "output shape/dtype changed",
                )
        for slot, (shape, _) in enumerate(specs):
            metric = phase["pairwise"][slot]
            require(metric["elements"] == math.prod(shape), "wrong element count")
            count = metric["bit_mismatches"]
            require(
                type(count) is int and 0 <= count <= metric["elements"],
                "invalid mismatches",
            )
            same = (
                phase["arms"][0]["outputs"][slot]["sha256"]
                == phase["arms"][1]["outputs"][slot]["sha256"]
            )
            require(same == (count == 0), "hash/count mismatch")
    if not allow_exact_failure:
        require(exact_passed(record), "exact algebraic oracle failed")


def audit(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    summary = json.loads((output / "summary.json").read_text())
    stage = manifest["stage"]
    require(
        summary["stage"] == stage and summary["manifest_sha256"] == sha(manifest_path),
        "wrong manifest",
    )
    require(
        not gate.gpu_query() and gate.gpu_config() == summary["gpu_config"],
        "GPU release/configuration failed",
    )
    require(not list((output / "triton").rglob("*.autotune.json")), "unexpected tuning")
    for digest in summary["binaries"]:
        require(
            sha(output / "binaries" / f"{digest}.cubin") == digest, "binary changed"
        )
    records = []
    for index, entry in enumerate(summary["records"]):
        require(entry["path"] == f"pair-{index:03d}.json", "wrong record path")
        record = gate.read_pinned(output / entry["path"], entry["sha256"])
        audit_record(
            stage,
            record,
            index,
            summary["binaries"],
            allow_exact_failure=summary["status"] == "failed",
        )
        records.append(record)
    complete = summary["status"] == "complete" and len(records) == len(tail.matrix())
    result = dict(
        status="complete" if complete else "terminal-failure",
        stage=stage,
        pairs=len(records),
        exact_oracles_passed=all(map(exact_passed, records)) if records else None,
        manifest=str(manifest_path.resolve()),
        manifest_sha256=sha(manifest_path),
        summary_sha256=sha(output / "summary.json"),
        sources_verified=len(manifest["sources"]),
        binaries_verified=len(summary["binaries"]),
        elements={
            name: sum(
                p["pairwise"][slot]["elements"] for r in records for p in r["phases"]
            )
            for slot, name in enumerate(tail.OUTPUT_NAMES[stage])
        },
        changed={
            name: sum(
                p["pairwise"][slot]["bit_mismatches"]
                for r in records
                for p in r["phases"]
            )
            for slot, name in enumerate(tail.OUTPUT_NAMES[stage])
        },
        gpu_processes_after="",
        limitation=__doc__,
    )
    gate.save_new(output / "analysis.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("state", "output"))
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        require(args.stage is not None, "stage required")
        prepare(args.stage, args.manifest, args.predecessor)
    else:
        require(args.output is not None, "output required")
        (run if args.action == "run" else audit)(args.manifest, args.output)


if __name__ == "__main__":
    main()
