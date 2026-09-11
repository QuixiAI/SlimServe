# SPDX-License-Identifier: Apache-2.0
"""Qualify only the KV overwrite adapter; neither serving nor LayerNorm accuracy."""

import argparse
import json
import os
import shutil
import subprocess
from itertools import product
from pathlib import Path

from benchmarks import analyze_glm53_attention_contracts as contracts
from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels import check_glm53_attention_norms as norms
from benchmarks.kernels import check_glm53_kda_gate as gate
from benchmarks.kernels import glm53_attention_overwrite as adapter
from benchmarks.kernels.glm53_rmsnorm_geometry import binary_sha
from slimserve.rmsnorm_diagnostic import single_launcher_receipt

CONTRACTS = contracts.RESULTS / "runtime-control/attention-contracts.json"
CONTRACTS_SHA = "8337ae7e4ec383563ed2df0230f21e9424556feccbec3a93a61deea94caff3c8"
SUMMARY = contracts.RESULTS / "attention-norm-rank-private-probe/summary.json"


def matrix():
    return [
        dict(rank=rank, rows=rows, seed=seed, magnitude=mag)
        for rank, rows, seed, mag in product(
            range(4), norms.ROWS, norms.SEEDS, norms.MAGNITUDES
        )
    ]


def prepare(path):
    require(not path.exists(), "preserve prior manifest")
    mapped = contracts.read_pinned(CONTRACTS, CONTRACTS_SHA)
    completed = contracts.read_pinned(contracts.PROBE, contracts.PROBE_SHA)
    previous = contracts.read_pinned(contracts.MANIFEST, completed["manifest_sha256"])
    historical = contracts.read_pinned(SUMMARY, completed["summary_sha256"])
    require(
        completed["pairs"] == 120 and completed["numerical_pass"] is False,
        "wrong predecessor",
    )
    require(len(historical["checks"]) == 120, "incomplete predecessor cases")
    require(len(mapped["pairs"]) == 8, "incomplete graph map")
    for case, check in zip(matrix(), historical["checks"]):
        require(
            {key: check[key] for key in case} == case, "historical case order changed"
        )
    sources = {}
    for name, digest in {**previous["sources"], **mapped["receipts"]}.items():
        current = sha(name)
        # Helpers/docs can evolve after a closed experiment. Keep compiler,
        # checkpoint/native objects and archived raw evidence strictly pinned.
        if (
            name in mapped["receipts"]
            or "/site-packages/" in name
            or not name.endswith((".py", ".md"))
        ):
            require(
                current == digest,
                f"historical compiler/native/evidence changed: {name}",
            )
        sources[name] = current
    for name in (
        __file__,
        adapter.__file__,
        contracts.__file__,
        gate.__file__,
        norms.ROOT / "benchmarks/kernels/glm53_rmsnorm_geometry.py",
        norms.ROOT / "perf/glm53-attention-isolation.md",
        CONTRACTS,
        SUMMARY,
        contracts.PROBE,
        contracts.MANIFEST,
    ):
        sources[str(Path(name).resolve())] = sha(name)
    records = [
        r
        for r in previous["records"]
        if r["arm"] == "combo" or r["info"]["widths"] == [512]
    ]
    require(len(records) == 8, "one combo/one KV source per rank required")
    for rank in range(4):
        require(
            sorted(r["arm"] for r in records if r["rank"] == rank)
            == ["combo", "split"],
            "missing rank source",
        )
    for record in records:
        for field in ("source", "debug_source", "ptx"):
            require(
                sha(record[field]) == record[field + "_sha256"],
                "kernel provenance changed",
            )
        require(
            sha(Path(record["ptx"]).with_suffix(".cubin")) == record["cubin_sha256"],
            "historical cubin changed",
        )
    manifest = dict(
        schema="glm53-kv-overwrite-v1",
        cases=matrix(),
        sources=sources,
        records=records,
        historical_summary=str(SUMMARY),
        historical_summary_sha256=completed["summary_sha256"],
        original_namespace=previous["original_namespace"],
        original_files=previous["original_files"],
        weight_sha256=previous["weight_sha256"],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    verify(manifest)
    gate.save_new(path, manifest)
    print(f"Prepared {len(matrix())} cases, {len(sources)} frozen receipts")


def verify(manifest):
    require(
        manifest["schema"] == "glm53-kv-overwrite-v1" and manifest["cases"] == matrix(),
        "plan changed",
    )
    norms.verify(manifest)


def compile_launchers(manifest, output, summary):
    import torch
    import triton

    by_rank = {}
    for rank in range(4):
        by_rank[rank] = {}
        with torch.cuda.device(rank), triton.knobs.cache.scope():
            triton.knobs.cache.dir = str(norms.rank_cache(output, rank))
            for index, record in enumerate(manifest["records"]):
                if record["rank"] != rank:
                    continue
                copied = output / f"source-{index}" / Path(record["source"]).name
                copied.parent.mkdir()
                shutil.copyfile(record["source"], copied)
                template = norms.load_preserving_provenance(
                    copied,
                    Path(record["source"]),
                    record["info"]["kernel"],
                    f"kv_overwrite_{index}",
                    debug_source=Path(record["debug_source"]),
                )
                saved = record["selected"]["config"]
                config = triton.Config(
                    {
                        k: v
                        for k, v in saved.items()
                        if k not in ("num_warps", "num_stages")
                    },
                    num_warps=saved["num_warps"],
                    num_stages=saved["num_stages"],
                )
                compiled = template._precompile_config(config)
                require(
                    binary_sha(compiled) == record["cubin_sha256"],
                    "actual binary differs",
                )
                launcher = compiled.make_launcher()
                actual = single_launcher_receipt(launcher)
                require(actual == record["selected"], "compiled key/config differs")
                cubin = (
                    norms.rank_cache(output, rank)
                    / actual["hash"]
                    / (record["info"]["kernel"] + ".cubin")
                )
                require(sha(cubin) == record["cubin_sha256"], "disk binary differs")
                summary["binaries"].append(
                    dict(
                        rank=rank,
                        arm=record["arm"],
                        selected=actual,
                        source=str(copied.relative_to(output)),
                        source_sha256=sha(copied),
                        cubin=str(cubin.relative_to(output.resolve())),
                        cubin_sha256=sha(cubin),
                    )
                )
                by_rank[rank][record["arm"]] = launcher
    require(
        len(summary["binaries"]) == 8, "all eight binaries required before numerics"
    )
    return by_rank


def direct_kv(launcher, data, weight):
    import torch

    x, w = data.cuda(), weight.cuda()
    result = torch.empty((len(data), 512), dtype=torch.bfloat16, device="cuda")
    launcher(
        x, w, result, len(data), 512, stream=torch.cuda.current_stream().cuda_stream
    )
    return result.cpu()


def execute_case(case, launchers, weights, historical):
    original = norms.packed_inputs(case["rows"], case["seed"], case["magnitude"])
    changed = norms.packed_inputs(case["rows"], case["seed"] + 100, case["magnitude"])
    baseline = norms.run_launches(
        {"combo": launchers["combo"]}, original, changed, weights
    )
    overwrite = adapter.KVOnlyOverwrite(launchers["combo"], launchers["split"])
    candidate = norms.run_launches({"combo": overwrite}, original, changed, weights)
    record = dict(
        **case,
        phases=[],
        input_sha256=[norms.tensor_sha(t) for t in (original, changed)],
    )
    require(record["input_sha256"] == historical["inputs"], "historical input differs")
    for phase, data in enumerate((original, changed)):
        kv_reference = direct_kv(launchers["split"], data, weights[0])
        expected = [kv_reference, *baseline[phase][1:]]
        base_hash = [norms.tensor_sha(t) for t in baseline[phase]]
        direct_hash = norms.tensor_sha(kv_reference)
        require(
            base_hash == historical["outputs"]["combo"][phase],
            "historical combo output differs",
        )
        require(
            direct_hash == historical["outputs"]["split"][phase][0],
            "historical split KV differs",
        )
        record["phases"].append(
            dict(
                baseline_sha256=base_hash,
                direct_kv_sha256=direct_hash,
                candidate_sha256=[norms.tensor_sha(t) for t in candidate[phase]],
                adapter_vs_expected=[
                    norms.compare(a, b) for a, b in zip(candidate[phase], expected)
                ],
                adapter_vs_original=[
                    norms.compare(a, b)
                    for a, b in zip(candidate[phase], baseline[phase])
                ],
                kv_oracle=norms.compare(
                    candidate[phase][0], norms.oracle(data[:, 1536:2048], weights[0])
                ),
            )
        )
    require(
        overwrite.combo_calls == overwrite.kv_calls == 4,
        "adapter host-call sequence differs",
    )
    record.update(
        combo_host_calls=4, kv_host_calls=4, replay_guards_mutation_passed=True
    )
    return record


def audit_record(record, index, historical):
    case = matrix()[index]
    require({k: record[k] for k in case} == case, "case order differs")
    require(record["input_sha256"] == historical["inputs"], "input receipt differs")
    require(
        record["combo_host_calls"] == record["kv_host_calls"] == 4
        and record["replay_guards_mutation_passed"] is True,
        "adapter launch checks missing",
    )
    require(len(record["phases"]) == 2, "both input phases required")
    for phase, item in enumerate(record["phases"]):
        require(
            item["baseline_sha256"] == historical["outputs"]["combo"][phase],
            "baseline receipt differs",
        )
        require(
            item["direct_kv_sha256"] == historical["outputs"]["split"][phase][0],
            "KV receipt differs",
        )
        expected = [item["direct_kv_sha256"], *item["baseline_sha256"][1:]]
        require(item["candidate_sha256"] == expected, "adapter output differs")
        require(
            len(item["adapter_vs_expected"]) == len(item["adapter_vs_original"]) == 3,
            "missing comparisons",
        )
        for slot, width in enumerate((512, 1536, 128)):
            metric = item["adapter_vs_expected"][slot]
            require(
                metric["elements"] == case["rows"] * width
                and metric["bit_mismatches"] == 0,
                "adapter numerical mismatch",
            )
            metric = item["adapter_vs_original"][slot]
            require(metric["elements"] == case["rows"] * width, "wrong element count")
            require(
                metric["bit_mismatches"]
                == historical["combo_vs_split"][phase][0]["bit_mismatches"]
                if slot == 0
                else metric["bit_mismatches"] == 0,
                "unexpected changed elements",
            )
        require(item["kv_oracle"]["max_bf16_ulp"] <= 1, "KV oracle gate failed")


def run(manifest_path, output):
    require(not output.exists(), "attempt exists; no retry")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3",
        "rank-matched GPUs required",
    )
    require(not gate.gpu_query(), "another GPU workload active")
    output.mkdir(parents=True)
    os.environ.update(
        TRITON_CACHE_DIR=str(output.resolve() / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(output.resolve() / "inductor"),
        TRITON_CACHE_AUTOTUNING="0",
    )
    summary = dict(
        status="running",
        records=[],
        binaries=[],
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
        require(
            torch.cuda.device_count() == 4
            and all(torch.cuda.get_device_capability(r) == (12, 0) for r in range(4)),
            "four SM120 GPUs required",
        )
        weights = norms.load_weights()
        require(
            {key: norms.tensor_sha(w) for key, w in zip(norms.WEIGHTS, weights)}
            == manifest["weight_sha256"],
            "weights differ",
        )
        historical = contracts.read_pinned(
            Path(manifest["historical_summary"]), manifest["historical_summary_sha256"]
        )
        launchers = compile_launchers(manifest, output, summary)
        for index, case in enumerate(matrix()):
            summary["active_case"] = dict(index=index, **case)
            with torch.cuda.device(case["rank"]):
                record = execute_case(
                    case, launchers[case["rank"]], weights, historical["checks"][index]
                )
            path = output / f"case-{index:03d}.json"
            gate.save_new(path, record)
            summary["records"].append(dict(path=path.name, sha256=sha(path)))
            audit_record(record, index, historical["checks"][index])
            if (index + 1) % 10 == 0:
                print(f"KV overwrite: completed {index + 1}/120 cases", flush=True)
        verify(manifest)
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        gate.save_new(output / "summary.json", summary)


def audit(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    summary = json.loads((output / "summary.json").read_text())
    require(summary["manifest_sha256"] == sha(manifest_path), "wrong manifest")
    require(
        not gate.gpu_query() and gate.gpu_config() == summary["gpu_config"],
        "GPU release/config failed",
    )
    require(not list((output / "triton").rglob("*.autotune.json")), "unexpected tuning")
    expected_binaries = {(r["rank"], r["arm"]): r for r in manifest["records"]}
    seen = set()
    for entry in summary["binaries"]:
        binding = (entry["rank"], entry["arm"])
        require(
            binding in expected_binaries and binding not in seen,
            "wrong/duplicate binary binding",
        )
        seen.add(binding)
        expected = expected_binaries[binding]
        require(
            all(
                entry[field] == expected[field]
                for field in ("selected", "source_sha256", "cubin_sha256")
            ),
            "binary binding differs from prescribed source/config",
        )
        for key in ("source", "cubin"):
            require(
                sha(output / entry[key]) == entry[key + "_sha256"],
                "compiled artifact changed",
            )
    historical = contracts.read_pinned(
        Path(manifest["historical_summary"]), manifest["historical_summary_sha256"]
    )
    records, failures = [], []
    for index, entry in enumerate(summary["records"]):
        require(entry["path"] == f"case-{index:03d}.json", "wrong case path")
        record = contracts.read_pinned(output / entry["path"], entry["sha256"])
        try:
            audit_record(record, index, historical["checks"][index])
        except ValueError as exc:
            require(summary["status"] == "failed", "completed run failed audit")
            failures.append(dict(index=index, error=str(exc)))
        records.append(record)
    complete = (
        summary["status"] == "complete"
        and len(records) == 120
        and len(summary["binaries"]) == 8
        and not failures
    )
    result = dict(
        status="complete" if complete else "terminal-failure",
        cases=len(records),
        failures=failures,
        manifest_sha256=sha(manifest_path),
        summary_sha256=sha(output / "summary.json"),
        sources_verified=len(manifest["sources"]),
        original_files_verified=len(manifest["original_files"]),
        binaries_verified=len(summary["binaries"]),
        changed_kv=sum(
            p["adapter_vs_original"][0]["bit_mismatches"]
            for r in records
            for p in r["phases"]
        ),
        historical_indexer_oracle_pass=False,
        production_qualified=False,
        gpu_processes_after="",
        limitation=__doc__,
    )
    gate.save_new(output / "analysis.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    else:
        require(args.output is not None, "output required")
        (run if args.action == "run" else audit)(args.manifest, args.output)


if __name__ == "__main__":
    main()
