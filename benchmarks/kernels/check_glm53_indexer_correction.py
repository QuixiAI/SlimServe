# SPDX-License-Identifier: Apache-2.0
"""Fixed all-rank selective LayerNorm correction probe, not a serving change."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks import analyze_glm53_indexer_precision as precision
from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels import check_glm53_attention_norms as norms
from benchmarks.kernels import check_glm53_attention_overwrite as previous
from benchmarks.kernels import check_glm53_kda_gate as common
from benchmarks.kernels.glm53_rmsnorm_geometry import binary_sha
from slimserve.rmsnorm_diagnostic import single_launcher_receipt

ROOT = norms.ROOT
SCREEN = precision.RESULTS / "runtime-control/indexer-cancellation-analysis-v1.json"
SCREEN_SHA = "3474f6692e98b1be257174827dd5387d8d08dedadd647f8c2935a0a18ec25a24"
SCHEMA = "glm53-indexer-correction-v1"
TIMING_ROWS = (1, 16, 640, 7616)
TIMING = dict(rank=0, seed=530901, magnitude=1.0, repeats=3, graph_calls=32, warmup=10)


def prepare(path):
    from benchmarks.kernels import glm53_indexer_correction as kernel

    require(not path.exists(), "preserve prior manifest")
    screen = norms.load_checked(SCREEN, SCREEN_SHA)
    require(
        screen["status"] == "complete" and not screen["gpu_run"], "CPU screen required"
    )
    documents = {
        key: norms.load_checked(precision.RESULTS / name, digest)
        for key, (name, digest) in precision.PINS.items()
    }
    precision.joined_cases(**documents)
    old = documents["manifest"]
    mapped = norms.load_checked(previous.CONTRACTS, previous.CONTRACTS_SHA)
    sources = {}
    for name, digest in {**old["sources"], **mapped["receipts"]}.items():
        current = sha(name)
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
    records = [r for r in old["records"] if r["arm"] == "combo"]
    require(
        [r["rank"] for r in records] == list(range(4)), "four original bundles required"
    )
    for record in records:
        for field in ("source", "debug_source", "ptx"):
            require(
                sha(record[field]) == record[field + "_sha256"],
                "original provenance changed",
            )
        require(
            sha(Path(record["ptx"]).with_suffix(".cubin")) == record["cubin_sha256"],
            "original binary changed",
        )
    paths = [
        Path(__file__),
        Path(kernel.__file__),
        Path(precision.__file__),
        Path(previous.__file__),
        Path(common.__file__),
        ROOT / "perf/glm53-indexer-correction-protocol.md",
        SCREEN,
        previous.CONTRACTS,
        *[precision.RESULTS / p for p, _ in precision.PINS.values()],
    ]
    site = Path(sys.prefix) / "lib/python3.12/site-packages"
    for folder in (
        "triton/runtime",
        "triton/compiler",
        "triton/language",
        "triton/language/extra/cuda",
        "triton/backends/nvidia",
    ):
        paths.extend((site / folder).glob("*.py"))
    paths.extend(
        site / p
        for p in (
            "triton/_C/libtriton.so",
            "triton/backends/nvidia/bin/ptxas",
            "triton/backends/nvidia/bin/ptxas-blackwell",
        )
    )
    sources.update({str(p.resolve()): sha(p) for p in paths})
    manifest = dict(
        schema=SCHEMA,
        sources=sources,
        records=records,
        cases=previous.matrix(),
        exponent=kernel.EXPONENT,
        options=kernel.OPTIONS,
        timing=TIMING,
        timing_rows=list(TIMING_ROWS),
        historical_summary=str(precision.RESULTS / precision.PINS["summary"][0]),
        historical_summary_sha256=precision.PINS["summary"][1],
        original_namespace=old["original_namespace"],
        original_files=old["original_files"],
        weight_sha256=old["weight_sha256"],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    verify(manifest)
    common.save_new(path, manifest)
    print(f"Prepared {len(manifest['cases'])} cases; {len(sources)} frozen receipts")


def verify(manifest):
    from benchmarks.kernels import glm53_indexer_correction as kernel

    require(
        manifest["schema"] == SCHEMA and manifest["cases"] == previous.matrix(),
        "matrix changed",
    )
    require(
        manifest["exponent"] == kernel.EXPONENT == -12
        and manifest["options"] == kernel.OPTIONS,
        "precision policy changed",
    )
    require(
        manifest["timing"] == TIMING and manifest["timing_rows"] == list(TIMING_ROWS),
        "timing policy changed",
    )
    norms.verify(manifest)


def compile_launchers(manifest, output, summary):
    import torch
    import triton
    from triton.compiler import ASTSource

    from benchmarks.kernels.glm53_indexer_correction import OPTIONS, correct_indexer

    launchers = {}
    for rank, record in enumerate(manifest["records"]):
        require(record["rank"] == rank, "rank order")
        with torch.cuda.device(rank), triton.knobs.cache.scope():
            triton.knobs.cache.dir = str(norms.rank_cache(output, rank))
            copied = output / f"source-{rank}" / Path(record["source"]).name
            copied.parent.mkdir()
            shutil.copyfile(record["source"], copied)
            template = norms.load_preserving_provenance(
                copied,
                Path(record["source"]),
                record["info"]["kernel"],
                f"indexer_combo_{rank}",
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
            original = template._precompile_config(config)
            require(
                binary_sha(original) == record["cubin_sha256"],
                "original in-memory binary differs",
            )
            combo = original.make_launcher()
            require(
                single_launcher_receipt(combo) == record["selected"],
                "original key/config differs",
            )
            cubin = (
                norms.rank_cache(output, rank)
                / record["selected"]["hash"]
                / (record["info"]["kernel"] + ".cubin")
            )
            require(
                sha(cubin) == record["cubin_sha256"], "original disk binary differs"
            )
            correction = triton.compile(
                ASTSource(
                    correct_indexer,
                    signature=dict(
                        Packed="*bf16",
                        Gamma="*bf16",
                        Bias="*bf16",
                        Output="*bf16",
                        Selected="*u8",
                        N="i32",
                    ),
                ),
                options=OPTIONS,
            )
            folder = output / f"correction-{rank}"
            folder.mkdir()
            artifacts = {}
            for kind in ("cubin", "ptx", "ttir"):
                value = correction.asm[kind]
                path = folder / f"kernel.{kind}"
                path.write_bytes(value if isinstance(value, bytes) else value.encode())
                artifacts[str(path.relative_to(output))] = sha(path)
            summary["binaries"].append(
                dict(
                    rank=rank,
                    original_source=str(copied.relative_to(output)),
                    original_source_sha256=sha(copied),
                    original_cubin=str(cubin.relative_to(output.resolve())),
                    original_cubin_sha256=sha(cubin),
                    original_selected=record["selected"],
                    correction_hash=correction.hash,
                    correction_artifacts=artifacts,
                    correction_metadata={
                        k: getattr(correction.metadata, k) for k in OPTIONS
                    },
                )
            )
            launchers[rank] = (combo, correction)
    return launchers


def guarded_candidate(combo, correction, data, changed, weights):
    """Original/repeated/changed/replayed inputs with separate selection guards."""
    import torch

    from benchmarks.kernels.glm53_indexer_correction import IndexerOnlyCorrection

    rows = len(data)
    storage = torch.full((rows + 2, 2336), 123.0, dtype=torch.bfloat16, device="cuda")
    x = storage[1:-1]
    wg = [w.cuda() for w in weights]
    buffers = [
        torch.full((rows + 2, stride), 123.0, dtype=torch.bfloat16, device="cuda")
        for _, stride in norms.LAYOUTS.values()
    ]
    outputs = [b[1:-1, :width] for b, width in zip(buffers, norms.LAYOUTS)]
    flags = torch.full((rows + 2, 128), 171, dtype=torch.uint8, device="cuda")
    selected = flags[1:-1]
    adapted = IndexerOnlyCorrection(combo, correction[(rows, 1, 1)], selected)

    def launch():
        adapted(
            x,
            *wg,
            *outputs,
            rows,
            rows,
            rows,
            stream=torch.cuda.current_stream().cuda_stream,
        )

    def snapshot(expected):
        require(torch.equal(x.cpu(), expected), "packed input mutated")
        require(torch.all(storage[[0, -1]] == 123).item(), "input guard mutated")
        require(
            all(torch.equal(w.cpu(), ref) for w, ref in zip(wg, weights)),
            "weights mutated",
        )
        for buffer, width in zip(buffers, norms.LAYOUTS):
            require(
                torch.all(buffer[[0, -1]] == 123).item(), "output row guard mutated"
            )
            require(
                torch.all(buffer[1:-1, width:] == 123).item(),
                "output stride/gate guard mutated",
            )
        require(torch.all(flags[[0, -1]] == 171).item(), "selection guard mutated")
        mask = selected.cpu()
        require(torch.all(mask <= 1).item(), "selection values invalid")
        return [t.cpu() for t in outputs], mask

    def exact(a, b):
        require(
            all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
            and torch.equal(a[1], b[1]),
            "candidate output/selection replay drift",
        )

    x.copy_(data)
    launch()
    first = snapshot(data)
    launch()
    exact(first, snapshot(data))
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    x.copy_(changed)
    graph.replay()
    second = snapshot(changed)
    launch()
    exact(second, snapshot(changed))
    x.copy_(data)
    graph.replay()
    exact(first, snapshot(data))
    require(
        all(not torch.equal(a, b) for a, b in zip(first[0], second[0])),
        "vacuous changed-input replay",
    )
    require(
        adapted.combo_calls == adapted.correction_calls == 4, "adapter host-call count"
    )
    return first, second


def phase_metrics(baseline, candidate, selected, reference, bias):
    import torch

    expected_mask = precision.cancellation_mask(baseline[2], bias, -12)
    require(
        torch.equal(selected.bool(), expected_mask),
        "GPU detector disagrees with CPU predicate",
    )
    require(
        all(torch.equal(a, b) for a, b in zip(baseline[:2], candidate[:2])),
        "Q/KV changed",
    )
    require(
        torch.equal(candidate[2][~expected_mask], baseline[2][~expected_mask]),
        "unselected indexer values changed",
    )
    failures = precision.above_one_ulp(baseline[2], reference)
    return dict(
        baseline_sha256=[norms.tensor_sha(t) for t in baseline],
        candidate_sha256=[norms.tensor_sha(t) for t in candidate],
        selected_sha256=norms.tensor_sha(selected),
        selected_elements=int(expected_mask.sum()),
        selected_rows=int(expected_mask.any(-1).sum()),
        missed_original_failures=int((failures & ~expected_mask).sum()),
        changed_elements=int(
            (candidate[2].view(torch.int16) != baseline[2].view(torch.int16)).sum()
        ),
        original_oracle=norms.compare(baseline[2], reference),
        corrected_oracle=norms.compare(candidate[2], reference),
        mismatches=norms.mismatch_examples(candidate[2], reference),
        detector_exact=True,
        non_target_and_unselected_exact=True,
    )


def execute_case(case, launchers, weights, historical):
    combo, correction = launchers
    inputs = [
        norms.packed_inputs(case["rows"], case["seed"] + 100 * phase, case["magnitude"])
        for phase in range(2)
    ]
    require(
        [norms.tensor_sha(t) for t in inputs] == historical["inputs"],
        "historical inputs differ",
    )
    baseline = norms.run_launches({"combo": combo}, *inputs, weights)
    candidate = guarded_candidate(combo, correction, *inputs, weights)
    phases = []
    for phase, (data, base, (values, selected)) in enumerate(
        zip(inputs, baseline, candidate)
    ):
        require(
            [norms.tensor_sha(t) for t in base]
            == historical["outputs"]["combo"][phase],
            "historical original output differs",
        )
        reference = norms.layernorm_oracle(data[:, 2048:2176], weights[2], weights[3])
        phases.append(phase_metrics(base, values, selected, reference, weights[3]))
    return dict(
        **case,
        input_sha256=historical["inputs"],
        phases=phases,
        replay_guards_mutation_passed=True,
    )


def audit_record(record, case, historical):
    require({k: record[k] for k in case} == case, "case order differs")
    require(record["input_sha256"] == historical["inputs"], "input hashes differ")
    require(
        record["replay_guards_mutation_passed"] is True and len(record["phases"]) == 2,
        "missing replay/phases",
    )
    for phase, item in enumerate(record["phases"]):
        require(
            item["baseline_sha256"] == historical["outputs"]["combo"][phase],
            "original output hashes differ",
        )
        require(
            item["candidate_sha256"][:2] == item["baseline_sha256"][:2],
            "Q/KV hashes differ",
        )
        require(
            item["detector_exact"] is True
            and item["non_target_and_unselected_exact"] is True,
            "preservation checks missing",
        )
        require(
            item["missed_original_failures"] == 0, "detector missed original failure"
        )
        require(
            0 <= item["selected_rows"] <= case["rows"]
            and 0
            <= item["changed_elements"]
            <= item["selected_elements"]
            <= case["rows"] * 128,
            "invalid selected/changed counts",
        )
        require(
            item["corrected_oracle"]["elements"] == case["rows"] * 128
            and item["corrected_oracle"]["max_bf16_ulp"] <= 1,
            "corrected one-ULP oracle failed",
        )


def timing_case(rows, launchers, weights):
    import torch

    from benchmarks.kernels.glm53_indexer_correction import IndexerOnlyCorrection

    combo, correction = launchers
    data = norms.packed_inputs(rows, TIMING["seed"], TIMING["magnitude"])
    x = data.cuda()
    wg = [w.cuda() for w in weights]
    outputs = [
        torch.empty((rows, stride), dtype=torch.bfloat16, device="cuda")[:, :width]
        for width, (_, stride) in norms.LAYOUTS.items()
    ]
    selected = torch.empty((rows, 128), dtype=torch.uint8, device="cuda")
    adapted = IndexerOnlyCorrection(combo, correction[(rows, 1, 1)], selected)
    args = (x, *wg, *outputs, rows, rows, rows)
    graphs = {}
    # Fixed control then correction, three paired readings, no configuration search.
    for name, launch in (("control", combo), ("correction", adapted)):
        for _ in range(TIMING["warmup"]):
            launch(*args, stream=torch.cuda.current_stream().cuda_stream)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(TIMING["graph_calls"]):
                launch(*args, stream=stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(stream)
        graphs[name] = graph
        for _ in range(TIMING["warmup"]):
            graph.replay()
    result = {name: [] for name in graphs}
    for _ in range(TIMING["repeats"]):
        for name, graph in graphs.items():
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            result[name].append(start.elapsed_time(end) * 1000 / TIMING["graph_calls"])
    return dict(
        rows=rows, microseconds_per_bundle=result, includes_selection_writes=True
    )


def run(manifest_path, output):
    require(not output.exists(), "attempt exists; no retry")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3",
        "rank-matched GPUs required",
    )
    require(not common.gpu_query(), "another GPU workload active")
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
        timings=[],
        manifest_sha256=sha(manifest_path),
        gpu_config=common.gpu_config(),
    )
    common.save_new(
        output / "attempt.json",
        dict(pid=os.getpid(), manifest_sha256=sha(manifest_path)),
    )
    try:
        import torch

        torch.set_num_threads(1)
        summary.update(torch=torch.__version__, cuda=torch.version.cuda)
        require(
            torch.cuda.device_count() == 4
            and all(torch.cuda.get_device_capability(r) == (12, 0) for r in range(4)),
            "four SM120 GPUs required",
        )
        weights = norms.load_weights()
        require(
            {k: norms.tensor_sha(w) for k, w in zip(norms.WEIGHTS, weights)}
            == manifest["weight_sha256"],
            "real weights differ",
        )
        historical = norms.load_checked(
            Path(manifest["historical_summary"]), manifest["historical_summary_sha256"]
        )
        launchers = compile_launchers(manifest, output, summary)
        for index, case in enumerate(manifest["cases"]):
            summary["active_case"] = dict(index=index, **case)
            with torch.cuda.device(case["rank"]):
                record = execute_case(
                    case, launchers[case["rank"]], weights, historical["checks"][index]
                )
            path = output / f"case-{index:03d}.json"
            common.save_new(path, record)
            summary["records"].append(dict(path=path.name, sha256=sha(path)))
            audit_record(record, case, historical["checks"][index])
            if (index + 1) % 10 == 0:
                print(f"Indexer correction: {index + 1}/120 cases", flush=True)
        with torch.cuda.device(TIMING["rank"]):
            for rows in TIMING_ROWS:
                summary["timings"].append(
                    timing_case(rows, launchers[TIMING["rank"]], weights)
                )
        verify(manifest)
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        common.save_new(output / "summary.json", summary)


def audit(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    summary = json.loads((output / "summary.json").read_text())
    require(summary["manifest_sha256"] == sha(manifest_path), "manifest changed")
    require(
        not list((output / "triton").rglob("*.autotune.json")), "unexpected autotuning"
    )
    require(
        not common.gpu_query() and common.gpu_config() == summary["gpu_config"],
        "GPU release/config changed",
    )
    historical = norms.load_checked(
        Path(manifest["historical_summary"]), manifest["historical_summary_sha256"]
    )
    for rank, binary in enumerate(summary["binaries"]):
        expected = manifest["records"][rank]
        require(
            binary["rank"] == rank
            and binary["original_selected"] == expected["selected"],
            "binary rank/choice changed",
        )
        require(
            binary["original_cubin_sha256"] == expected["cubin_sha256"]
            and binary["original_source_sha256"] == expected["source_sha256"],
            "original binary/source differs",
        )
        files = {
            binary["original_source"]: binary["original_source_sha256"],
            binary["original_cubin"]: binary["original_cubin_sha256"],
            **binary["correction_artifacts"],
        }
        for name, digest in files.items():
            require(sha(output / name) == digest, "probe artifact changed")
        require(
            binary["correction_metadata"] == manifest["options"],
            "correction compiler policy differs",
        )
    failures, records = [], []
    for index, item in enumerate(summary["records"]):
        require(item["path"] == f"case-{index:03d}.json", "case path/order changed")
        record = norms.load_checked(output / item["path"], item["sha256"])
        try:
            audit_record(record, manifest["cases"][index], historical["checks"][index])
        except ValueError as error:
            failures.append(dict(index=index, error=str(error)))
        records.append(record)
    complete = (
        summary["status"] == "complete"
        and len(records) == 120
        and len(summary["binaries"]) == 4
        and not failures
    )
    require(
        summary["status"] != "complete" or complete,
        "completed probe lacks qualification",
    )
    if complete:
        require(
            [r["rows"] for r in summary["timings"]] == list(TIMING_ROWS),
            "timing matrix incomplete",
        )
    result = dict(
        status="complete" if complete else "terminal-failure",
        cases=len(records),
        failures=failures,
        manifest_sha256=sha(manifest_path),
        summary_sha256=sha(output / "summary.json"),
        sources_verified=len(manifest["sources"]),
        original_files_verified=len(manifest["original_files"]),
        selected_elements=sum(
            p["selected_elements"] for r in records for p in r["phases"]
        ),
        selected_rows=sum(p["selected_rows"] for r in records for p in r["phases"]),
        changed_elements=sum(
            p["changed_elements"] for r in records for p in r["phases"]
        ),
        corrected_oracle_max_ulp=max(
            (
                p["corrected_oracle"]["max_bf16_ulp"]
                for r in records
                for p in r["phases"]
            ),
            default=None,
        ),
        historical_indexer_oracle_pass=False,
        production_qualified=False,
        timings=summary["timings"],
        gpu_processes_after="",
        scope=__doc__,
    )
    common.save_new(output / "analysis.json", result)
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
