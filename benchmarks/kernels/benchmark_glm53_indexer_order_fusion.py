# SPDX-License-Identifier: Apache-2.0
"""Qualify and time the native selected-pool ordering fusion, not serving TPS.

Replay mode: actual layer23 inputs on each original GPU; both arms receive one
warmup, five eager and five graph calls, all88 outputs archived. Timing mode:
GPU0, fixed five A/B/A rounds for actual prefill and synthetic decode geometry,
both warm-cache graph throughput and separately eviction-conditioned latency.
No automatic retries, source edits, frequency changes or serving activation.
"""

import argparse
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from benchmarks.kernels.compare_cuda_sass import body_sha, function_bodies
from benchmarks.kernels.replay_glm53_indexer import sha, tensor_sha, verified_selection
from benchmarks.kernels.replay_glm53_indexer_ties import (
    canonical_selection,
    replay_device,
)


def verify_assertion_only_changes(before, after):
    """Exact exception for relocated assertion metadata, never selection math."""
    assert before.keys() == after.keys()
    changed = [pc for pc in before if before[pc] != after[pc]]
    assert changed == [0xF0, 0x100, 0x110, 0x130], changed
    assert "@!P0 BRA P1, 0x200" in before[0xE0]
    assert "CALL.ABS.NOINC R2" in before[0x1E0]
    # The unchanged valid-input branch skips [0xf0,0x200). All other instructions
    # and ALL scheduling/control words, including these four, must stay exact.
    for pc in changed:
        assert before[pc].split("\n")[1] == after[pc].split("\n")[1]
    return [dict(pc=hex(pc), before=before[pc], after=after[pc]) for pc in changed]


def verify_generic_decode_qualification(control, candidate, proof, test_source):
    """Accept only the fixed matched test corpus on the exact two binaries."""
    cases = []
    for path, arm in ((control, "before"), (candidate, "candidate")):
        root = ET.parse(path).getroot()
        for suite in root.iter("testsuite"):
            assert all(
                int(suite.get(key, 0)) == 0 for key in ("failures", "errors", "skipped")
            )
        tests = list(root.iter("testcase"))
        assert len(tests) == 32
        for test in tests:
            assert not any(
                test.find(key) is not None for key in ("failure", "error", "skipped")
            )
            props = {
                p.attrib["name"]: p.attrib["value"]
                for p in test.findall("./properties/property")
            }
            assert props["native_sha256"] == proof[arm + "_binary"]["sha256"]
            assert props["test_source_sha256"] == sha(test_source)
            assert test.attrib["name"].startswith(
                "test_single_block_decode_changed_inputs["
            )
        cases.append(sorted((t.attrib["classname"], t.attrib["name"]) for t in tests))
        assert len(set(cases[-1])) == 32
    assert cases[0] == cases[1]
    return dict(
        tests_per_binary=32,
        scope=(
            "generic insertion/radix score sets, bounds and changed-input graphs; "
            "not GLM serving"
        ),
        sources={str(p): sha(p) for p in (control, candidate, test_source)},
    )


def verify_native_comparison(proof, directory, generic_qualification=None):
    assert not proof["removed_functions"]
    name = "arch = sm_120f _ZN4vllm22glm53TopKPerRowPrefillEPKfPKiS3_Pii"
    added = "arch = sm_120f _ZN4vllm22glm53TopKPerRowOrderedEPKfPKiS3_Pii"
    assert proof["before_functions"] == 4188 and proof["candidate_functions"] == 4189
    extra = set(proof["changed_common_functions"]) - {name}
    if extra:
        expected = {
            f"arch = sm_120f _ZN4vllm16topKPerRowDecodeILi512ELb{radix}"
            "ELb0ELb0EEEvPKfPKiPiiiiiiPfiS4_"
            for radix in (0, 1)
        }
        assert extra == expected and generic_qualification is not None
    assert proof["identical_common_functions"] == 4187 - len(extra)
    assert set(proof["changed_common_functions"]) == {name} | extra
    assert set(proof["added_functions"]) == {name, added} | extra

    def read_body(path, arm):
        with path.open() as stream:
            for symbol, body in function_bodies(stream):
                if symbol == name:
                    assert proof["bodies"][name][arm + "_instances"] == {
                        body_sha(body): 1
                    }
                    return body
        raise AssertionError("missing captured selector body")

    before = read_body(directory / "before.sass", "before")
    after = read_body(directory / "candidate.sass", "candidate")
    assert sha(directory / "before.sass") == proof["before_sass"]["sha256"]
    assert sha(directory / "candidate.sass") == proof["candidate_sass"]["sha256"]
    return dict(
        all4187_other_function_copies_identical=not extra,
        identical_other_function_copies=4187 - len(extra),
        generic_decode_codegen_changes=sorted(extra),
        generic_decode_qualification=generic_qualification,
        valid_input_selector_instructions_identical=True,
        assertion_only_changes=verify_assertion_only_changes(before, after),
        raw_sha256={
            name: sha(directory / name) for name in ("before.sass", "candidate.sass")
        },
    )


def load_capture(run, device):
    matches = []
    for path in (run / "trace").glob("index-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        forward = next(e for e in events if e["kind"] == "forward_begin")
        if forward["device"] == f"cuda:{device}":
            matches.append((path, events))
    assert len(matches) == 1
    path, events = matches[0]
    assert events[0]["capture_layer"] == 23
    assert events[0]["selection_ties"] == "smaller-pool-id"
    assert len([e for e in events if e["kind"] == "request_complete"]) == 3
    captured, files = {}, []
    for stage in ("logits", "starts", "ends", "indices"):
        items = [
            e
            for e in events
            if e["kind"] == "tensor"
            and e["chunk"] == 1
            and e["stage"] == f"layer-23.call-0.{stage}"
        ]
        assert [e["match"] for e in items] == [1, 2, 3]
        assert len({e["sha256"] for e in items}) == 1
        item = items[0]
        archive = Path(item["archive"]["path"])
        assert sha(archive) == item["archive"]["sha256"]
        assert archive.stat().st_size == item["archive"]["bytes"]
        value = torch.load(archive, weights_only=True, map_location="cpu")
        assert tensor_sha(value) == item["sha256"]
        captured[stage] = value
        files.append(dict(stage=stage, path=str(archive), sha256=sha(archive)))
    _, ties = verified_selection(**captured)
    assert ties == 1 and captured["logits"].shape == (7616, 1904)
    expected = canonical_selection(*(captured[k] for k in ("logits", "starts", "ends")))
    assert torch.equal(expected, captured["indices"])
    return (
        captured,
        expected,
        dict(
            journal=str(path),
            journal_sha256=sha(path),
            inputs=files,
            canonical_oracle_sha256=tensor_sha(expected),
        ),
    )


def replay(args, result, save, operations, canonicalize):
    for device in range(4):
        captured, expected, provenance = load_capture(args.run, device)
        observations, outputs = [], []
        for label, operation in operations.items():
            rows, matrices = replay_device(
                captured,
                expected,
                device,
                "canonical-ties",
                operation,
                canonicalize if label == "post-sort" else lambda _: None,
            )
            # Both arms require exact CPU-oracle equality in the shared replay.
            for row in rows:
                row["selector"] = label
            observations.extend(rows)
            outputs.extend(matrices)
        archive = args.output / f"gpu-{device}-all22-outputs.pt"
        with archive.open("xb") as stream:
            torch.save(torch.stack(outputs), stream)
        result["replays"].append(
            dict(
                device=device,
                **provenance,
                observations=observations,
                output_archive=dict(
                    path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size
                ),
            )
        )
        save()
        print(
            json.dumps(dict(device=device, verified_calls=len(observations))),
            flush=True,
        )


def timing_cases(captured):
    # Actual first prefill chunk and last583 rows: preserve the real cutoff tie.
    for begin, end in ((0, 7616), (7033, 7616)):
        yield (
            f"actual-prefill-{end - begin}",
            {k: captured[k][begin:end].clone() for k in ("logits", "starts", "ends")},
        )
    # Registered decode has physical262144 columns even for a short context.
    # These are synthetic score distributions, not captured decode inputs.
    for rows in (1, 8, 16, 64):
        for visible in (250, 8192, 32768, 262144):
            generator = torch.Generator().manual_seed(9711 + rows + visible)
            logits = torch.zeros(rows, 262144)
            logits[:, :visible] = torch.randn(rows, visible, generator=generator)
            yield (
                f"synthetic-decode-{rows}-visible-{visible}",
                dict(
                    logits=logits,
                    starts=torch.zeros(rows, dtype=torch.int32),
                    ends=torch.full((rows,), visible, dtype=torch.int32),
                ),
            )


def make_call(operation, label, logits, starts, ends, output, canonicalize):
    rows, columns = logits.shape

    def call():
        operation(logits, starts, ends, output, rows, columns, 1, 512)
        if label == "post-sort":
            canonicalize(output)

    return call


def measure(args, result, save, operations, canonicalize):
    captured, _, provenance = load_capture(args.run, 0)
    result["timing_input_provenance"] = provenance
    eviction = torch.zeros(256 * 2**20 // 4, device="cuda", dtype=torch.float32)
    for name, values in timing_cases(captured):
        expected = canonical_selection(
            *(values[k] for k in ("logits", "starts", "ends"))
        )
        logits, starts, ends = (values[k].cuda() for k in ("logits", "starts", "ends"))
        output = torch.empty_like(expected, device="cuda")
        rows, columns = logits.shape
        graphs = {}
        for label, operation in operations.items():
            call = make_call(
                operation, label, logits, starts, ends, output, canonicalize
            )
            call()
            assert torch.equal(output.cpu(), expected)
            for count in (1, 20):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(count):
                        call()
                graph.replay()
                assert torch.equal(output.cpu(), expected)
                graphs[label, count] = graph
        item = dict(
            name=name,
            shape=[rows, columns],
            expected_sha256=tensor_sha(expected),
            visible_range=[int(ends.min()), int(ends.max())],
            rounds=[],
        )
        result["timings"].append(item)
        for repeat in range(5):
            record = dict(repeat=repeat + 1)
            item["rounds"].append(record)
            for phase, label in (
                ("A", "post-sort"),
                ("B", "fused"),
                ("A2", "post-sort"),
            ):
                for _ in range(3):
                    graphs[label, 20].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(5):
                    graphs[label, 20].replay()
                end.record()
                end.synchronize()
                warm_us = start.elapsed_time(end) * 1000 / 100
                cold_us = []
                for _ in range(10):
                    # Same stream, outside timing: read/write twice the L2 size.
                    # Report as eviction-conditioned, not measured DRAM traffic.
                    eviction.add_(1)
                    start.record()
                    graphs[label, 1].replay()
                    end.record()
                    end.synchronize()
                    cold_us.append(start.elapsed_time(end) * 1000)
                assert torch.equal(output.cpu(), expected)
                record[phase] = dict(warm_us=warm_us, eviction_conditioned_us=cold_us)
                save()
        item["median_us"] = {
            phase: dict(
                warm=statistics.median(r[phase]["warm_us"] for r in item["rounds"]),
                eviction_conditioned=statistics.median(
                    v
                    for r in item["rounds"]
                    for v in r[phase]["eviction_conditioned_us"]
                ),
            )
            for phase in ("A", "B", "A2")
        }
        print(json.dumps(dict(name=name, median_us=item["median_us"])), flush=True)
        save()
        del graphs, graph, call, values, logits, starts, ends, output
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "timing"))
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-comparison", required=True, type=Path)
    parser.add_argument("--generic-decode-control-tests", type=Path)
    parser.add_argument("--generic-decode-candidate-tests", type=Path)
    args = parser.parse_args()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert not active, f"GPU compute processes already active: {active}"
    from slimserve.canonical_indexer import canonicalize, enabled
    from vllm import _custom_ops as ops

    assert not enabled()
    assert torch.cuda.device_count() == 4
    assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(4))
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    root = Path(__file__).resolve().parents[2]
    native = root / "vllm/_C_stable_libtorch.abi3.so"
    receipt = json.loads((args.run / "summary.json").read_text())
    assert receipt["status"] == "complete" and receipt["diagnostic_only"]
    assert receipt["git_commit"] == "8f79f3b5cb45d6ea11d9451938bc43c1791cccda"
    proof = json.loads(args.native_comparison.read_text())
    assert (
        proof["before_binary"]["sha256"]
        == receipt["runtime"]["native_sha256"][str(native.relative_to(root))]
    )
    assert proof["candidate_binary"]["sha256"] == sha(native)
    generic_qualification = None
    if args.generic_decode_control_tests or args.generic_decode_candidate_tests:
        assert args.generic_decode_control_tests and args.generic_decode_candidate_tests
        generic_qualification = verify_generic_decode_qualification(
            args.generic_decode_control_tests,
            args.generic_decode_candidate_tests,
            proof,
            root / "tests/kernels/test_sparse_topk_indices.py",
        )
    binary_check = verify_native_comparison(
        proof, args.native_comparison.parent, generic_qualification
    )
    args.output.mkdir(parents=True, exist_ok=False)
    sources = (
        "benchmarks/kernels/benchmark_glm53_indexer_order_fusion.py",
        "benchmarks/kernels/compare_cuda_sass.py",
        "benchmarks/kernels/replay_glm53_indexer_ties.py",
        "benchmarks/kernels/replay_glm53_indexer.py",
        "slimserve/canonical_indexer.py",
        "slimserve/canonical_indexer_kernel.py",
        "csrc/libtorch_stable/sampler.cu",
        "tests/kernels/test_sparse_topk_indices.py",
        "vllm/_custom_ops.py",
    )

    def environment():
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,clocks.sm,clocks.mem,power.draw",
                "--format=csv,noheader",
            ],
            text=True,
        )

    result = dict(
        status="running",
        diagnostic_only=True,
        mode=args.mode,
        protocol=__doc__,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        sources={name: sha(root / name) for name in sources},
        native_sha256=sha(native),
        torch=torch.__version__,
        source_receipt=dict(
            path=str(args.run / "summary.json"), sha256=sha(args.run / "summary.json")
        ),
        native_comparison=dict(
            path=str(args.native_comparison), sha256=sha(args.native_comparison)
        ),
        binary_check=binary_check,
        environment_before=environment(),
        replays=[],
        timings=[],
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    operations = {
        "post-sort": ops.glm53_top_k_per_row_prefill,
        "fused": ops.glm53_top_k_per_row_ordered,
    }
    (replay if args.mode == "replay" else measure)(
        args, result, save, operations, canonicalize
    )
    assert sha(native) == result["native_sha256"]
    assert all(sha(root / name) == digest for name, digest in result["sources"].items())
    result.update(status="complete", environment_after=environment())
    save()


if __name__ == "__main__":
    main()
