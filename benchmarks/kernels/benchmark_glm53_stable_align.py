# SPDX-License-Identifier: Apache-2.0
"""Fixed large-M alignment A/B/A, isolated warm-cache graph latency only.

14 sizes x random/skew plus the first actual M640 route from each of four
ranks: 32 cases. Compare installed alignment alone and alignment plus canonical
sort separately with the stable probe. Five A/B/A rounds, three warmup graph
replays and five timed 20-call replays per arm: 4,800 samples, no exclusions.
Routing scores/IDs, weights and GEMM arithmetic are outside this experiment.
All buffers are hot; this does not establish model cache effects or serving TPS.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.glm53_stable_align_probe import build
from benchmarks.kernels.replay_glm53_indexer import sha, tensor_sha
from slimserve.canonical_moe import canonicalize
from tests.kernels.test_glm53_canonical_moe import expected_alignment
from tests.kernels.test_glm53_stable_align import inputs
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.glm_route_align import (
    marlin_block_size_m,
)


def graph_for(call):
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [call() for _ in range(20)]
    return graph, outputs


def measure(graph):
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 20)
    return samples


def cases():
    for tokens in (17, 31, 32, 33, 63, 64, 65, 129, 256, 640, 1024, 4096, 7616, 8192):
        for pattern, ids in zip(("random", "skew"), inputs(tokens)[:2]):
            yield pattern, ids, None
    trace = Path("perf/results/2026-09-09/stable-route-quality-diagnostic/trace")
    paths = sorted(trace.glob("model-*.jsonl"))
    assert len(paths) == 4
    for rank, path in enumerate(paths):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        (row,) = [
            r
            for r in rows
            if r["kind"] == "moe_snapshot"
            and r["stage"] == "moe3.router.ids"
            and r["match"] == 1
        ]
        archive = Path(row["path"])
        assert sha(archive) == row["file_sha256"]
        ids = torch.load(archive, map_location="cpu", weights_only=True)
        assert ids.shape == (640, 8)
        yield (
            f"actual-worker-{rank}",
            ids,
            dict(
                path=str(archive),
                sha256=sha(archive),
                trace=str(path),
                trace_sha256=sha(path),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (12, 0)
    probe = build()
    import vllm._C_stable_libtorch as core
    import vllm._moe_C as moe

    sources = [
        Path(__file__),
        Path(probe.__file__),
        Path(core.__file__),
        Path(moe.__file__),
    ]
    sources += list(
        map(
            Path,
            [
                "benchmarks/kernels/glm53_stable_align_probe.cu",
                "benchmarks/kernels/glm53_stable_align_probe.py",
                "benchmarks/kernels/benchmark_mhc_output_parallel.py",
                "benchmarks/kernels/replay_glm53_indexer.py",
                "csrc/quixicore/serving/glm_moe_stable_align.cuh",
                "csrc/libtorch_stable/moe/moe_align_sum_kernels.cu",
                "tests/kernels/test_glm53_stable_align.py",
                "tests/kernels/test_glm53_canonical_moe.py",
                "slimserve/canonical_moe.py",
                "slimserve/canonical_moe_kernel.py",
                "vllm/model_executor/layers/fused_moe/moe_align_block_size.py",
                "vllm/model_executor/layers/fused_moe/router/glm_route_align.py",
                "vllm/_custom_ops.py",
            ],
        )
    )
    hashes = {str(p): sha(p) for p in sources}
    result = dict(
        status="running",
        diagnostic_only=True,
        protocol=__doc__,
        source_sha256=hashes,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_properties=str(torch.cuda.get_device_properties(0)),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
        hardware=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,power.limit",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        cases=[],
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for pattern, cpu_ids, provenance in cases():
        tokens = len(cpu_ids)
        block = marlin_block_size_m(tokens, 8, 288)
        ids = cpu_ids.cuda()
        expected = expected_alignment(cpu_ids, block)

        def atomic(ids=ids, block=block):
            return moe_align_block_size(ids, block, 288)

        def ordered(atomic=atomic, tokens=tokens, block=block):
            out = atomic()
            canonicalize(*out, tokens=tokens, block_size=block)
            return out

        def stable(ids=ids, block=block):
            return probe.run(ids, block)

        graphs = {
            name: graph_for(call)
            for name, call in [
                ("atomic", atomic),
                ("atomic-plus-sort", ordered),
                ("stable", stable),
            ]
        }

        def check(
            graphs=graphs,
            expected=expected,
            tokens=tokens,
            block=block,
            ids=ids,
            cpu_ids=cpu_ids,
        ):
            for name, (graph, outputs) in graphs.items():
                graph.replay()
                for output in outputs:
                    assert all(
                        torch.equal(a.cpu(), b)
                        for a, b in zip(output[1:], expected[1:])
                    )
                    if name != "atomic":
                        assert torch.equal(output[0].cpu(), expected[0])
                    else:
                        # Audit atomic alignment semantically without pretending
                        # its assignment order is stable or timing the audit.
                        copy = output[0].clone()
                        canonicalize(
                            copy, output[1], output[2], tokens=tokens, block_size=block
                        )
                        assert torch.equal(copy.cpu(), expected[0])
            assert torch.equal(ids.cpu(), cpu_ids)

        check()
        case = dict(
            tokens=tokens,
            block_size=block,
            pattern=pattern,
            input_sha256=tensor_sha(cpu_ids),
            provenance=provenance,
            comparisons=[],
        )
        for baseline in ("atomic", "atomic-plus-sort"):
            rounds = []
            for repeat in range(5):
                rounds.append(
                    dict(
                        repeat=repeat + 1,
                        control=measure(graphs[baseline][0]),
                        candidate=measure(graphs["stable"][0]),
                        return_control=measure(graphs[baseline][0]),
                    )
                )
            medians = {
                arm: statistics.median(v for r in rounds for v in r[arm])
                for arm in ("control", "candidate", "return_control")
            }
            case["comparisons"].append(
                dict(
                    baseline=baseline,
                    candidate="stable",
                    rounds=rounds,
                    medians_us=medians,
                )
            )
            print(
                json.dumps(
                    dict(
                        tokens=tokens,
                        pattern=pattern,
                        baseline=baseline,
                        medians_us=medians,
                    )
                ),
                flush=True,
            )
        check()
        case["correctness"] = (
            "all graph outputs, canonical atomic semantics and input bytes pass"
        )
        result["cases"].append(case)
        save()
        del graphs, ids
    assert len(result["cases"]) == 32
    assert hashes == {str(p): sha(p) for p in sources}, "frozen source/native changed"
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
