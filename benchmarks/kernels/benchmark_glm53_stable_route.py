# SPDX-License-Identifier: Apache-2.0
"""Fixed isolated A/B/A of origin-stable small-M routing, NOT serving TPS.

32 synthetic shapes: M1..16, random/skew, sigmoid/renormalize/scale2.5/BM8.
Compare against BOTH the installed native router and native + diagnostic sort.
Each comparison uses five A/B/A rounds, three warmup graph replays, and five
timed 20-call graph replays per arm/round. All samples retained; no retries.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.glm53_stable_route_probe import build
from benchmarks.kernels.replay_glm53_indexer import sha, tensor_sha
from slimserve.canonical_moe import canonicalize
from tests.kernels.test_glm53_canonical_moe import expected_alignment
from vllm.model_executor.layers.fused_moe.router.glm_route_align import (
    alignment_geometry,
)
from vllm.quixicore.ops import quixicore_ops as qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability() == (12, 0)
    probe = build()
    import vllm._quixicore_C as native

    sources = [
        Path(__file__),
        Path(probe.__file__),
        Path(native.__file__),
        Path("csrc/quixicore/serving/glm_moe_routing.cuh"),
        Path("benchmarks/kernels/glm53_stable_route_probe.cu"),
        Path("benchmarks/kernels/glm53_stable_route_probe.py"),
        Path("benchmarks/kernels/benchmark_mhc_output_parallel.py"),
        Path("benchmarks/kernels/replay_glm53_indexer.py"),
        Path("tests/kernels/test_glm53_canonical_moe.py"),
        Path("slimserve/canonical_moe.py"),
        Path("slimserve/canonical_moe_kernel.py"),
        Path("vllm/model_executor/layers/fused_moe/router/glm_route_align.py"),
        Path("vllm/quixicore/ops.py"),
    ]
    hashes = {str(p): sha(p) for p in sources}
    result = dict(
        status="running",
        diagnostic_only=True,
        protocol=__doc__,
        source_sha256=hashes,
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

    def bits_equal(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(
            a.contiguous().view(torch.uint8).cpu(),
            b.contiguous().view(torch.uint8).cpu(),
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
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 20)
        return samples

    for tokens in range(1, 17):
        for pattern in ("random", "skew"):
            gen = torch.Generator().manual_seed(53100 + tokens)
            cpu_logits = torch.randn(tokens, 288, generator=gen) * 2
            cpu_bias = torch.randn(288, generator=gen) * 0.5
            if pattern == "skew":
                cpu_logits = cpu_logits[:1].expand_as(cpu_logits).clone()
            logits, bias = cpu_logits.cuda(), cpu_bias.cuda()
            capacity, blocks = alignment_geometry(tokens, 8, 288, 8)

            def native_call(logits=logits, bias=bias, capacity=capacity, blocks=blocks):
                return qc.glm_route_align(
                    logits, bias, 8, 0, True, 2.5, 8, capacity, blocks
                )

            def sorted_call(native_call=native_call, tokens=tokens):
                out = native_call()
                canonicalize(*out[2:], tokens=tokens, block_size=8)
                return out

            def stable_call(logits=logits, bias=bias):
                return probe.run(logits, bias, 0, True, 2.5, 8, True)

            reference = native_call()
            expected = expected_alignment(reference[1], 8)
            graphs = {
                name: graph_for(call)
                for name, call in (
                    ("native", native_call),
                    ("native-plus-sort", sorted_call),
                    ("stable", stable_call),
                )
            }

            def check(
                graphs=graphs,
                reference=reference,
                expected=expected,
                logits=logits,
                bias=bias,
                cpu_logits=cpu_logits,
                cpu_bias=cpu_bias,
            ):
                for name, (graph, outputs) in graphs.items():
                    graph.replay()
                    for out in outputs:
                        for a, b in zip(out[:2], reference[:2]):
                            bits_equal(a, b)
                        for a, b in zip(out[3:], reference[3:]):
                            bits_equal(a, b)
                        if name != "native":
                            for a, b in zip(out[2:], expected):
                                bits_equal(a, b)
                bits_equal(logits, cpu_logits)
                bits_equal(bias, cpu_bias)

            check()
            case = dict(
                tokens=tokens,
                pattern=pattern,
                block_size=8,
                logits_sha256=tensor_sha(cpu_logits),
                bias_sha256=tensor_sha(cpu_bias),
                comparisons=[],
            )
            for baseline in ("native", "native-plus-sort"):
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
                    arm: statistics.median([v for r in rounds for v in r[arm]])
                    for arm in ("control", "candidate", "return_control")
                }
                case["comparisons"].append(
                    dict(baseline=baseline, rounds=rounds, medians_us=medians)
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
            case["correctness"] = "all graph outputs/ID/weight bits/input bytes pass"
            result["cases"].append(case)
            save()
            del graphs, reference, logits, bias
    assert {str(p): sha(p) for p in sources} == hashes, "frozen source/native changed"
    assert len(result["cases"]) == 32
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
