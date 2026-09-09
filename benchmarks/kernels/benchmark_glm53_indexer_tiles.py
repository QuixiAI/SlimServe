#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated tile sweep for GLM pooled-indexer scoring on SM120.

Uses the installed serving Triton function, BF16 synthetic queries/pooled keys,
FP32 head weights, actual long-prefill dimensions and fixed A/B/A rounds.
Rotating inputs exceed L2. No model, quantization or serving dispatcher changes.
Correctness requires all visible scores within rtol/atol 1e-5 of the original
tile, sampled scores within the same tolerance of an independent CPU FP64
oracle, and identical top-512 sets on predetermined rows. Unsupported configs
and every failed gate remain in the output; they are never timed as winners.
"""

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
from pathlib import Path

import torch

BASELINE = (8, 64, 4)


def configurations():
    return list(itertools.product((1, 2, 4, 8, 16), (32, 64, 128), (4, 8)))


def oracle(q, weights, keys, visible, rows, columns):
    query = q[rows].cpu().double()
    weight = weights[rows].cpu().double()
    pooled = keys[0, columns].cpu().double()
    dots = torch.einsum("rhd,pd->rhp", query, pooled) * (128**-0.5)
    values = (dots.clamp_min(0) * weight[:, :, None]).sum(1)
    valid = torch.tensor(columns)[None, :] < (visible[rows].cpu() // 4)[:, None]
    return values, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2202)
    parser.add_argument("--pools", type=int, default=15232)
    parser.add_argument("--prefix", type=int, default=53312)
    parser.add_argument("--fixtures", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--config", type=int, nargs=3, action="append")
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("new output in an existing directory required")
    if min(args.rows, args.pools, args.fixtures, args.rounds, args.replays) < 1:
        parser.error("positive dimensions/counts required")
    if args.prefix < 0 or args.prefix + args.rows > 4 * args.pools:
        parser.error("visibility must fit the pooled-key context")
    if (args.prefix + 1) // 4 < 512:
        parser.error("this top-512 diagnostic requires at least 512 visible pools")
    configs = [tuple(c) for c in args.config] if args.config else configurations()
    if any(c not in configurations() for c in configs):
        parser.error("unsupported requested geometry")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    torch.set_num_threads(4)
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources

    from vllm.model_executor.layers import glm5_next_indexer as gi

    if BASELINE[:2] != (gi._ROW_TILE, gi._POOL_TILE):
        raise RuntimeError("serving baseline changed; update the experiment explicitly")
    visible = torch.arange(1, args.rows + 1, device="cuda", dtype=torch.int32)
    visible += args.prefix
    samples = sorted({0, 1, 3, 7, args.rows // 4, args.rows // 2, args.rows - 1})
    samples = [r for r in samples if r < args.rows]
    columns = sorted({0, 1, 31, 32, 63, 64, 127, 128, 511, args.pools - 1})
    columns = [p for p in columns if p < args.pools]
    fixtures = []
    for i in range(args.fixtures):
        generator = torch.Generator(device="cuda").manual_seed(8103 + i)
        q = torch.randn(
            (args.rows, 32, 128), generator=generator, device="cuda"
        ).bfloat16()
        weights = torch.rand((args.rows, 32), generator=generator, device="cuda") * (
            32**-0.5
        )
        keys = torch.randn(
            (1, args.pools, 128), generator=generator, device="cuda"
        ).bfloat16()
        expected = oracle(q, weights, keys, visible, samples, columns)
        fixtures.append((q, weights, keys, expected))
    input_bytes = sum(t.numel() * t.element_size() for row in fixtures for t in row[:3])
    if input_bytes <= 128 * 1024**2:
        parser.error("rotating inputs must exceed the SM120 128 MiB L2")
    tables = {}
    for rt in {c[0] for c in [BASELINE, *configs]}:
        tiles = ((gi.triton.cdiv(args.rows, rt) + 1 + 15) // 16) * 16
        start = torch.arange(tiles, device="cuda", dtype=torch.int32) * rt
        end = torch.clamp(start + rt, max=args.rows)
        start = torch.where(start < args.rows, start, 0)
        end = torch.where(end > start, end, 0)
        # Empty padded programs must have no query rows, not repeat row zero.
        end[gi.triton.cdiv(args.rows, rt) :] = 0
        tables[rt] = (start, end, torch.zeros_like(start))

    record = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "input_bytes": input_bytes,
        "sample_rows": samples,
        "oracle_columns": columns,
        "baseline": BASELINE,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_source_sha256": hashlib.sha256(
            Path(gi.__file__).read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "triton": gi.triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "results": [],
    }
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(record, indent=2) + "\n")

    def call(config, fixture, output):
        rt, pt, warps = config
        q, weights, keys, _ = fixture
        starts, ends, requests = tables[rt]
        return gi._pooled_logits_matmul_kernel[
            (len(starts), gi.triton.cdiv(args.pools, pt))
        ](
            q,
            weights,
            keys,
            starts,
            ends,
            requests,
            visible,
            output,
            args.pools,
            128**-0.5,
            H=32,
            D=128,
            KP=4,
            RT=rt,
            PT=pt,
            num_warps=warps,
            num_stages=3,
        )

    def capture(config):
        outputs = [
            torch.empty((args.rows, args.pools), device="cuda") for _ in fixtures
        ]
        compiled = None
        for data, output in zip(fixtures, outputs):
            output.fill_(float("nan"))
            compiled = call(config, data, output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for data, output in zip(fixtures, outputs):
                call(config, data, output)
        return graph, outputs, compiled

    def validate(reference, actual):
        maximum = 0.0
        for fixture, expected, got in zip(fixtures, reference, actual):
            for start in range(0, args.rows, 64):
                stop = min(start + 64, args.rows)
                valid = (
                    torch.arange(args.pools, device="cuda")[None, :]
                    < (visible[start:stop] // 4)[:, None]
                )
                a, b = expected[start:stop][valid], got[start:stop][valid]
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
                if a.numel():
                    maximum = max(maximum, (a - b).abs().max().item())
            values, valid = fixture[3]
            observed = got[samples][:, columns].cpu().double()
            torch.testing.assert_close(
                values[valid], observed[valid], atol=1e-5, rtol=1e-5
            )
            mask = (
                torch.arange(args.pools, device="cuda")[None, :]
                < (visible[samples] // 4)[:, None]
            )
            selected = []
            for output in (expected, got):
                scores = output[samples].masked_fill(~mask, float("-inf"))
                selected.append(scores.topk(512).indices.sort(1).values)
            if not torch.equal(*selected):
                raise AssertionError("top-512 sets changed on predetermined rows")
        return {"max_abs": maximum, "top512_sets_exact": True, "oracle_pass": True}

    def time_graph(graph):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / (args.replays * args.fixtures)

    try:
        baseline_graph, reference, _ = capture(BASELINE)
        baseline_graph.replay()
        torch.cuda.synchronize()
        record["baseline_check"] = validate(reference, reference)
        save()
        for config in configs:
            row = {"config": config, "status": "running"}
            record["results"].append(row)
            save()
            try:
                graph, outputs, compiled = capture(config)
            except (OutOfResources, CompilationError) as error:
                row.update(status="unsupported", error=str(error))
                save()
                print(json.dumps(row), flush=True)
                continue
            row["resources"] = {
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
            }
            graph.replay()
            torch.cuda.synchronize()
            try:
                row["check"] = validate(reference, outputs)
            except AssertionError as error:
                row.update(status="failed_gates", error=str(error))
                del graph, outputs
                save()
                print(json.dumps(row), flush=True)
                continue
            for _ in range(3):
                baseline_graph.replay()
                graph.replay()
            torch.cuda.synchronize()
            timings = []
            for i in range(args.rounds):
                a, b, a2 = (
                    time_graph(baseline_graph),
                    time_graph(graph),
                    time_graph(baseline_graph),
                )
                timings.append(
                    {
                        "round": i,
                        "a_us": a,
                        "b_us": b,
                        "a2_us": a2,
                        "speedup": (a + a2) / (2 * b) - 1,
                    }
                )
            row.update(
                status="complete",
                timings=timings,
                median_speedup=statistics.median(r["speedup"] for r in timings),
            )
            save()
            print(json.dumps(row), flush=True)
            del graph, outputs
        record["status"] = (
            "complete_with_rejections"
            if any(r["status"] != "complete" for r in record["results"])
            else "complete"
        )
        save()
    except Exception as error:
        record.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
