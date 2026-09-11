#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated mHC+RMS fusion probe; never changes serving operators or weights.

The reference matches the generated GLM serving norm: FP32 normalization and
weight multiply, then BF16 output. The Python eager IR has an extra cast that
the cached serving Triton does not retain. Compare compiled behavior explicitly.
Uses all 90 checkpoint mHC sites and their real BF16 norm weights, synthetic
activations, three magnitudes, changed-input graphs, and fixed A/B/A timings.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from contextlib import ExitStack
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_mhc_output_parallel import (
    checkpoint_parameters,
    inputs,
)


def normalize_inplace(x, weight):
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    return x.copy_((xf * inv * weight.float()).to(x.dtype))


def norm_oracle(x, weight):
    xf = x.double()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
    return (xf * inv * weight.double()).bfloat16()


def ulp_distance(a, b):
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or a.shape != b.shape:
        raise ValueError("matching BF16 tensors required")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("finite BF16 tensors required")

    def ordered(t):
        bits = t.view(torch.int16).int() & 0xFFFF
        magnitude = bits & 0x7FFF
        return torch.where(bits >= 0x8000, 0x8000 - magnitude, 0x8000 + magnitude)

    return (ordered(a) - ordered(b)).abs().max().item()


def parameters(model):
    from safetensors import safe_open

    sites, identity = checkpoint_parameters(model)
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with ExitStack() as stack:
        shards = {}
        for i, site in enumerate(sites):
            name = "input_layernorm" if i % 2 == 0 else "post_attention_layernorm"
            key = f"model.language_model.layers.{i // 2}.{name}.weight"
            filename = index[key]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(model / filename, framework="pt", device="cpu")
                )
            weight = shards[filename].get_tensor(key)
            if weight.shape != (4096,) or weight.dtype != torch.bfloat16:
                raise ValueError("expected the actual BF16 GLM norm weight")
            identity[i]["norm"] = key
            identity[i]["norm_sha256"] = hashlib.sha256(
                weight.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            site.append(weight.cuda())
    return sites, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.replays) < 1:
        parser.error("new output and positive rounds/replays required")
    if any(batch not in (1, 2, 4, 8) for batch in args.batch):
        parser.error("only cooperative decode batches 1/2/4/8")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    required_env = {
        "VLLM_DSV4_MHC_MODE": "0",
        "VLLM_DSV4_MHC_COOP_MAX_T": "8",
        "VLLM_DSV4_MHC_SPLITS": "64",
    }
    for key, value in required_env.items():
        if key in os.environ and os.environ[key] != value:
            parser.error(f"this probe requires {key}={value}")
        os.environ[key] = value
    from vllm.quixicore.ops import quixicore_ops as qc

    norm = torch.compile(normalize_inplace, fullgraph=True, dynamic=True)
    root = Path(__file__).resolve().parents[2]
    native = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "vllm").glob("*.so"))
    }
    sites, identity = parameters(args.model)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "native_sha256": native,
        "parameters": identity,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "environment": required_env,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "checks": [],
        "timings": [],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def call(data, site_index, fused_norm):
        x, residual, post, comb, fn, scale, base, weight = data
        constants = [1e-5, 1e-6, 1e-6, 2.0, 20, weight if fused_norm else None, 1e-5]
        if site_index == 0:
            out = [residual, *qc.dsv4_mhc_pre(residual, fn, scale, base, *constants)]
        else:
            out = list(
                qc.dsv4_mhc_fused_post_pre(
                    x, residual, post, comb, fn, scale, base, *constants
                )
            )
        return out

    def compare(reference, candidate, oracle=None):
        row = {
            "mixes_exact": all(
                torch.equal(a, b) for a, b in zip(reference[:3], candidate[:3])
            ),
            "candidate_reference_ulp": ulp_distance(candidate[-1], reference[-1]),
        }
        if oracle is not None:
            row["reference_oracle_ulp"] = ulp_distance(reference[-1].cpu(), oracle)
            row["candidate_oracle_ulp"] = ulp_distance(candidate[-1].cpu(), oracle)
        row["passes"] = row["mixes_exact"] and all(
            value <= 1 for key, value in row.items() if key.endswith("_ulp")
        )
        return row

    try:
        for batch in args.batch:
            data = []
            for i, site in enumerate(sites):
                row = inputs(batch, 911 + i)
                row[4:] = site
                data.append(row)
            for magnitude in (0.125, 1.0, 8.0):
                for i, row in enumerate(data):
                    fresh = inputs(batch, 1201 + i, magnitude)
                    for target, value in zip(row[:4], fresh[:4]):
                        target.copy_(value)
                    reference = call(row, i, False)
                    oracle = norm_oracle(reference[-1].cpu(), row[-1].cpu())
                    norm(reference[-1], row[-1])
                    candidate = call(row, i, True)
                    result["checks"].append(
                        {
                            "batch": batch,
                            "site": i,
                            "magnitude": magnitude,
                            **compare(reference, candidate, oracle),
                        }
                    )
                save()

            graphs, outputs = {}, {}
            for candidate in (False, True):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = []
                    for i, row in enumerate(data):
                        out = call(row, i, candidate)
                        if not candidate:
                            norm(out[-1], row[-1])
                        captured.append(out)
                graphs[candidate], outputs[candidate] = graph, captured
            for replay in range(3):
                for i, row in enumerate(data):
                    fresh = inputs(batch, 1501 + 97 * replay + i)
                    for target, value in zip(row[:4], fresh[:4]):
                        target.copy_(value)
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                for i in range(len(data)):
                    result["checks"].append(
                        {
                            "batch": batch,
                            "site": i,
                            "graph_replay": replay,
                            **compare(outputs[False][i], outputs[True][i]),
                        }
                    )
                save()

            if not args.check_only and all(row["passes"] for row in result["checks"]):
                for graph in graphs.values():
                    for _ in range(3):
                        graph.replay()
                torch.cuda.synchronize()
                samples = []
                for repeat in range(args.rounds):
                    sample = {"round": repeat}
                    for name, candidate in (
                        ("a_us", False),
                        ("b_us", True),
                        ("a2_us", False),
                    ):
                        start, end = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        start.record()
                        for _ in range(args.replays):
                            graphs[candidate].replay()
                        end.record()
                        end.synchronize()
                        sample[name] = (
                            start.elapsed_time(end) * 1000 / (args.replays * 90)
                        )
                    sample["paired_speedup"] = (sample["a_us"] + sample["a2_us"]) / (
                        2 * sample["b_us"]
                    ) - 1
                    samples.append(sample)
                result["timings"].append(
                    {
                        "batch": batch,
                        "samples": samples,
                        "median_paired_speedup": statistics.median(
                            row["paired_speedup"] for row in samples
                        ),
                    }
                )
                print(json.dumps(result["timings"][-1]), flush=True)
                save()
            del graphs, outputs, captured, graph, data
        result["status"] = (
            "complete"
            if all(row["passes"] for row in result["checks"])
            else "failed_gates"
        )
        save()
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save()
        raise
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
