#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify an exact-source FA2 architecture rebuild, not serving performance.

Run each binary in a separate, memory-limited image process on an idle GPU.
GLM vision uses noncausal BF16 attention, 16 heads and head dimension 64.
Packed QKV strides, variable lengths, three magnitudes and changed-input graph
replays are checked against an independent chunked CPU FP64 attention oracle.
An optional prior run additionally compares the two binaries' actual outputs.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

LENGTHS = ((1,), (17, 33), (1024,), (1024, 512), (4096,))
MAGNITUDES = (0.01, 1.0, 3.0)
NRMS_LIMIT = 0.006
PEAK_ERROR_LIMIT = 0.015625


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def tensor_digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def oracle(qkv, lengths):
    """No GPU, fused attention, masking library or reduced-precision matmul."""
    outputs = []
    offset = 0
    for length in lengths:
        q, k, v = qkv[offset : offset + length].double().unbind(1)
        q, k, v = (item.transpose(0, 1) for item in (q, k, v))
        chunks = []
        for start in range(0, length, 128):
            scores = (q[:, start : start + 128] @ k.transpose(1, 2)) / 8.0
            chunks.append((scores.softmax(-1) @ v).transpose(0, 1))
        outputs.append(torch.cat(chunks))
        offset += length
    return torch.cat(outputs)


def errors(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"attention shape mismatch: {actual.shape}, {expected.shape}")
    actual, expected = actual.double(), expected.double()
    delta = actual - expected
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    nrms = float(
        delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-30)
    )
    max_abs = float(delta.abs().max())
    peak_error = max_abs / max(float(expected.abs().max()), 1e-30)
    return {
        "finite": finite,
        "nrms": nrms,
        "max_abs": max_abs,
        "error_over_reference_peak": peak_error,
        "exact": bool(torch.equal(actual, expected)),
        "passed": finite and nrms <= NRMS_LIMIT and peak_error <= PEAK_ERROR_LIMIT,
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = args.output / "summary.json"
    report = {
        "status": "running",
        "method": __doc__,
        "command": sys.argv,
        "probe_sha256": digest(Path(__file__)),
        "binary": {"path": str(args.library), "sha256": digest(args.library)},
        "limits": {"nrms": NRMS_LIMIT, "error_over_reference_peak": PEAK_ERROR_LIMIT},
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("CUDA_") or key in ("BASH_ENV", "LD_LIBRARY_PATH")
        },
        "cases": [],
    }

    def save():
        report["loaded_libraries"] = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/" in line
                and any(
                    name in line for name in ("libcuda.", "libcudart.", "_vllm_fa2_C")
                )
            }
        )
        record.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError(f"GPU compute processes already active: {active}")
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("this compatibility probe requires SM120")
        report["gpu"] = torch.cuda.get_device_name()
        torch.ops.load_library(str(args.library))
        op = torch.ops._vllm_fa2_C.varlen_fwd
        report["schema"] = str(op.default._schema)
        prior = None
        if args.reference:
            prior_path = args.reference / "summary.json"
            prior = json.loads(prior_path.read_text())
            if prior["status"] != "complete" or prior["schema"] != report["schema"]:
                raise ValueError(
                    "reference must be complete and have identical operator schema"
                )
            if prior["probe_sha256"] != report["probe_sha256"]:
                raise ValueError("reference probe source differs")
            report["reference"] = {
                "path": str(args.reference),
                "sha256": digest(prior_path),
            }
        save()
        for lengths in LENGTHS:
            for magnitude in MAGNITUDES:
                total = sum(lengths)
                packed = torch.empty(
                    (total, 3, 16, 64), dtype=torch.bfloat16, device="cuda"
                )
                q, k, v = packed.unbind(1)
                cumulative = [0]
                for length in lengths:
                    cumulative.append(cumulative[-1] + length)
                cu = torch.tensor(cumulative, dtype=torch.int32, device="cuda")
                maximum = max(lengths)

                def forward(q=q, k=k, v=v, cu=cu, maximum=maximum):
                    return op(
                        q,
                        k,
                        v,
                        None,
                        cu,
                        cu,
                        None,
                        None,
                        None,
                        None,
                        maximum,
                        maximum,
                        0.0,
                        0.125,
                        False,
                        False,
                        -1,
                        -1,
                        0.0,
                        False,
                        0,
                        None,
                    )[0]

                packed.zero_()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_out = forward()
                for variant in range(2):
                    index = len(report["cases"])
                    row = {
                        "index": index,
                        "lengths": lengths,
                        "magnitude": magnitude,
                        "variant": variant,
                        "status": "running",
                    }
                    report["cases"].append(row)
                    save()
                    generator = torch.Generator().manual_seed(1729 + index)
                    cpu = (
                        torch.randn((total, 3, 16, 64), generator=generator) * magnitude
                    ).bfloat16()
                    row["input_sha256"] = tensor_digest(cpu)
                    packed.copy_(cpu)
                    eager = forward().cpu()
                    graph.replay()
                    replay = graph_out.cpu()
                    reference = oracle(cpu, lengths)
                    row["oracle"] = errors(eager, reference)
                    row["graph_exact"] = bool(torch.equal(eager, replay))
                    filename = f"case-{index}.pt"
                    torch.save(eager, args.output / filename)
                    row["output_file"] = filename
                    row["output_sha256"] = digest(args.output / filename)
                    if prior is not None:
                        old = prior["cases"][index]
                        if row["input_sha256"] != old["input_sha256"]:
                            raise ValueError("cross-binary inputs differ")
                        old_path = args.reference / old["output_file"]
                        if digest(old_path) != old["output_sha256"]:
                            raise ValueError("reference output digest mismatch")
                        row["prior_binary"] = errors(
                            eager, torch.load(old_path, weights_only=True)
                        )
                    passed = row["oracle"]["passed"] and row["graph_exact"]
                    if prior is not None:
                        passed = passed and row["prior_binary"]["passed"]
                    row["status"] = "passed" if passed else "failed"
                    save()
                    print(json.dumps(row), flush=True)
                del graph, graph_out, packed, q, k, v
        report["status"] = (
            "complete"
            if all(row["status"] == "passed" for row in report["cases"])
            else "failed"
        )
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    raise SystemExit(int(run(parser.parse_args())["status"] != "complete"))
