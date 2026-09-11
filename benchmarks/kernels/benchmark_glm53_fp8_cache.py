#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Actual-weight FP8 shared-expert cache sensitivity, not serving throughput.

Compare the old 24-copy rotation against a byte-sized rotation exceeding three
SM120 L2 capacities, then return to 24 copies. Same installed production kernel,
same checkpoint weight, synthetic BF16 inputs. All fixed A/B/A samples survive.
Run only on an idle GPU under an external memory cap, never beside serving.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

L2_BYTES = 128 * 2**20


def rotation_count(weight_bytes):
    if weight_bytes <= 0:
        raise ValueError("positive weight bytes required")
    return 3 * L2_BYTES // weight_bytes + 1


def load_weights(path, layer, rank):
    """Same TP4 row/column slices and gate-up order as the shared expert."""
    prefix = f"model.language_model.layers.{layer}.mlp.shared_experts."
    with safe_open(path, framework="pt", device="cpu") as source:

        def tensor(suffix):
            return source.get_tensor(prefix + suffix)

        row = slice(rank * 512, (rank + 1) * 512)
        scale_row = slice(rank * 4, (rank + 1) * 4)
        gate, up = tensor("gate_proj.weight"), tensor("up_proj.weight")
        down = tensor("down_proj.weight")
        if any(weight.dtype != torch.float8_e4m3fn for weight in (gate, up, down)):
            raise ValueError("expected actual FP8 shared-expert weights")
        if (
            gate.shape != (2048, 4096)
            or up.shape != gate.shape
            or down.shape != (4096, 2048)
        ):
            raise ValueError("unexpected actual shared-expert checkpoint shapes")
        gate_up = torch.cat(
            [gate[row].view(torch.uint8), up[row].view(torch.uint8)]
        ).view(torch.float8_e4m3fn)
        scales = torch.cat(
            [
                tensor("gate_proj.weight_scale")[scale_row],
                tensor("up_proj.weight_scale")[scale_row],
            ]
        )
        return {
            "gate_up": (gate_up, scales),
            "down": (
                down[:, row].contiguous(),
                tensor("down_proj.weight_scale")[:, scale_row].contiguous(),
            ),
        }


def oracle_weight(q, scale):
    n, k = q.shape
    # Match the defined FP32 product -> BF16 weight, then independent FP64 GEMM.
    return (
        (q.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None])
        .reshape(n, k)
        .bfloat16()
        .double()
    )


def digest_tensor(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 23, 44])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=16)
    args = parser.parse_args()
    if (
        not 0 <= args.rank < 4
        or min(args.rounds, args.replays) < 1
        or any(not 3 <= layer <= 44 for layer in args.layers)
        or any(not 1 <= batch <= 16 for batch in args.batches)
    ):
        parser.error("TP4 rank0..3, MoE layers3..44, batch1..16, positive counts")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPU compute processes already active: {active}")
    args.output.mkdir(parents=True, exist_ok=False)
    record = args.output / "summary.json"
    result = {
        "status": "running",
        "diagnostic_only": True,
        "method": __doc__,
        "command": sys.argv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "l2_bytes": L2_BYTES,
        "weights": [],
        "cases": [],
    }

    def save():
        record.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        from vllm.quixicore.ops import quixicore_ops

        torch.set_num_threads(4)
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("requires the SM120 128-MiB-L2 target")
        result["gpu"] = torch.cuda.get_device_name()
        with Path("vllm/_quixicore_C.cpython-312-x86_64-linux-gnu.so").open(
            "rb"
        ) as handle:
            result["native_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        sidecar = args.model / "fp8-swapset.safetensors"
        for layer in args.layers:
            for kind, (cpu_q, cpu_s) in load_weights(sidecar, layer, args.rank).items():
                weight_bytes = cpu_q.numel() * cpu_q.element_size()
                count = rotation_count(weight_bytes)
                identity = {
                    "layer": layer,
                    "kind": kind,
                    "shape": list(cpu_q.shape),
                    "weight_sha256": digest_tensor(cpu_q),
                    "scale_sha256": digest_tensor(cpu_s),
                    "weight_bytes": weight_bytes,
                    "old_rotation_bytes": 24 * weight_bytes,
                    "streaming_copies": count,
                    "streaming_weight_bytes": count * weight_bytes,
                }
                result["weights"].append(identity)
                save()
                banks = [(cpu_q.cuda(), cpu_s.cuda()) for _ in range(count)]
                if len({q.data_ptr() for q, _ in banks}) != count:
                    raise ValueError("weight rotation aliases allocations")
                dequant = oracle_weight(cpu_q, cpu_s)
                for batch in args.batches:
                    generator = torch.Generator().manual_seed(3100 + layer * 31 + batch)
                    cpu_x = torch.randn(
                        batch, cpu_q.shape[1], generator=generator
                    ).bfloat16()
                    x = cpu_x.cuda()
                    expected = cpu_x.double() @ dequant.t()
                    eager = quixicore_ops.decode_gemm_fp8(x, *banks[0]).cpu()
                    delta = eager.double() - expected
                    nrms = float(
                        delta.square().mean().sqrt()
                        / expected.square().mean().sqrt().clamp_min(1e-30)
                    )
                    peak = float(
                        (
                            delta.abs()
                            / expected.abs().amax(1, keepdim=True).clamp_min(1e-30)
                        ).max()
                    )
                    row = {
                        "layer": layer,
                        "kind": kind,
                        "batch": batch,
                        "status": "running",
                        "input_sha256": digest_tensor(cpu_x),
                        "oracle_nrms": nrms,
                        "oracle_row_peak_error": peak,
                        "rounds": [],
                    }
                    result["cases"].append(row)
                    save()
                    if not torch.isfinite(eager).all() or nrms > 0.004 or peak > 2**-7:
                        raise ValueError("independent FP64 GEMM oracle failed")
                    graphs = {}
                    for name, copies in (("old24", 24), ("streaming", count)):
                        for _ in range(3):
                            quixicore_ops.decode_gemm_fp8(x, *banks[0])
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            outputs = [
                                quixicore_ops.decode_gemm_fp8(x, *banks[i])
                                for i in range(copies)
                            ]
                        graph.replay()
                        torch.cuda.synchronize()
                        if not all(torch.equal(out.cpu(), eager) for out in outputs):
                            raise ValueError("rotation graph differs from eager")
                        graphs[name] = (graph, outputs, copies)
                    for repeat in range(args.rounds):
                        sample = {"repeat": repeat + 1}
                        row["rounds"].append(sample)
                        for phase, name in (
                            ("A", "old24"),
                            ("B", "streaming"),
                            ("A2", "old24"),
                        ):
                            graph, outputs, copies = graphs[name]
                            # Changed inputs, exact equality: multiplication by -1.
                            x.copy_(cpu_x if repeat % 2 == 0 else -cpu_x)
                            graph.replay()
                            torch.cuda.synchronize()
                            begin, end = (
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                            begin.record()
                            for _ in range(args.replays):
                                graph.replay()
                            end.record()
                            end.synchronize()
                            sample[phase] = (
                                begin.elapsed_time(end) * 1000 / (args.replays * copies)
                            )
                            expected_eager = eager if repeat % 2 == 0 else -eager
                            if not all(
                                torch.equal(out.cpu(), expected_eager)
                                for out in outputs
                            ):
                                raise ValueError(
                                    "changed-input rotation graph mismatch"
                                )
                            save()
                    row["median_us"] = {
                        phase: statistics.median(
                            sample[phase] for sample in row["rounds"]
                        )
                        for phase in ("A", "B", "A2")
                    }
                    row["status"] = "complete"
                    save()
                    print(json.dumps(row), flush=True)
                    del graphs, graph, outputs
                del banks, dequant
        result["status"] = "complete"
    except BaseException as error:
        result["status"], result["error"] = "failed", repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
