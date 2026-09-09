#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record/check installed Marlin outputs across decode and prefill tile sizes.

Uses actual layer3 TP4 rank0 checkpoint bytes and the serving auto dispatcher.
Inputs/routes are deliberately synthetic. This is a changed-input graph
regression, not an independent numerical oracle or a serving benchmark.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.profile_glm53_marlin import load_layer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["record", "check"], required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    root = Path(__file__).resolve().parents[2]
    identity = {
        "native_sha256": digest(root / "vllm/_moe_C_stable_libtorch.abi3.so"),
        "checkpoint_index_sha256": digest(args.model / "model.safetensors.index.json"),
        "source_sha256": digest(Path(__file__)),
        "torch": torch.__version__,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    manifest = args.reference_dir / "manifest.json"
    if args.mode == "record":
        args.reference_dir.mkdir(parents=True, exist_ok=False)
        manifest.write_text(json.dumps(identity, indent=2) + "\n")
    else:
        previous = json.loads(manifest.read_text())
        for key in ("checkpoint_index_sha256", "source_sha256", "torch"):
            if previous[key] != identity[key]:
                raise ValueError(f"reference mismatch: {key}")
        if previous["native_sha256"] == identity["native_sha256"]:
            raise ValueError("check requires a different native library")

    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.scalar_type import scalar_types

    torch.set_num_threads(8)
    torch.manual_seed(131)
    weights, weight_identity = load_layer(args.model, 3, 0)
    w13, s13, g13, w2, s2, g2, workspace = weights
    result = {
        "status": "running",
        "mode": args.mode,
        "identity": identity,
        "weights": weight_identity,
        "cases": [],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        # These exercise all five auto-selected M tile sizes: 8/16/32/48/64.
        for batch in (1, 8, 16, 256, 512, 1024, 1536, 2048):
            x = torch.empty(batch, 4096, device="cuda", dtype=torch.bfloat16)
            # Eight distinct experts per row, balanced across all 288 experts.
            ids = ((torch.arange(batch * 8) * 37) % 288).reshape(batch, 8)
            ids = ids.to(device="cuda", dtype=torch.int32)
            routing_weights = torch.full(
                (batch, 8), 2.5 / 8, device="cuda", dtype=torch.float32
            )

            def call(x=x, routing_weights=routing_weights, ids=ids):
                return fused_marlin_moe(
                    x,
                    w13,
                    w2,
                    None,
                    None,
                    s13,
                    s2,
                    routing_weights,
                    ids,
                    scalar_types.float4_e2m1f.id,
                    global_scale1=g13,
                    global_scale2=g2,
                    workspace=workspace,
                    clamp_limit=10.0,
                )

            x.normal_()
            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = call()
            for replay in range(3):
                reference_path = args.reference_dir / f"batch{batch}-replay{replay}.pt"
                if args.mode == "record":
                    x.normal_()
                    reference = None
                else:
                    reference = torch.load(reference_path, weights_only=True)
                    x.copy_(reference["input"])
                eager = call()
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(captured).all():
                    raise ValueError("nonfinite output")
                output = captured.cpu()
                row = {
                    "batch": batch,
                    "replay": replay,
                    "graph_eager_exact": torch.equal(captured, eager),
                }
                if reference is None:
                    torch.save({"input": x.cpu(), "output": output}, reference_path)
                else:
                    row.update(
                        reference_exact=torch.equal(output, reference["output"]),
                        max_abs=(output.float() - reference["output"].float())
                        .abs()
                        .max()
                        .item(),
                    )
                result["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
            del graph, captured, eager
        failed = [
            row
            for row in result["cases"]
            if not row["graph_eager_exact"] or not row.get("reference_exact", True)
        ]
        result["status"] = "failed_gates" if failed else "complete"
        save()
        if failed:
            raise SystemExit(1)
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
