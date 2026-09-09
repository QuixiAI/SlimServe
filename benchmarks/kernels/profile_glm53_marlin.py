#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated Marlin counter probe with actual GLM53 weights and captured routes.

Not an end-to-end benchmark. Activations and router weights are synthetic;
expert IDs, checkpoint bytes, TP4 rank-local shapes and Marlin preparation are
real. Profile only CUDA-profiler start/stop regions, with clock control disabled.
Cold-cache counters do not represent all overlap/cache behavior in serving.
"""

import argparse
import hashlib
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open


def route_cases(path, batches, count):
    rows = {batch: [] for batch in batches}
    header = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "header":
                header = row
            elif row["kind"] == "invalid":
                raise ValueError("journal contains an invalid capture")
            elif row["kind"] == "decode" and len(row["request_ids"]) in rows:
                rows[len(row["request_ids"])].append(row)
    if header is None or (
        header["num_layers"],
        header["first_moe_layer"],
        header["num_experts"],
        header["top_k"],
    ) != (45, 3, 288, 8):
        raise ValueError("expected the GLM53 target routing journal")
    selected = []
    for batch in batches:
        if len(rows[batch]) < count:
            raise ValueError(f"insufficient batch-{batch} observations")
        indices = (
            [len(rows[batch]) // 2]
            if count == 1
            else [i * (len(rows[batch]) - 1) // (count - 1) for i in range(count)]
        )
        selected.extend((batch, rows[batch][i]) for i in indices)
    return selected


def load_layer(model, layer_id, rank):
    """Use checkpoint planar bytes, then the SAME Marlin load-time conversion."""
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_nvfp4_moe_layer_for_marlin,
    )

    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    collected = {kind: [] for kind in ("w13", "s13", "g13", "w2", "s2", "g2")}
    with ExitStack() as stack:
        shards = {}

        def tensor(key):
            filename = index[key]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(model / filename, framework="pt", device="cpu")
                )
            return shards[filename].get_tensor(key)

        for expert in range(288):
            prefix = f"model.language_model.layers.{layer_id}.mlp.experts.{expert}."
            gate = prefix + "gate_proj."
            up = prefix + "up_proj."
            down = prefix + "down_proj."
            row = slice(rank * 512, (rank + 1) * 512)
            wg, wu = tensor(gate + "weight_packed"), tensor(up + "weight_packed")
            wd = tensor(down + "weight_packed")
            if (
                wg.shape != (2048, 2048)
                or wu.shape != wg.shape
                or wd.shape != (4096, 1024)
            ):
                raise ValueError("unexpected checkpoint expert dimensions")
            collected["w13"].append(torch.cat([wg[row], wu[row]]))
            collected["s13"].append(
                torch.cat(
                    [
                        tensor(gate + "weight_scale")[row],
                        tensor(up + "weight_scale")[row],
                    ]
                )
            )
            gg, gu = (
                tensor(gate + "weight_global_scale"),
                tensor(up + "weight_global_scale"),
            )
            if not torch.equal(gg, gu):
                raise ValueError(
                    "gate/up global scales differ; do not approximate the loader"
                )
            collected["g13"].append((1.0 / gg.float()).reshape(()))
            collected["w2"].append(wd[:, rank * 256 : (rank + 1) * 256].contiguous())
            collected["s2"].append(
                tensor(down + "weight_scale")[
                    :, rank * 32 : (rank + 1) * 32
                ].contiguous()
            )
            collected["g2"].append(
                (1.0 / tensor(down + "weight_global_scale").float()).reshape(())
            )
        raw = {key: torch.stack(value) for key, value in collected.items()}
    identity = {
        key: {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256": hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }
        for key, t in raw.items()
    }
    device = {key: t.cuda() for key, t in raw.items()}
    layer = SimpleNamespace(
        num_experts=288,
        hidden_size=4096,
        intermediate_size_per_partition=512,
        params_dtype=torch.bfloat16,
    )
    w13, s13, g13, w2, s2, g2 = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        device["w13"],
        device["s13"],
        device["g13"],
        device["w2"],
        device["s2"],
        device["g2"],
        True,
    )
    return (w13, s13, g13, w2, s2, g2, layer.workspace), identity


def run_case(weights, batch, record, layer_id):
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        _fused_marlin_moe,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.scalar_type import scalar_types

    ids = torch.tensor(
        [r[layer_id - 3] for r in record["routes"]], device="cuda", dtype=torch.int32
    )
    if ids.shape != (batch, 8) or ids.min() < 0 or ids.max() >= 288:
        raise ValueError("invalid selected routing dimensions or expert IDs")
    ordered = ids.sort(-1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("duplicate selected expert IDs")
    sorted_ids, expert_ids, padded = moe_align_block_size(ids, 8, 288)
    x = torch.randn(batch, 4096, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.full(
        (batch, 8), 2.5 / 8, device="cuda", dtype=torch.float32
    )
    w13, s13, g13, w2, s2, g2, workspace = weights

    def call():
        return _fused_marlin_moe(
            x,
            w13,
            w2,
            None,
            None,
            s13,
            s2,
            routing_weights,
            8,
            scalar_types.float4_e2m1f,
            False,
            None,
            8,
            sorted_ids,
            expert_ids,
            padded,
            topk_ids=ids,
            global_scale1=g13,
            global_scale2=g2,
            workspace=workspace,
            clamp_limit=10.0,
        )

    for _ in range(3):
        eager = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0.01, atol=0.01)
    if not torch.isfinite(captured).all():
        raise ValueError("nonfinite isolated Marlin output")
    label = f"layer{layer_id}-batch{batch}-step{record['step']}"
    torch.cuda.nvtx.range_push(label)
    torch.cuda.profiler.start()
    measured = call()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    torch.cuda.nvtx.range_pop()
    torch.testing.assert_close(measured, eager, rtol=0.01, atol=0.01)
    return {
        "label": label,
        "layer": layer_id,
        "batch": batch,
        "step": record["step"],
        "expert_ids": ids.cpu().tolist(),
        "unique_experts": ids.unique().numel(),
        "padded_rows": padded.item(),
        "request_ids": record["request_ids"],
        "computed_tokens": record["computed_tokens"],
        "finite": True,
        "eager_graph_max_abs": (captured.float() - eager.float()).abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 23, 44])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--cases-per-batch", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.rank < 4 or args.cases_per_batch < 1:
        parser.error("output must be new; rank 0..3; cases per batch positive")
    if any(not 3 <= layer <= 44 for layer in args.layers) or any(
        not 1 <= b <= 16 for b in args.batch
    ):
        parser.error("only target MoE layers 3..44 and batches 1..16")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    cases = route_cases(args.journal, args.batch, args.cases_per_batch)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "journal_sha256": hashlib.sha256(args.journal.read_bytes()).hexdigest(),
        "checkpoint_index_sha256": hashlib.sha256(
            (args.model / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "layers": {},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        torch.manual_seed(103)
        for layer in args.layers:
            weights, identity = load_layer(args.model, layer, args.rank)
            result["layers"][layer] = identity
            save()
            for batch, record in cases:
                row = run_case(weights, batch, record, layer)
                result["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
            del weights
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
