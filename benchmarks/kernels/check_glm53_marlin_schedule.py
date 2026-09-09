#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check a Marlin schedule against an independent checkpoint-byte CPU oracle.

The oracle decodes planar nibbles/scales directly, uses BF16 weight fragments
and FP64 dot products, then the serving BF16 boundaries and GLM clamp/SILU.
It does not use Marlin's repack or dequant helpers. This is not model quality.
"""

import argparse
import hashlib
import json
import subprocess
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_glm53_marlin_schedule import prepare_case
from benchmarks.kernels.profile_glm53_marlin import load_layer, route_cases

AUTO = ((-1, -1, -1), (-1, -1, -1))
CANDIDATE = ((128, 64, 1), (64, 128, 2))


def decode_fragments(packed, scales):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    if scales.shape != (packed.shape[0], codes.shape[1] // 16):
        raise ValueError("unexpected planar group-16 shape")
    values = table[codes] * scales.double().repeat_interleave(16, dim=-1)
    return values.to(torch.bfloat16).double()


def oracle(model, layer, rank, expert_ids, x, routing_weights):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    output = torch.empty(len(expert_ids), 8, 4096, dtype=torch.bfloat16)
    with ExitStack() as stack:
        shards = {}

        def tensor(key):
            filename = index[key]
            if filename not in shards:
                shards[filename] = stack.enter_context(
                    safe_open(model / filename, framework="pt", device="cpu")
                )
            return shards[filename].get_tensor(key)

        def projection(prefix, token, down=False):
            w, s = tensor(prefix + "weight_packed"), tensor(prefix + "weight_scale")
            if down:
                w, s = (
                    w[:, rank * 256 : (rank + 1) * 256],
                    s[:, rank * 32 : (rank + 1) * 32],
                )
            else:
                w, s = (
                    w[rank * 512 : (rank + 1) * 512],
                    s[rank * 512 : (rank + 1) * 512],
                )
            scale = (1.0 / tensor(prefix + "weight_global_scale").float()).item()
            values = decode_fragments(w, s) @ token.double()
            return values, scale

        for token, row in enumerate(expert_ids):
            for route, expert in enumerate(row):
                prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
                gate, gg = projection(prefix + "gate_proj.", x[token])
                up, gu = projection(prefix + "up_proj.", x[token])
                gate = (gate * gg).to(torch.bfloat16).double().clamp(max=10)
                up = (up * gu).to(torch.bfloat16).double().clamp(-10, 10)
                # The installed activation kernel returns BF16 SiLU before the
                # separate up multiply, even in its vectorized clamp path.
                silu = torch.nn.functional.silu(gate).to(torch.bfloat16).double()
                activated = (silu * up).to(torch.bfloat16)
                down, gd = projection(prefix + "down_proj.", activated, down=True)
                # The existing weighted Marlin epilogue rounds the dot product
                # and (global scale * routing weight) separately to BF16, then
                # multiplies BF16 values. Its power-of-two dequant bias cancels
                # and does not change normal-range BF16 rounding. Do not replace
                # those two boundaries with one high-precision final multiply.
                final_scale = (
                    (routing_weights[token, route].float() * gd)
                    .to(torch.bfloat16)
                    .double()
                )
                output[token, route] = (
                    down.to(torch.bfloat16).double() * final_scale
                ).to(torch.bfloat16)
    return output.flatten(0, 1)


def errors(value, ref):
    diff = value.double() - ref.double()
    if not torch.isfinite(diff).all():
        raise ValueError("nonfinite comparison")
    return {
        "max_abs": diff.abs().max().item(),
        "normalized_rms": (
            diff.square().mean().sqrt()
            / ref.double().square().mean().sqrt().clamp_min(1e-12)
        ).item(),
        "changed_fraction": (value != ref).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3, 23, 44])
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.replays < 1
        or any(not 0 <= r < 4 for r in args.ranks)
        or any(not 3 <= layer <= 44 for layer in args.layers)
    ):
        parser.error(
            "new output, positive replays, ranks 0..3 and layers 3..44 required"
        )
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs already have compute processes: {active}")
    torch.set_num_threads(8)
    torch.manual_seed(713)
    _, record = route_cases(args.journal, [1], 1)[0]
    result = {
        "status": "running",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "journal_sha256": hashlib.sha256(args.journal.read_bytes()).hexdigest(),
        "checkpoint_index_sha256": hashlib.sha256(
            (args.model / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cases": [],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        for rank in args.ranks:
            for layer in args.layers:
                weights, _ = load_layer(args.model, layer, rank)
                case = prepare_case(weights, 1, record, layer)
                graphs = {}
                for name, config in (("auto", AUTO), ("candidate", CANDIDATE)):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        case["call"]("moe", config)
                    graphs[name] = graph
                for replay in range(args.replays):
                    case["input"].normal_()
                    reference = oracle(
                        args.model,
                        layer,
                        rank,
                        case["identity"]["expert_ids"],
                        case["input"].cpu(),
                        case["routing_weights"].cpu(),
                    )
                    row = {
                        "rank": rank,
                        "layer": layer,
                        "replay": replay,
                        "expert_ids": case["identity"]["expert_ids"],
                    }
                    result["cases"].append(row)
                    outputs = {}
                    for name, graph in graphs.items():
                        graph.replay()
                        torch.cuda.synchronize()
                        outputs[name] = case["outputs"]["moe"].cpu()
                        row[name] = errors(outputs[name], reference)
                    row["candidate_vs_auto"] = errors(
                        outputs["candidate"], outputs["auto"]
                    )
                    save()
                    # A high-precision oracle changes accumulation order. Require
                    # both implementations to be close, and no material increase
                    # relative to the measured auto-scheduler arithmetic error.
                    assert row["auto"]["normalized_rms"] < 0.002
                    assert row["candidate"]["normalized_rms"] < max(
                        0.0005, row["auto"]["normalized_rms"] * 1.25
                    )
                    torch.testing.assert_close(
                        outputs["candidate"], outputs["auto"], rtol=0.01, atol=0.01
                    )
                    print(json.dumps(row), flush=True)
                del graphs, case, weights
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
