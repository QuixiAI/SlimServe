#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rejected Phase 4.1 cross-item pipeline versus drain control and Marlin.

Fixed batch1/TP4 shapes and NT16/K512/four-stage geometry; no parameter sweep.
Actual checkpoint weights and captured routes, synthetic BF16 inputs. Both GEMMs
are unweighted here to isolate staging; this is not a fused MoE or serving result.
Forty-five independently allocated fixtures exceed three SM120 L2 capacities
for each projection separately; these are data fixtures, not kernel variants.
"""

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from benchmarks.kernels.check_glm53_marlin_schedule import decode_fragments, errors
from benchmarks.kernels.profile_glm53_marlin import route_cases


def load_selected(model, index, layer, experts):
    """Rank0 slices, actual global-scale convention, gate|up concatenation."""
    result = {key: [] for key in ("w13", "s13", "g13", "w2", "s2", "g2")}
    with ExitStack() as stack:
        shards = {}

        def get(key):
            name = index[key]
            if name not in shards:
                shards[name] = stack.enter_context(
                    safe_open(model / name, framework="pt", device="cpu")
                )
            return shards[name].get_tensor(key)

        for expert in experts:
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            gate, up, down = (
                prefix + part + "." for part in ("gate_proj", "up_proj", "down_proj")
            )
            result["w13"].append(
                torch.cat(
                    [
                        get(gate + "weight_packed")[:512],
                        get(up + "weight_packed")[:512],
                    ]
                )
            )
            result["s13"].append(
                torch.cat(
                    [
                        get(gate + "weight_scale")[:512],
                        get(up + "weight_scale")[:512],
                    ]
                )
            )
            gg, gu = (get(p + "weight_global_scale") for p in (gate, up))
            if not torch.equal(gg, gu):
                raise ValueError("gate/up global scales differ")
            result["g13"].append((1 / gg.float()).reshape(()))
            result["w2"].append(get(down + "weight_packed")[:, :256].contiguous())
            result["s2"].append(get(down + "weight_scale")[:, :32].contiguous())
            result["g2"].append(
                (1 / get(down + "weight_global_scale").float()).reshape(())
            )
    return {key: torch.stack(value) for key, value in result.items()}


def repack(packed):
    """Byte-neutral layout from the archived prototype, no quant conversion."""
    e, n, kh = packed.shape
    codes = (
        torch.stack([packed & 15, packed >> 4], -1).reshape(e, n, kh // 16, 32).int()
    )
    words = torch.zeros(e, n, kh // 16, 4, dtype=torch.int32, device=packed.device)
    for q in range(4):
        for pos, k in (
            (3, 0),
            (7, 1),
            (2, 8),
            (6, 9),
            (1, 16),
            (5, 17),
            (0, 24),
            (4, 25),
        ):
            words[..., q] |= codes[..., k + 2 * q] << (4 * pos)
    return words.view(torch.uint8).reshape(e, n, kh)


def build(path):
    from torch.utils.cpp_extension import load

    path.mkdir(parents=True, exist_ok=True)
    return load(
        name="glm53_nvfp4_pipeline",
        sources=[str(Path(__file__).with_name("nvfp4_decode_pipeline.cu").resolve())],
        extra_include_paths=[str(Path(__file__).resolve().parents[2] / "csrc")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "-gencode=arch=compute_120f,code=sm_120f",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        build_directory=str(path),
        verbose=True,
    )


def capture(calls):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if not args.build_only and (not args.model or not args.journal or not args.output):
        parser.error("model, journal and output required")
    if args.output and args.output.exists():
        parser.error("output must be new")
    ext = build(args.build_dir)
    if args.build_only:
        return
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs busy: {active}")
    torch.set_num_threads(4)
    torch.manual_seed(105341)
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_nvfp4_moe_layer_for_marlin,
    )
    from vllm.scalar_type import scalar_types

    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = {
        "status": "running",
        "roadmap": "4.1",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "method": __doc__,
        "fixtures": [],
        "timings": {},
        "journal_sha256": hashlib.sha256(args.journal.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        if torch.cuda.get_device_capability() != (12, 0):
            raise ValueError("SM120 required")
        result["gpu"] = torch.cuda.get_device_name()
        root = Path(__file__).resolve().parents[2]
        result["source_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/kernels/benchmark_glm53_nvfp4_pipeline.py",
                "benchmarks/kernels/nvfp4_decode_pipeline.cu",
                "benchmarks/kernels/nvfp4_decode_sm120.cuh",
            )
        }
        calls = {
            phase: {kind: [] for kind in ("drain", "pipeline", "marlin")}
            for phase in ("gate_up", "down")
        }
        keepalive = []
        for layer in (3, 23, 44):
            for _, record in route_cases(args.journal, [1], 15):
                experts = record["routes"][0][layer - 3]
                if len(experts) != 8 or len(set(experts)) != 8:
                    raise ValueError("expected eight distinct captured experts")
                raw = load_selected(args.model, index, layer, experts)
                device = {key: value.cuda() for key, value in raw.items()}
                layer_stub = SimpleNamespace(
                    num_experts=8,
                    hidden_size=4096,
                    intermediate_size_per_partition=512,
                    params_dtype=torch.bfloat16,
                )
                marlin = prepare_nvfp4_moe_layer_for_marlin(
                    layer_stub,
                    *[device[k] for k in ("w13", "s13", "g13", "w2", "s2", "g2")],
                    True,
                )
                ids = torch.arange(8, dtype=torch.int32, device="cuda")
                sorted_ids, expert_ids, padded = moe_align_block_size(ids[None], 8, 8)
                topk_weights = torch.full((1, 8), 2.5 / 8, device="cuda")
                counts = torch.ones(8, dtype=torch.int32, device="cuda")
                x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
                gate_reference = None
                for phase, prefix, mi in (("gate_up", "13", 0), ("down", "2", 3)):
                    if phase == "down":
                        x = torch.empty(8, 512, device="cuda", dtype=torch.bfloat16)
                        apply_moe_activation(
                            MoEActivation.SILU, x, gate_reference, clamp_limit=10.0
                        )
                    n, k = device["w" + prefix].shape[1], x.shape[1]
                    rows = torch.full((8, 16), -1, dtype=torch.int32, device="cuda")
                    rows[:, 0] = 0 if phase == "gate_up" else ids
                    w = repack(device["w" + prefix])
                    outputs = {
                        kind: torch.empty(8, n, device="cuda", dtype=torch.bfloat16)
                        for kind in calls[phase]
                    }
                    native_args = (
                        x,
                        w,
                        device["s" + prefix],
                        device["g" + prefix],
                        ids,
                        rows,
                        counts,
                        ids,
                    )
                    fns = {
                        kind: partial(
                            ext.run, *native_args, outputs[kind], kind == "pipeline"
                        )
                        for kind in ("drain", "pipeline")
                    }
                    fns["marlin"] = partial(
                        ops.moe_wna16_marlin_gemm,
                        x,
                        outputs["marlin"],
                        marlin[mi],
                        None,
                        marlin[mi + 1],
                        None,
                        marlin[mi + 2],
                        None,
                        None,
                        None,
                        layer_stub.workspace,
                        sorted_ids,
                        expert_ids,
                        padded,
                        topk_weights,
                        moe_block_size=8,
                        top_k=8 if phase == "gate_up" else 1,
                        mul_topk_weights=False,
                        b_q_type=scalar_types.float4_e2m1f,
                        size_m=x.shape[0],
                        size_n=n,
                        size_k=k,
                        is_k_full=True,
                        use_atomic_add=False,
                        use_fp32_reduce=True,
                        is_zp_float=False,
                    )
                    for fn in fns.values():
                        fn()
                    torch.testing.assert_close(
                        outputs["pipeline"], outputs["drain"], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        outputs["pipeline"], outputs["marlin"], rtol=0.01, atol=0.01
                    )
                    # Independent actual-byte FP64 oracle on fixed strided output rows.
                    sample = torch.arange(0, n, n // 32)
                    expected = []
                    cpu_x = x.cpu()
                    for slot in range(8):
                        weight = decode_fragments(
                            raw["w" + prefix][slot, sample],
                            raw["s" + prefix][slot]
                            .view(torch.uint8)[sample]
                            .view(torch.float8_e4m3fn),
                        )
                        value = (
                            weight @ cpu_x[0 if phase == "gate_up" else slot].double()
                        )
                        expected.append(value * raw["g" + prefix][slot].double())
                    expected = torch.stack(expected).bfloat16()
                    actual = outputs["pipeline"].cpu()[:, sample]
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
                    result["fixtures"].append(
                        {
                            "layer": layer,
                            "step": record["step"],
                            "experts": experts,
                            "phase": phase,
                            "drain_bit_exact": True,
                            "versus_marlin": errors(
                                outputs["pipeline"], outputs["marlin"]
                            ),
                            "sampled_oracle": errors(actual, expected),
                            "weight_sha256": hashlib.sha256(
                                raw["w" + prefix].numpy().tobytes()
                            ).hexdigest(),
                        }
                    )
                    print(
                        f"checked layer={layer} step={record['step']} phase={phase}",
                        flush=True,
                    )
                    for kind, fn in fns.items():
                        calls[phase][kind].append(fn)
                    keepalive.append((native_args, outputs, marlin, layer_stub, fns))
                    if phase == "gate_up":
                        gate_reference = outputs["marlin"]
                save()
                del raw, device
                gc.collect()
        # A/B/A plus the retained production comparator: fixed counts, all retained.
        for phase, implementations in calls.items():
            graphs = {kind: capture(fns) for kind, fns in implementations.items()}
            samples = []
            for _ in range(3):
                samples.append(
                    {
                        kind: measure(graphs[kind], len(implementations[kind]), 10)
                        for kind in ("marlin", "drain", "pipeline")
                    }
                )
                samples[-1]["return_drain"] = measure(
                    graphs["drain"], len(implementations["drain"]), 10
                )
                samples[-1]["return_marlin"] = measure(
                    graphs["marlin"], len(implementations["marlin"]), 10
                )
            result["timings"][phase] = {
                "samples_us": samples,
                "median_us": {
                    kind: statistics.median(row[kind] for row in samples)
                    for kind in samples[0]
                },
            }
            print(phase, result["timings"][phase], flush=True)
            save()
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
