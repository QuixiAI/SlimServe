#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Large-prefill Marlin tile A/B/A, actual NVFP4 weights, synthetic inputs/routes.

Reuses the decode probe's preparation, correctness and timing. This is a kernel
screen, not serving performance. The full expert weight set exceeds SM120 L2.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import (
    capture,
    correctness,
    measure,
    parse_config,
    prepare_case,
)
from benchmarks.kernels.check_glm53_marlin_schedule import errors, oracle
from benchmarks.kernels.profile_glm53_marlin import load_layer


def parse_prefill_config(value):
    if value == "64,512,1":
        return 64, 512, 1
    return parse_config(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[3])
    parser.add_argument("--batch", type=int, nargs="+", default=[7616])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--stages", type=int, choices=[2, 3, 4])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--installed-wide", action="store_true")
    parser.add_argument(
        "--configs",
        type=parse_prefill_config,
        nargs="+",
        default=[
            (64, 256, 1),
            (64, 128, 1),
            (64, 128, 2),
            (128, 64, 1),
            (128, 128, 1),
        ],
    )
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.rank < 4:
        parser.error("new output path and rank 0..3 required")
    if min(args.rounds, args.replays, *args.batch) < 1 or max(args.batch) > 7616:
        parser.error("positive counts and batches <=7616 required")
    if any(not 3 <= layer <= 44 for layer in args.layers):
        parser.error("target layers 3..44 required")
    if any(b not in (8, 16, 32, 48, 64) for b in args.block_sizes):
        parser.error("block sizes must be in 8/16/32/48/64")
    if args.oracle and not args.pipeline:
        parser.error("--oracle requires --pipeline")
    if args.installed_wide and (
        os.environ.get("VLLM_GLM53_MARLIN_PREFILL_WIDE") != "1"
        or args.stages != 3
        or not args.pipeline
        or args.configs != [(64, 512, 1)]
    ):
        parser.error("installed-wide requires flag1, stages3, pipeline and K64/N512/1")
    extension = None
    if args.stages is not None:
        if args.build_dir is None or args.block_sizes != [64]:
            parser.error("stage experiment requires --build-dir and --block-sizes 64")
        from torch.utils.cpp_extension import load

        args.build_dir.mkdir(parents=True, exist_ok=True)
        extension = load(
            name="marlin_prefill_stages",
            sources=[
                str(Path(__file__).with_name("marlin_prefill_stages.cu").resolve())
            ],
            extra_include_paths=[str(Path(__file__).resolve().parents[2] / "csrc")],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-static-global-template-stub=false",
                "-gencode=arch=compute_120f,code=sm_120f",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            build_directory=str(args.build_dir),
            verbose=True,
        )
        if args.build_only:
            return
    elif args.build_only:
        parser.error("--build-only requires --stages")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPUs busy: {active}")
    torch.manual_seed(103)
    from vllm import _custom_ops as ops

    def gemm(*pos, **kw):
        if args.installed_wide:
            kw.update(thread_k=-1, thread_n=-1, blocks_per_sm=-1)
            return ops.moe_wna16_marlin_gemm(*pos, **kw)
        if extension is None or kw["thread_k"] == -1:
            return ops.moe_wna16_marlin_gemm(*pos, **kw)
        if kw["thread_k"] != 64 or kw["moe_block_size"] != 64:
            raise ValueError("pipeline probe requires K64/M64 tiles")
        return extension.run(
            pos[0],
            pos[1],
            pos[2],
            pos[4],
            pos[6],
            pos[10],
            pos[11],
            pos[12],
            pos[13],
            pos[14],
            kw["top_k"],
            kw["mul_topk_weights"],
            kw["thread_n"],
            kw["blocks_per_sm"],
            args.stages,
        )

    def baseline_gemm(*pos, **kw):
        if args.installed_wide and kw["moe_block_size"] == 64:
            # Explicit schedules bypass the new dispatcher; this is the measured
            # installed auto baseline for all tested M64 target shapes.
            kw.update(thread_k=64, thread_n=256, blocks_per_sm=1)
        return ops.moe_wna16_marlin_gemm(*pos, **kw)

    result = {
        "status": "running",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "native_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[2]
                / "vllm/_moe_C_stable_libtorch.abi3.so"
            ).read_bytes()
        ).hexdigest(),
        "probe_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest()
        if extension is not None
        else None,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "layers": {},
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for layer in args.layers:
            weights, identity = load_layer(args.model, layer, args.rank)
            result["layers"][layer] = identity
            print(f"Loaded actual layer {layer} weights", flush=True)
            for batch in args.batch:
                block_size = next(
                    (b for b in [8, 16, 32, 48, 64] if batch * 8 / 288 / b < 0.9), 64
                )
                for routing in ("uniform", "skewed"):
                    generator = torch.Generator().manual_seed(7103 + batch)
                    logits = torch.randn(batch, 288, generator=generator)
                    if routing == "skewed":
                        logits[:, :32] += 2
                    ids = logits.topk(8, dim=-1).indices.tolist()
                    record = {"routes": [[row] * 42 for row in ids], "step": 0}
                    reference = prepare_case(
                        weights,
                        batch,
                        record,
                        layer,
                        block_size=block_size,
                        gemm=baseline_gemm,
                    )
                    for phase in ("moe",) if args.pipeline else ("gate_up", "down"):
                        auto = ((-1, -1, -1),) * 2 if args.pipeline else (-1, -1, -1)
                        baseline = (
                            None
                            if args.check_only
                            else capture([reference], phase, auto)
                        )
                        for candidate_block, config in (
                            (b, c) for b in args.block_sizes for c in args.configs
                        ):
                            if args.pipeline:
                                config = (config, config)
                            case = prepare_case(
                                weights,
                                batch,
                                record,
                                layer,
                                block_size=candidate_block,
                                gemm=gemm,
                                alignment=reference["alignment"]
                                if candidate_block == block_size
                                else None,
                            )
                            case["input"].copy_(reference["input"])
                            case["outputs"]["activation"].copy_(
                                reference["outputs"]["activation"]
                            )
                            case["refs"] = reference["refs"]
                            row = {
                                "layer": layer,
                                "batch": batch,
                                "routing": routing,
                                "baseline_block_size": block_size,
                                "block_size": candidate_block,
                                "phase": phase,
                                "config": config,
                                "samples": [],
                            }
                            result["measurements"].append(row)
                            try:
                                row["correctness"] = correctness([case], phase, config)
                                if args.installed_wide:
                                    w13, s13, g13, w2, s2, g2, workspace = weights
                                    for name, a, w, s, g, topk in (
                                        ("gate_up", case["input"], w13, s13, g13, 8),
                                        (
                                            "down",
                                            case["outputs"]["activation"],
                                            w2,
                                            s2,
                                            g2,
                                            1,
                                        ),
                                    ):
                                        actual = case["outputs"][name]
                                        expected = extension.run(
                                            a,
                                            torch.empty_like(actual),
                                            w,
                                            s,
                                            g,
                                            workspace,
                                            *case["alignment"],
                                            case["routing_weights"],
                                            topk,
                                            topk == 1,
                                            512,
                                            1,
                                            3,
                                        )
                                        if not torch.equal(actual, expected):
                                            raise AssertionError(
                                                f"native/probe mismatch: {name}"
                                            )
                                    row["native_probe_exact"] = True
                                if args.oracle:
                                    selected = [0, batch // 2, batch - 1]
                                    expected = oracle(
                                        args.model,
                                        layer,
                                        args.rank,
                                        [ids[i] for i in selected],
                                        case["input"][selected].cpu(),
                                        case["routing_weights"][selected].cpu(),
                                    )
                                    row["oracle"] = {}
                                    for name in ("gate_up", "activation", "moe"):
                                        got = (
                                            case["outputs"][name]
                                            .view(batch, 8, -1)[selected]
                                            .flatten(0, 1)
                                            .cpu()
                                        )
                                        row["oracle"][name] = errors(
                                            got, expected[name]
                                        )
                                        torch.testing.assert_close(
                                            got, expected[name], rtol=0.01, atol=0.01
                                        )
                                if args.check_only:
                                    row["status"] = "checked"
                                    save()
                                    print(json.dumps(row), flush=True)
                                    del case
                                    continue
                                candidate = capture([case], phase, config)
                            except RuntimeError as error:
                                if not any(
                                    s in str(error)
                                    for s in (
                                        "Invalid thread config",
                                        "Unsupported shapes",
                                    )
                                ):
                                    raise
                                row.update(status="unsupported", error=str(error))
                                save()
                                del case
                                continue
                            for _ in range(args.rounds):
                                row["samples"].append(
                                    {
                                        name: measure(graph, 1, args.replays)
                                        for name, graph in (
                                            ("a_before_us", baseline),
                                            ("b_us", candidate),
                                            ("a_after_us", baseline),
                                        )
                                    }
                                )
                            row["median_us"] = {
                                k: statistics.median(s[k] for s in row["samples"])
                                for k in row["samples"][0]
                            }
                            row["paired_speedup"] = statistics.median(
                                (s["a_before_us"] + s["a_after_us"]) / (2 * s["b_us"])
                                for s in row["samples"]
                            )
                            row["status"] = "complete"
                            print(
                                json.dumps(
                                    {
                                        k: v
                                        for k, v in row.items()
                                        if k not in ("correctness", "samples")
                                    }
                                ),
                                flush=True,
                            )
                            save()
                            del candidate
                            del case
                        del baseline
                    del reference
                    gc.collect()
                    torch.cuda.empty_cache()
            del weights
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
