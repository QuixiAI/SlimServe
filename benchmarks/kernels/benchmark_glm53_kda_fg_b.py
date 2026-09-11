#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.2 paired K128 projection, one fixed kernel versus serving strided bmm.

Actual rank0 TP4 BF16 weights, original merged-input strides, synthetic inputs.
Three checkpoint layers with 129 distinct copies each exceed three128MiB L2
capacities. Three A/B/A rounds, ten replays; no geometry/configuration sweep.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.cpp_extension import load

from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from benchmarks.kernels.benchmark_glm53_nvfp4_pipeline import capture
from benchmarks.kernels.check_glm53_marlin_schedule import errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if not args.build_only and (
        not args.model or not args.output or args.output.exists()
    ):
        parser.error("model and NEW output required")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    ext = load(
        name="glm53_kda_fg_b",
        sources=[str(root / "benchmarks/kernels/kda_fg_b_decode.cu")],
        extra_include_paths=[str(root / "csrc")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "-gencode=arch=compute_120f,code=sm_120f",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        build_directory=str(args.build_dir),
        verbose=True,
    )
    if args.build_only:
        return
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPUs busy: {active}")
    if torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("SM120 required")
    torch.set_num_threads(4)
    torch.manual_seed(1142)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = []
    for layer in (0, 22, 44):
        pair = []
        for projection in ("f_b_proj", "g_b_proj"):
            key = f"model.language_model.layers.{layer}.self_attn.{projection}.weight"
            with safe_open(
                args.model / index[key], framework="pt", device="cpu"
            ) as source:
                pair.append(source.get_tensor(key)[:2048].contiguous())
        weight = torch.stack(pair)
        if weight.shape != (2, 2048, 128) or weight.dtype != torch.bfloat16:
            raise ValueError("unexpected fixed-recipe projection weight")
        weights.append(weight)
    result = {
        "status": "running",
        "roadmap": "4.2",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "method": __doc__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "weight_sha256": [
            hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest()
            for w in weights
        ],
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "benchmarks/kernels/benchmark_glm53_kda_fg_b.py",
                "benchmarks/kernels/kda_fg_b_decode.cu",
            "benchmarks/kernels/kda_fg_b_decode_sm120.cuh",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        banks = [weight.cuda() for _ in range(129) for weight in weights]
        if len({w.data_ptr() for w in banks}) != 387:
            raise RuntimeError("weight rotation must not alias")
        for batch in (1, 8, 16):
            base = torch.randn(batch, 6528, device="cuda", dtype=torch.bfloat16)
            x = base[:, -256:].view(batch, 2, 128).transpose(0, 1)
            row = {"batch": batch, "x_stride": list(x.stride()), "correctness": []}
            for i, w in enumerate(banks[:3]):
                a = torch.bmm(x, w.transpose(1, 2))
                b = torch.empty_like(a)
                ext.run(x, w, b)
                expected = torch.bmm(
                    x.cpu().double(), weights[i].double().transpose(1, 2)
                ).bfloat16()
                torch.testing.assert_close(b.cpu(), expected, rtol=0.01, atol=0.001)
                torch.testing.assert_close(b, a, rtol=0.01, atol=0.001)
                row["correctness"].append(
                    {
                        "versus_bmm": errors(b, a),
                        "versus_fp64": errors(b.cpu(), expected),
                    }
                )
            a_out = [
                torch.empty(2, batch, 2048, device="cuda", dtype=torch.bfloat16)
                for _ in banks
            ]
            b_out = [torch.empty_like(out) for out in a_out]
            a = capture(
                [
                    partial(torch.bmm, x, w.transpose(1, 2), out=out)
                    for w, out in zip(banks, a_out)
                ]
            )
            b = capture([partial(ext.run, x, w, out) for w, out in zip(banks, b_out)])
            # Changed input also exercises the captured strided read, not a baked value.
            x.mul_(0.5)
            a.replay()
            b.replay()
            for av, bv in zip(a_out, b_out):
                torch.testing.assert_close(bv, av, rtol=0.01, atol=0.001)
            samples = []
            for _ in range(3):
                samples.append(
                    {
                        "before_us": measure(a, len(banks), 10),
                        "candidate_us": measure(b, len(banks), 10),
                        "after_us": measure(a, len(banks), 10),
                    }
                )
            row["samples"] = samples
            row["median_us"] = {
                k: statistics.median(s[k] for s in samples) for k in samples[0]
            }
            result["cases"].append(row)
            print(row, flush=True)
            save()
            del a, b, a_out, b_out
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
