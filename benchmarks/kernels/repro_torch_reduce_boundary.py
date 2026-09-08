#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate the installed ATen global-reduction shared-memory boundary.

Builds an isolated FP64 row-sum extension from the installed Reduce.cuh.
--barrier changes exactly one boundary between block-y and block-x reduction.
Neither variant replaces an installed Torch kernel or changes serving code.
Run each variant under compute-sanitizer --tool racecheck to test the hypothesis.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

BOUNDARY = """      value = block_y_reduce<output_vec_size>(value, shared_memory);
      if (config.should_block_x_reduce()) {"""
WITH_BARRIER = """      value = block_y_reduce<output_vec_size>(value, shared_memory);
      __syncthreads();
      if (config.should_block_x_reduce()) {"""

WRAPPER = r"""
#include <c10/cuda/CUDAGuard.h>
struct SumDouble {
    __host__ __device__ double operator()(double a, double b) const { return a + b; }
};
at::Tensor row_sum(at::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                x.scalar_type() == at::kDouble, "expected contiguous CUDA FP64 [B,V]");
    const c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty({x.size(0), 1}, x.options());
    auto iter = at::TensorIterator::reduce_op(out, x);
    at::native::gpu_reduce_kernel<double, double>(
        iter, at::native::func_wrapper<double>(SumDouble{}), 0.0);
    return out.squeeze(1);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 3, 8, 16])
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if min(*args.batch, args.vocab, args.repeats) < 1:
        parser.error("shapes and repeats must be positive")
    if not args.build_only:
        if args.output is None or args.output.exists():
            parser.error("run requires a new --output path")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPUs already have compute processes: {active}")

    from torch.utils.cpp_extension import load_inline

    header_path = Path(torch.__file__).parent / "include/ATen/native/cuda/Reduce.cuh"
    original = header_path.read_text()
    if original.count(BOUNDARY) != 1:
        raise RuntimeError("installed Reduce.cuh does not have the expected boundary")
    header = original.replace(BOUNDARY, WITH_BARRIER) if args.barrier else original
    variant = "barrier" if args.barrier else "original"
    build_dir = args.build_dir / variant
    build_dir.mkdir(parents=True, exist_ok=True)
    extension = load_inline(
        name=f"torch_reduce_boundary_{variant}",
        cpp_sources="at::Tensor row_sum(at::Tensor x);",
        cuda_sources=header + WRAPPER,
        functions=["row_sum"],
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-fvisibility=hidden"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--expt-extended-lambda",
            "-Xcompiler=-fvisibility=hidden",
        ],
        with_cuda=True,
        verbose=True,
    )
    if args.build_only:
        print(f"built isolated {variant} probe; no GPU workload run", flush=True)
        return
    result = {
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "header_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "variant": variant,
        "vocab": args.vocab,
        "repeats": args.repeats,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    for batch in args.batch:
        # Distinct row values expose accidental cross-row mixing. Expected
        # sums are exactly representable integers, calculated on the CPU.
        row_values = torch.arange(1, batch + 1, dtype=torch.float64)
        x = row_values[:, None].expand(batch, args.vocab).contiguous().cuda()
        expected = (row_values * args.vocab).tolist()
        mismatches = 0
        for _ in range(args.repeats):
            got = extension.row_sum(x).cpu().tolist()
            mismatches += got != expected
        row = {
            "batch": batch,
            "expected": expected,
            "last": got,
            "mismatches": mismatches,
        }
        result["rows"].append(row)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if any(row["mismatches"] for row in result["rows"]):
        raise SystemExit("numerical mismatches observed; see output")


if __name__ == "__main__":
    main()
