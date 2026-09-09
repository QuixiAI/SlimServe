#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolate Marlin shared-memory phase boundaries without replacing serving.

Builds two actual NVFP4/BF16 M8 templates from this repository's header, with
line information. The optional barriers alter only the isolated source copy.
Use racecheck --kernel-name kns=marlin_shared_probe to skip checkpoint repacks
and the separately checked installed-library reference calls.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import prepare_case
from benchmarks.kernels.profile_glm53_marlin import load_layer, route_cases

DECL = """void run(at::Tensor a, at::Tensor out, at::Tensor weight,
at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
at::Tensor sorted_ids, at::Tensor experts, at::Tensor padded,
at::Tensor routing_weights, int top_k, bool weighted, int thread_k,
int thread_n, int blocks_per_sm);"""

WRAPPER = r"""
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

void run(at::Tensor a, at::Tensor out, at::Tensor weight,
         at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
         at::Tensor sorted_ids, at::Tensor experts, at::Tensor padded,
         at::Tensor routing_weights, int top_k, bool weighted, int thread_k,
         int thread_n, int blocks_per_sm) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16 && a.is_contiguous());
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 &&
              out.is_contiguous());
  TORCH_CHECK(weight.is_contiguous() && scales.is_contiguous());
  TORCH_CHECK(scales.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(global_scale.scalar_type() == at::kFloat &&
              routing_weights.scalar_type() == at::kFloat);
  TORCH_CHECK((thread_k == 128 && thread_n == 64) ||
              (thread_k == 64 && thread_n == 128));
  TORCH_CHECK(blocks_per_sm >= 1 && blocks_per_sm <= 3);
  const c10::cuda::CUDAGuard guard(a.device());
  int m = a.size(0), k = a.size(1), n = out.size(1);
  TORCH_CHECK(out.size(0) == m * top_k && k % thread_k == 0 && n % thread_n == 0);
  TORCH_CHECK(weight.size(0) == 288 && weight.size(1) == k / 16);
  TORCH_CHECK(scales.size(1) == k / 16 && scales.size(2) == n);
  int sms, max_shared;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, a.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, a.get_device()));
  TORCH_CHECK(workspace.numel() >= sms * 4);
  if (blocks_per_sm > 1) max_shared = max_shared / blocks_per_sm - 1024;
  long temp_size = std::min((long)n * sorted_ids.numel(), (long)sms * 4 * 8 * 256) * 2;
  auto temp = at::empty({temp_size}, a.options().dtype(at::kFloat));
  auto kernel = MARLIN_NAMESPACE_NAME::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 128, 1, 4, 8, true, 4, 1, false>;
  if (thread_n == 128) kernel = MARLIN_NAMESPACE_NAME::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), 128, 1, 8, 4, true, 4, 1, false>;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared));
  auto stream = at::cuda::getCurrentCUDAStream();
  kernel<<<sms * blocks_per_sm, 128, max_shared, stream>>>(
      (const int4*)a.data_ptr(), (const int4*)weight.data_ptr(),
      (int4*)out.data_ptr(), (int4*)temp.data_ptr(), nullptr, nullptr,
      (const int4*)scales.data_ptr(), global_scale.data_ptr<float>(), nullptr,
      nullptr, sorted_ids.data_ptr<int>(), experts.data_ptr<int>(),
      padded.data_ptr<int>(),
      routing_weights.data_ptr<float>(), top_k, weighted, k / 16, m, n, k,
      workspace.data_ptr<int>(), false, false, true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def isolated_source(original, boundary):
    if boundary not in ("original", "compute", "output", "both"):
        raise ValueError("unknown boundary")
    compute = "  auto thread_block_reduce = [&]() {"
    output = "  auto write_result = [&](bool last) {"
    if original.count(compute) != 1 or original.count(output) != 1:
        raise ValueError("Marlin header no longer matches expected boundaries")
    if boundary in ("compute", "both"):
        original = original.replace(compute, compute + "\n    __syncthreads();")
    if boundary in ("output", "both"):
        original = original.replace(output, output + "\n    __syncthreads();")
    return original


def build(args):
    from torch.utils.cpp_extension import load_inline

    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    version = subprocess.check_output(
        [str(cuda_home / "bin/nvcc"), "--version"], text=True
    )
    match = re.search(r"release (\d+\.\d+)", version)
    if not match or match.group(1) != torch.version.cuda:
        raise ValueError("probe toolkit must match the installed Torch toolkit")
    root = Path(__file__).resolve().parents[2]
    header_path = root / "csrc/libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"
    original = header_path.read_text()
    namespace = "marlin_shared_probe_" + args.boundary
    source = (
        f"#define MARLIN_NAMESPACE_NAME {namespace}\n"
        + isolated_source(original, args.boundary)
        + WRAPPER
    )
    build_dir = args.build_dir / args.boundary
    build_dir.mkdir(parents=True, exist_ok=True)
    extension = load_inline(
        name=namespace,
        cpp_sources=DECL,
        cuda_sources=source,
        functions=["run"],
        build_directory=str(build_dir),
        extra_include_paths=[str(root / "csrc")],
        extra_cflags=["-O3", "-std=c++20", "-fvisibility=hidden"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-static-global-template-stub=false",
            "-gencode=arch=compute_120f,code=sm_120f",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-Xcompiler=-fvisibility=hidden",
        ],
        with_cuda=True,
        verbose=True,
    )
    return extension, hashlib.sha256(original.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary",
        choices=["original", "compute", "output", "both"],
        default="original",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    if args.replays < 1 or any(not 1 <= batch <= 16 for batch in args.batch):
        parser.error("positive replays and batches 1..16 required")
    if not args.build_only and (
        args.model is None
        or args.journal is None
        or args.output is None
        or args.output.exists()
    ):
        parser.error("run requires model, journal and new output")
    if not args.build_only:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPUs already have compute processes: {active}")
    extension, header_hash = build(args)
    if args.build_only:
        return

    def gemm(*pos, **kw):
        # Only the two observed M8 schedules, no auto heuristic in this wrapper.
        tk, tn, b = kw["thread_k"], kw["thread_n"], kw["blocks_per_sm"]
        if tk == -1:
            tk, tn, b = (128, 64, 2) if kw["top_k"] == 8 else (64, 128, 3)
        extension.run(
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
            tk,
            tn,
            b,
        )
        return pos[1]

    result = {
        "status": "running",
        "boundary": args.boundary,
        "header_sha256": header_hash,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "cases": [],
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        torch.manual_seed(127)
        weights, _ = load_layer(args.model, 3, 0)
        for batch, record in route_cases(args.journal, args.batch, 1):
            reference = prepare_case(weights, batch, record, 3)
            probe = prepare_case(weights, batch, record, 3, gemm=gemm)
            for config in (((128, 64, 2), (64, 128, 3)), ((128, 64, 1), (64, 128, 2))):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    probe["call"]("moe", config)
                for replay in range(args.replays):
                    reference["input"].normal_()
                    probe["input"].copy_(reference["input"])
                    expected = reference["call"]("moe", config).clone()
                    graph.replay()
                    got = probe["outputs"]["moe"]
                    torch.cuda.synchronize()
                    row = {
                        "batch": batch,
                        "config": config,
                        "replay": replay,
                        "exact": torch.equal(got, expected),
                        "max_abs": (got.float() - expected.float()).abs().max().item(),
                    }
                    result["cases"].append(row)
                    save()
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)
                    print(json.dumps(row), flush=True)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        save()
        raise
    save()


if __name__ == "__main__":
    main()
