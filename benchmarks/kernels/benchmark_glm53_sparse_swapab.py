#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One source-led swapAB candidate on fixed SM120 prefill shapes. No sweep."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from benchmarks.kernels.benchmark_glm53_sparse_prefill_tiles import capture
from benchmarks.kernels.glm53_sparse_swapab import sparse_swapab
from vllm.v1.attention.backends.mla import quixicore_mla_sparse_prefill as pf


def build_native(path):
    from torch.utils.cpp_extension import load

    path.mkdir(parents=True, exist_ok=True)
    return load(
        name="glm53_sparse_swapab",
        sources=[str(Path(__file__).with_name("glm53_sparse_swapab.cu").resolve())],
        extra_include_paths=[str(Path(__file__).resolve().parents[2] / "csrc")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120f,code=sm_120f",
            "-Xptxas=-v",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        build_directory=str(path),
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    implementation = parser.add_mutually_exclusive_group()
    implementation.add_argument("--build-dir", type=Path)
    implementation.add_argument("--installed", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="one candidate launch; no timing or correctness claim",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.build_only and args.build_dir is None:
        parser.error("--build-only requires --build-dir")
    native = build_native(args.build_dir) if args.build_dir else None
    if args.installed:
        from vllm.quixicore.ops import _qc

        native = _qc()
    if args.build_only:
        if native is None:
            parser.error("--build-only requires --build-dir")
        return
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        parser.error(f"GPUs busy: {active}")
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(4311)
    result = {
        "status": "running",
        "roadmap": "4.3/4.5",
        "diagnostic_only": True,
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            Path(__file__).with_name("glm53_sparse_swapab.py").read_bytes()
        ).hexdigest(),
        "cases": [],
        "native": native is not None,
    }
    if native is not None:
        result["native_binary_sha256"] = hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest()
        result["native_source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).with_name("glm53_sparse_swapab.cu"),
                Path(__file__).resolve().parents[2]
                / "csrc/quixicore/serving/glm53_sparse_swapab.cuh",
                Path(__file__).resolve().parents[2]
                / "csrc/quixicore/tm_cuda/tm_cuda_glm53_sparse.cu",
            )
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for batch, context in (
            (2048, 32768),
            (7616, 32768),
            (2048, 131072),
            (7616, 131072),
        ):
            q = (torch.randn(batch, 32, 512, device="cuda") * 0.2).bfloat16()
            kv = (torch.randn(context // 64, 64, 512, device="cuda") * 0.5).bfloat16()
            bt = torch.arange(context // 64, device="cuda", dtype=torch.int32)[
                None
            ].repeat(batch, 1)
            stride = torch.randint(context // 8, (batch, 1), device="cuda") * 2 + 1
            offset = torch.randint(context // 4, (batch, 1), device="cuda")
            group = (torch.arange(512, device="cuda")[None] * stride + offset) % (
                context // 4
            )
            indices = (
                (group[:, :, None] * 4 + torch.arange(4, device="cuda"))
                .reshape(batch, 2048)
                .int()
            )
            lengths = torch.full((batch,), 2048, device="cuda", dtype=torch.int32)
            outs = [torch.empty_like(q) for _ in range(2)]

            def launch(
                arm,
                batch=batch,
                q=q,
                kv=kv,
                bt=bt,
                indices=indices,
                lengths=lengths,
                outs=outs,
            ):
                if arm == 1 and native is not None:
                    if args.installed:
                        outs[arm] = native.mla_prefill_bf16_sparse_nope_sm120(
                            q,
                            kv,
                            bt,
                            indices,
                            lengths,
                            64,
                            1 / math.sqrt(512),
                            kv.stride(0) * 2,
                        )
                        return None
                    native.run(
                        q, kv, bt, indices, lengths, outs[arm], 64, 1 / math.sqrt(512)
                    )
                    return None
                implementation = (pf._sparse_mla_prefill_kernel, sparse_swapab)[arm]
                return implementation[(batch,)](
                    q,
                    kv,
                    bt,
                    indices,
                    lengths,
                    outs[arm],
                    indices.shape[1],
                    bt.shape[1],
                    kv.stride(0),
                    64,
                    1 / math.sqrt(512),
                    H=32,
                    D=512,
                    BLOCK_N=32,
                    num_warps=4,
                    num_stages=2,
                )

            if args.profile_only:
                launch(1)
                torch.cuda.synchronize()
                result.update(status="profile_only", batch=batch, context=context)
                return
            kernels = [launch(arm) for arm in range(2)]
            torch.testing.assert_close(outs[1], outs[0], rtol=0.01, atol=0.016)
            # Small independent FP64 attention oracle, actual gathered BF16
            # values, not another implementation sharing the candidate layout.
            for row in (0, batch // 2, batch - 1):
                selected = kv.view(context, 512)[indices[row].long()].double()
                scores = q[row].double() @ selected.T / math.sqrt(512)
                expected = scores.softmax(-1) @ selected
                torch.testing.assert_close(
                    outs[1][row].double(), expected, rtol=0.01, atol=0.016
                )
            max_abs = (outs[0].float() - outs[1].float()).abs().max().item()
            graphs = [capture(lambda arm=arm: launch(arm)) for arm in range(2)]
            # Mixed empty, short and changed selections in the same captured
            # shape. Restore full useful rows before timing.
            lengths[0] = 0
            lengths[1] = 37
            indices[2, :64] = -1
            q.mul_(2)
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outs[1], outs[0], rtol=0.01, atol=0.016)
            assert torch.count_nonzero(outs[1][0]).item() == 0
            lengths.fill_(2048)
            indices[2, :64] = indices[3, :64]
            samples = []
            for _ in range(3):
                samples.append(
                    {
                        name: measure(graphs[arm], 1, 5)
                        for name, arm in (("before", 0), ("candidate", 1), ("after", 0))
                    }
                )
            row = {
                "batch": batch,
                "context": context,
                "max_abs": max_abs,
                "oracle_rows": 3,
                "changed_input_graph_parity": True,
                "samples_us": samples,
                "median_us": {
                    k: statistics.median(s[k] for s in samples) for k in samples[0]
                },
                "resources": [
                    {
                        "registers": k.n_regs,
                        "spills": k.n_spills,
                        "shared": k.metadata.shared,
                    }
                    if k is not None
                    else {"installed": True}
                    if args.installed
                    else dict(native.info())
                    for k in kernels
                ],
            }
            result["cases"].append(row)
            print(row, flush=True)
            save()
            del graphs, outs, q, kv, bt, indices, lengths
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
