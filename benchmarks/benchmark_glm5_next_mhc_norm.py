# SPDX-License-Identifier: Apache-2.0
"""Graph timings for GLM mHC + RMSNorm, without changing serving dispatch."""

import argparse
import functools
import json
import statistics

import torch
import triton

from vllm import _custom_ops as ops
from vllm.quixicore.ops import quixicore_ops as qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 8, 16, 64, 257, 576]
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.manual_seed(37)
    rows = []
    for tokens in args.tokens:
        residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(tokens, 4096, device="cuda", dtype=torch.bfloat16)
        fn = torch.randn(24, 16384, device="cuda") * 0.01
        scale = torch.tensor([0.2, 0.2, 0.2], device="cuda")
        base = torch.randn(24, device="cuda") * 0.01
        weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        post, comb, _ = qc.dsv4_mhc_pre(
            residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20
        )
        call_args = (
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-5,
            1e-6,
            1e-6,
            2.0,
            20,
        )
        fused = functools.partial(qc.dsv4_mhc_fused_post_pre, *call_args, weight, 1e-5)
        norm_out = torch.empty_like(x)

        def separate(call_args=call_args, norm_out=norm_out, weight=weight):
            layer = qc.dsv4_mhc_fused_post_pre(*call_args)[-1]
            ops.rms_norm(norm_out, layer, weight, 1e-5)
            return norm_out

        expected = separate()
        actual = fused()[-1]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0.008)
        torch.cuda.synchronize()  # timer captures on a fresh stream
        timings = {"separate_us": [], "fused_us": []}
        for repeat in range(5):
            arms = [("separate_us", separate), ("fused_us", fused)]
            if repeat % 2:
                arms.reverse()
            for name, call in arms:
                timings[name].append(
                    triton.testing.do_bench_cudagraph(call, rep=200) * 1000
                )
        row = {"tokens": tokens, **timings}
        row["speedup"] = statistics.median(timings["separate_us"]) / statistics.median(
            timings["fused_us"]
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.out, "x") as stream:
        json.dump(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "rows": rows,
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
