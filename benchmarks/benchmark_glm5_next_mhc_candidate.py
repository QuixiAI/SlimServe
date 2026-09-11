# SPDX-License-Identifier: Apache-2.0
"""Compare mHC serving and diagnostic schedules against owned native code."""

import argparse
import functools
import json
import statistics

import torch
from glm5_next_mhc_candidate import transition
from triton.testing import do_bench_cudagraph

from vllm import _custom_ops as ops
from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition
from vllm.quixicore.ops import quixicore_ops as qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 64])
    parser.add_argument("--only-serving", action="store_true")
    parser.add_argument("--prototype-bm", type=int, choices=[1, 16], default=1)
    args = parser.parse_args()
    torch.manual_seed(23)
    rows = []
    for m in args.tokens:
        residual = torch.randn(m, 4, 4096, device="cuda", dtype=torch.bfloat16)
        x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
        fn = torch.randn(24, 16384, device="cuda") * 0.01
        scale = torch.tensor([0.2, 0.2, 0.2], device="cuda")
        base = torch.randn(24, device="cuda") * 0.01
        weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        post, comb, _ = qc.dsv4_mhc_pre(
            residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20
        )
        native_args = (
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
        native = functools.partial(qc.dsv4_mhc_fused_post_pre, *native_args)
        expected = native()
        norm_expected = torch.empty_like(x)
        ops.rms_norm(norm_expected, expected[-1], weight, 1e-5)

        def native_norm(native=native, weight=weight, norm_expected=norm_expected):
            result = native()
            ops.rms_norm(norm_expected, result[-1], weight, 1e-5)
            return *result[:-1], norm_expected

        serving = functools.partial(
            mhc_transition,
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-5,
            1e-6,
            2.0,
            20,
            weight,
            1e-5,
        )
        serving_result = serving()
        torch.testing.assert_close(serving_result[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(
            serving_result[-1], norm_expected, atol=0.008, rtol=0.008
        )
        arms = {"native": native, "native_norm": native_norm, "serving_norm": serving}
        errors = {}
        for bk in () if args.only_serving else (256, 512, 1024):
            for norm in (False, True):
                name = f"bk{bk}_norm{int(norm)}"
                call = functools.partial(
                    transition,
                    x,
                    residual,
                    post,
                    comb,
                    fn,
                    scale,
                    base,
                    weight if norm else None,
                    bk=bk,
                    bm=args.prototype_bm,
                )
                try:
                    result = call()
                    torch.cuda.synchronize()
                except Exception as error:
                    # Failed runs are inconclusive, not timing evidence or
                    # a reason to hide a numerical mismatch or install fallback.
                    errors[name] = repr(error)
                    print(name, repr(error), flush=True)
                    continue
                torch.testing.assert_close(result[0], expected[0], atol=0, rtol=0)
                torch.testing.assert_close(result[1], expected[1], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(result[2], expected[2], atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    result[3],
                    norm_expected if norm else expected[3],
                    atol=0.008,
                    rtol=0.008,
                )
                arms[name] = call
        torch.cuda.synchronize()
        times = {name: [] for name in arms}
        for repeat in range(3):
            for name in list(arms)[:: 1 if repeat % 2 == 0 else -1]:
                times[name].append(do_bench_cudagraph(arms[name], rep=100) * 1000)
        row = {
            "tokens": m,
            "us": times,
            "medians": {name: statistics.median(v) for name, v in times.items()},
            "inconclusive_tiles": errors,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    with open(args.out, "x") as stream:
        json.dump(rows, stream, indent=2)


if __name__ == "__main__":
    main()
