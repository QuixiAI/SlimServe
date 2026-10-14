# SPDX-License-Identifier: Apache-2.0
"""Paired KDA gate projection tile tuning and serving-dispatch comparison."""

import argparse
import functools
import json
import statistics

import torch
import triton
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.mamba.ops.kda_gate_projection import (
    _gate_pair_kernel,
    kda_gate_pair,
)


def paired(af, ag, wf, wg, bn):
    m, n = af.shape[0], wf.shape[0]
    out_f = torch.empty((m, n), dtype=af.dtype, device=af.device)
    out_g = torch.empty_like(out_f)
    _gate_pair_kernel[(triton.cdiv(n, bn), triton.cdiv(m, 16), 2)](
        af,
        ag,
        wf,
        wg,
        out_f,
        out_g,
        m,
        n,
        af.stride(0),
        ag.stride(0),
        16,
        bn,
        num_warps=4,
    )
    return out_f, out_g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 64, 257])
    args = parser.parse_args()
    torch.manual_seed(42)
    rows = []
    for n in (1024, 2048):
        for m in args.tokens:
            # Both inputs are views into the same merged projection, with
            # the actual local Q/K/V/beta width before the two gate inputs.
            backing = torch.randn(
                m, 3 * n + n // 128 + 256, device="cuda", dtype=torch.bfloat16
            )
            af, ag = backing[:, -256:-128], backing[:, -128:]
            wf = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16) * 0.1
            wg = torch.randn_like(wf) * 0.1

            def reference(af=af, ag=ag, wf=wf, wg=wg):
                return af @ wf.T, ag @ wg.T

            expected = reference()
            arms = {
                "separate": reference,
                "serving": functools.partial(kda_gate_pair, af, ag, wf, wg),
            }
            for bn in (16, 32, 64):
                call = functools.partial(paired, af, ag, wf, wg, bn)
                for actual, ref in zip(call(), expected):
                    torch.testing.assert_close(actual, ref, atol=0.008, rtol=0.008)
                arms[f"paired_n{bn}"] = call
            torch.cuda.synchronize()
            timings = {name: [] for name in arms}
            for repeat in range(3):
                for name in list(arms)[:: 1 if repeat % 2 == 0 else -1]:
                    timings[name].append(1000 * do_bench_cudagraph(arms[name], rep=50))
            row = {
                "tokens": m,
                "local_dim": n,
                "us": timings,
                "medians": {
                    key: statistics.median(value) for key, value in timings.items()
                },
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    with open(args.out, "x") as stream:
        json.dump(rows, stream, indent=2)


if __name__ == "__main__":
    main()
