# SPDX-License-Identifier: Apache-2.0
"""Build or time the isolated mHC urgent/deferred graph experiment.

Use --compile-only with CUDA_VISIBLE_DEVICES='' while a serving gate owns
the GPUs. Real timing includes an actual KDA-shaped BF16 projection or the
FP32 MoE router projection, not a synthetic sleep or independent busy loop.
"""

import argparse
import functools
import json

import torch
import triton
from glm5_next_mhc_deferred import DeferredTransition, _finalize_phase
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition


def compile_only():
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = {
        "PART": "*fp32",
        "RES": "*bf16",
        "SCALE": "*fp32",
        "BASE": "*fp32",
        "POST": "*fp32",
        "COMB": "*fp32",
        "OUT": "*bf16",
        "NORM": "*bf16",
    }
    for deferred in (False, True):
        constants = dict(
            RMS_EPS=1e-5,
            HC_EPS=1e-6,
            POST_MULT=2.0,
            ITERATIONS=20,
            NORM_EPS=1e-5,
            DEFERRED=deferred,
        )
        kernel = triton.compile(
            ASTSource(_finalize_phase, signature, constexprs=constants),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4},
        )
        print(
            json.dumps(
                dict(
                    deferred=deferred,
                    hash=kernel.hash,
                    shared=kernel.metadata.shared,
                    ptx_lines=len(kernel.asm["ptx"].splitlines()),
                )
            ),
            flush=True,
        )


@torch.no_grad()
def measure(m, consumer, opaque=False):
    torch.manual_seed(6000 + m)
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(m, 4, 4096, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 16384, device="cuda") * 0.01
    scale = torch.tensor([0.2, 0.2, 0.2], device="cuda")
    base = torch.randn(24, device="cuda") * 0.01
    norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    _, post, comb, _ = mhc_transition(
        None, residual, None, None, fn, scale, base, 1e-5, 1e-6, 2.0, 20
    )
    fp32 = consumer == "router"
    # TP8 KDA: 3*1024 qkv + 8 beta + replicated f_a/g_a, each128.
    width = 288 if fp32 else 3336
    weight = torch.randn(width, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
    plan = DeferredTransition(
        x, residual, post, comb, fn, scale, base, norm, weight, projection_fp32=fp32
    )
    projected = torch.empty(
        m, width, device="cuda", dtype=torch.float32 if fp32 else x.dtype
    )

    def baseline():
        result = mhc_transition(
            x, residual, post, comb, fn, scale, base, 1e-5, 1e-6, 2.0, 20, norm, 1e-5
        )
        if fp32:
            torch.mm(result[-1], weight.T, out_dtype=torch.float32, out=projected)
        else:
            torch.mm(result[-1], weight.T, out=projected)
        return *result, projected

    def compare(actual, expected):
        for i, (a, b) in enumerate(zip(actual, expected)):
            # Residual, normalized BF16 input and consumer must be exact.
            # FP32 coefficients retain the established tight rounding gate.
            tol = 1e-6 if i in (1, 2) else 0
            if i == 1:
                # Public opaque op exposes [T,4,1]; low-level prototype [T,4].
                a = a.reshape_as(b)
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)

    for overlap in (False, True):
        call = functools.partial(plan.run, overlap=overlap)
        expected = tuple(t.clone() for t in baseline())
        compare(call(), expected)
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for _ in range(8):
            x.normal_()
            residual.normal_()
            graph.replay()
            expected = baseline()
            compare(actual, expected)
        del graph
    torch.cuda.synchronize()
    arms = {
        "baseline": baseline,
        "serial_split": functools.partial(plan.run, overlap=False),
        "overlapped": functools.partial(plan.run, overlap=True),
    }
    if opaque:
        from benchmarks import glm5_next_mhc_project  # noqa: F401

        def opaque_call():
            return torch.ops.vllm.glm5_mhc_project_candidate(
                x,
                residual,
                post,
                comb,
                fn,
                scale,
                base,
                norm,
                weight,
                plan.side.cuda_stream,
                fp32,
            )

        compare(opaque_call(), baseline())
        arms["opaque_allocating"] = opaque_call
    timings = {name: [] for name in arms}
    for repeat in range(3):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            timings[name].append(do_bench_cudagraph(arms[name], rep=100) * 1000)
    return dict(
        tokens=m, consumer=consumer, parity=True, changed_input_replays=16, us=timings
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--opaque", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        compile_only()
        return
    for m in args.tokens:
        for consumer in ("kda", "router"):
            print(json.dumps(measure(m, consumer, args.opaque)), flush=True)


if __name__ == "__main__":
    main()
