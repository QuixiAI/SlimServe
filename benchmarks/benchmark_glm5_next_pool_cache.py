# SPDX-License-Identifier: Apache-2.0
"""Isolated CUDA-graph update+score timing; not serving throughput."""

import argparse
import json
import statistics

import torch
import triton
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.glm5_next_indexer import (
    _insert_rows_kernel,
    _pooled_logits_kernel,
)
from vllm.model_executor.layers.glm5_next_pool_cache import (
    _cached_pool_logits,
    cached_pool_logits,
    update_pool_cache,
)


@torch.no_grad()
def measure(rows, context, tune_score=False, adaptive=False):
    torch.manual_seed(42)
    raw_bs, compact_bs = 1152, 4608
    raw_pages = triton.cdiv(context, raw_bs)
    compact_pages = triton.cdiv(context, compact_bs)
    raw = torch.empty(
        rows * raw_pages, 2, raw_bs, 256, device="cuda", dtype=torch.bfloat16
    )[:, 0]
    compact = torch.empty(
        rows * compact_pages, 2, compact_bs, 64, device="cuda", dtype=torch.bfloat16
    )[:, 0]
    raw_bt = torch.arange(rows * raw_pages, device="cuda", dtype=torch.int32).view(
        rows, -1
    )
    compact_bt = torch.arange(
        rows * compact_pages, device="cuda", dtype=torch.int32
    ).view(rows, -1)
    ape = torch.randn(4, 128, device="cuda")
    current = []
    positions = torch.arange(context, device="cuda", dtype=torch.int64)

    def raw_insert(src, slots):
        _insert_rows_kernel[(triton.cdiv(len(slots), 64),)](
            src,
            raw,
            slots,
            len(slots),
            raw.stride(0),
            BLOCK_SIZE=raw_bs,
            ROW=256,
            BLOCK_T=64,
        )

    for req in range(rows):
        source = torch.randn(context, 256, device="cuda", dtype=torch.bfloat16)
        raw_insert(source, positions + req * raw_pages * raw_bs)
        update_pool_cache(
            source, positions + req * compact_pages * compact_bs, ape, compact
        )
        current.append(source[-1:].clone())
    packed = torch.cat(current)
    ids = torch.arange(rows, device="cuda", dtype=torch.int64)
    raw_slots = ids * raw_pages * raw_bs + context - 1
    compact_slots = ids * compact_pages * compact_bs + context - 1
    q = torch.randn(rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, 32, device="cuda") * 32**-0.5
    requests = ids.int()
    visible = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    reference = torch.empty(rows, 1048576 // 4, device="cuda")
    result = torch.empty_like(reference)

    def old_score():
        _pooled_logits_kernel[(rows, 128)](
            q,
            weights,
            ape,
            raw,
            raw_bt,
            requests,
            visible,
            reference,
            reference.shape[1],
            raw_bt.stride(0),
            raw.stride(0),
            128**-0.5,
            BLOCK_SIZE=raw_bs,
            H=32,
            D=128,
            KP=4,
            ROW=256,
            BLOCK_P=16,
        )

    def new_score():
        cached_pool_logits(q, weights, compact, compact_bt, requests, visible, result)

    def old_step():
        raw_insert(packed, raw_slots)
        old_score()

    def new_step():
        update_pool_cache(packed, compact_slots, ape, compact)
        new_score()

    old_step()
    new_step()
    expected, actual = reference[:, : context // 4], result[:, : context // 4]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    if context // 4 > 512:
        assert torch.equal(
            actual.topk(512).indices.sort().values,
            expected.topk(512).indices.sort().values,
        )
    torch.cuda.synchronize()
    metrics = dict(
        rows=rows,
        context=context,
        max_logit_error=(actual - expected).abs().max().item(),
    )
    for name, fn in (
        ("raw_score_us", old_score),
        ("compact_score_us", new_score),
        ("raw_step_us", old_step),
        ("compact_step_us", new_step),
    ):
        metrics[name] = 1000 * do_bench_cudagraph(fn, rep=100)
    if adaptive:
        from benchmarks.glm5_next_adaptive_pool_score import adaptive_pool_logits

        def candidate():
            adaptive_pool_logits(
                q, weights, compact, compact_bt, requests, visible, result
            )

        candidate()
        torch.testing.assert_close(
            result[:, : context // 4], expected, atol=1e-6, rtol=1e-6
        )
        if context // 4 > 512:
            assert torch.equal(
                result[:, : context // 4].topk(512).indices.sort().values,
                expected.topk(512).indices.sort().values,
            )
        metrics["adaptive_parity"] = True
        timings = {"baseline": [], "adaptive": []}
        arms = {"baseline": new_score, "adaptive": candidate}
        for repeat in range(3):
            order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
            for name in order:
                timings[name].append(1000 * do_bench_cudagraph(arms[name], rep=100))
        metrics["adaptive_comparison_us"] = timings
    if tune_score:
        metrics["score_sweep"] = []
        for bp in (16, 32, 64, 128):
            for programs in (32, 64, 128, 256):

                def candidate(bp=bp, programs=programs):
                    _cached_pool_logits[(rows, programs)](
                        q,
                        weights,
                        compact,
                        compact_bt,
                        requests,
                        visible,
                        result,
                        result.shape[1],
                        compact_bt.stride(0),
                        compact.stride(0),
                        128**-0.5,
                        compact_bs,
                        32,
                        bp,
                        num_warps=4,
                    )

                record = dict(pool_tile=bp, programs=programs, warps=4)
                try:
                    candidate()
                    torch.testing.assert_close(
                        result[:, : context // 4], expected, atol=1e-6, rtol=1e-6
                    )
                    if context // 4 > 512:
                        assert torch.equal(
                            result[:, : context // 4].topk(512).indices.sort().values,
                            expected.topk(512).indices.sort().values,
                        ), "top-512 set changed"
                    record["parity"] = True
                except AssertionError as error:
                    record.update(parity=False, error=str(error))
                    metrics["score_sweep"].append(record)
                    continue
                record["us"] = [
                    1000 * do_bench_cudagraph(candidate, rep=50) for _ in range(3)
                ]
                record["median_us"] = statistics.median(record["us"])
                metrics["score_sweep"].append(record)
    return metrics


def compile_score_tiles():
    """CPU-only SM80 compilation; not a correctness or speed measurement."""
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(
        Q="*bf16",
        W="*fp32",
        CACHE="*bf16",
        BT="*i32",
        ROW_REQ="*i32",
        VISIBLE="*i32",
        OUT="*fp32",
        MAX_POOLS="i32",
        BT_STRIDE="i32",
        PAGE_STRIDE="i32",
        SCALE="fp32",
    )
    for bp in (16, 32, 64, 128):
        kernel = triton.compile(
            ASTSource(
                _cached_pool_logits, signature, constexprs=dict(BS=4608, H=32, BP=bp)
            ),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4},
        )
        print(
            json.dumps(
                dict(pool_tile=bp, hash=kernel.hash, shared=kernel.metadata.shared)
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[1000, 16384, 131072]
    )
    parser.add_argument(
        "--tune-score",
        action="store_true",
        help="parity-gated tile/program sweep; no serving source changes",
    )
    parser.add_argument("--compile-score-only", action="store_true")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="compare adaptive tile/CTA candidate; use PYTHONPATH=.",
    )
    args = parser.parse_args()
    if args.compile_score_only:
        compile_score_tiles()
        return
    for rows in args.rows:
        for context in args.contexts:
            print(
                json.dumps(measure(rows, context, args.tune_score, args.adaptive)),
                flush=True,
            )


if __name__ == "__main__":
    main()
