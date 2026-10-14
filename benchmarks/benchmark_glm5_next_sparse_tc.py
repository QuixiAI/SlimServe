# SPDX-License-Identifier: Apache-2.0
"""Isolated sparse MLA comparison with eleven strided, disjoint layer caches.

Run only after serving releases the GPU. At c32/131K the actual BF16 cache
allocation is about48GB; do not silently shrink the requested workload.
Passing this comparison is not a serving or long-context quality gate.
"""

import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

from benchmarks.glm5_next_sparse_tc_candidate import sparse_tc_nope
from vllm.quixicore import quixicore_ops as qc


@torch.no_grad()
def measure(rows, context, heads, layers):
    torch.manual_seed(12400 + rows + context + heads)
    bs, width = 576, 2080
    pages = (context + bs - 1) // bs
    cache_bytes = rows * pages * layers * bs * 512 * 2
    free, _ = torch.cuda.mem_get_info()
    if cache_bytes > free * 0.8:
        raise RuntimeError(f"requested cache needs{cache_bytes}bytes; only{free}free")
    # Same page stride as the eleven-layer packed MLA group. Different
    # requests own different physical pages, including at full c32.
    bank = torch.empty(
        rows * pages, layers, bs, 512, device="cuda", dtype=torch.bfloat16
    )
    bank.normal_(std=0.5)
    caches = [bank[:, layer] for layer in range(layers)]
    table = torch.arange(rows * pages, device="cuda", dtype=torch.int32).reshape(
        rows, pages
    )
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    indices = torch.full((rows, width), -1, device="cuda", dtype=torch.int32)
    selected = min(2048, context)
    for row in range(rows):
        indices[row, :selected] = torch.randperm(context, device="cuda")[
            :selected
        ].int()
        # Model expansion reserves the last32 positions for the local tail.
        # Keep the middle -1 holes in short-context measurements.
        indices[row, 2048:] = torch.arange(context - 32, context, device="cuda").int()
    tlen = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    scale = 256**-0.5  # Actual registered QK head width before latent absorption.

    def native():
        return [
            qc.mla_decode_bf16_sparse_nope(
                q,
                cache,
                table,
                indices,
                tlen,
                bs,
                scale,
                128,
                cache.stride(0) * cache.element_size(),
            )
            for cache in caches
        ]

    arms = {"native_p128": native}
    for split in (32, 64, 128):

        def candidate(split=split):
            return [
                sparse_tc_nope(q, cache, table, indices, tlen, scale, split=split)
                for cache in caches
            ]

        arms[f"tc_split{split}"] = candidate
    errors = {}
    for name, fn in arms.items():
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = fn()
        maximum = 0.0
        changed = 0
        for _ in range(3):
            q.normal_(std=0.2)
            graph.replay()
            expected = native()
            for a, b in zip(actual, expected):
                error = (a.float() - b.float()).abs().max().item()
                assert error < 1e-3, (name, rows, context, error)
                maximum = max(maximum, error)
                changed += (a != b).count_nonzero().item()
        errors[name] = dict(max_abs_vs_native=maximum, changed_values=changed)
        del graph, actual, expected
    times = {name: [] for name in arms}
    for repeat in range(5):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            times[name].append(1000 * do_bench_cudagraph(arms[name], rep=100) / layers)
    return dict(
        rows=rows,
        heads=heads,
        physical_context=context,
        selected_nonnegative_entries=selected + 32,
        layers=layers,
        cache_bytes=cache_bytes,
        page_stride_bytes=caches[0].stride(0) * 2,
        softmax_scale=scale,
        errors=errors,
        us_per_layer=times,
        short_context_cache_caveat="May be L2-hot; full serving A/B remains required.",
        serving_speedup_claim=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--contexts", type=int, nargs="+", default=[1000, 131072])
    parser.add_argument("--heads", type=int, nargs="+", default=[8])
    parser.add_argument("--layers", type=int, default=11)
    args = parser.parse_args()
    if min(args.rows + [args.layers]) < 1 or min(args.contexts) < 32:
        parser.error("positive rows/layers and context>=32 required")
    if any(heads not in (8, 16) for heads in args.heads):
        parser.error("heads must be8 or16")
    for context in args.contexts:
        for rows in args.rows:
            for heads in args.heads:
                print(
                    json.dumps(measure(rows, context, heads, args.layers)), flush=True
                )
