# SPDX-License-Identifier: Apache-2.0
"""Isolated prefill score timing over eleven caches; not end-to-end TPS.

Run only after other GPU jobs exit. GPU parity tests must pass first.
"""

import argparse
import json
import statistics
import sys

import torch
from triton.testing import do_bench_cudagraph

from benchmarks.glm5_next_pool_layout import packed_indexer_cache
from benchmarks.glm5_next_pool_query_split_candidate import split_query_pool_logits
from benchmarks.glm5_next_pool_query_tile_candidate import query_tiled_pool_logits
from vllm.model_executor.layers.glm5_next_pool_cache import cached_pool_logits


@torch.no_grad()
def measure(rows, context, layout, variants=("query2", "query4"), block_size=4608):
    print(json.dumps(dict(event="start", rows=rows, context=context, layout=layout)),
          file=sys.stderr, flush=True)
    torch.manual_seed(93400 + rows + context)
    layers, bs = 11, block_size
    pages = (context + bs - 1) // bs
    cache = packed_indexer_cache(layers, 3 * pages, bs, "cuda")
    table = torch.randperm(3 * pages, device="cuda").int().view(3, pages)
    q = torch.randn(layers, rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(layers, rows, 32, device="cuda") * 32**-0.5
    ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    req = (ids // max(1, rows // 3) if layout == "grouped" else ids) % 3
    if layout == "mixed":
        req = torch.where(ids < rows // 2, 0, ids % 3)
    visible = context - rows + ids + 1
    columns = context // 4 + 1
    buffers = {name: torch.full((rows * columns + 64,), -77., device="cuda")
               for name in ("native", *variants)}
    outputs = {name: buf[32:-32].view(rows, columns)
               for name, buf in buffers.items()}
    valid = torch.arange(columns, device="cuda")[None, :] < visible[:, None] // 4

    def one(name, layer):
        args = (q[layer], weights[layer], cache[layer], table, req, visible,
                outputs[name])
        if name == "native":
            cached_pool_logits(*args)
        else:
            function = (split_query_pool_logits if name.startswith("split")
                        else query_tiled_pool_logits)
            function(*args, query_tile=int(name[-1]))

    bitexact = {name: True for name in variants}
    for layer in range(layers):
        for name in outputs:
            outputs[name].fill_(float("-inf"))
            one(name, layer)
        expected = outputs["native"].masked_fill(~valid, 0)
        for name in bitexact:
            actual = outputs[name].masked_fill(~valid, 0)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            bitexact[name] &= torch.equal(actual, expected)
            assert torch.equal(
                outputs[name].masked_fill(~valid, float("-inf")).topk(512)
                .indices.sort().values,
                outputs["native"].masked_fill(~valid, float("-inf")).topk(512)
                .indices.sort().values,
            )
        assert all((b[:32] == -77).all() and (b[-32:] == -77).all()
                   for b in buffers.values())

    def run(name):
        for layer in range(layers):
            one(name, layer)

    timing = {name: [] for name in outputs}
    for repeat in range(5):
        for name in (list(outputs) if repeat % 2 == 0 else list(reversed(outputs))):
            timing[name].append(do_bench_cudagraph(lambda name=name: run(name), rep=100)
                                * 1000 / layers)
    # Timing leaves the last layer in each output; verify it again.
    for name in bitexact:
        torch.testing.assert_close(outputs[name].masked_fill(~valid, 0),
                                   outputs["native"].masked_fill(~valid, 0),
                                   atol=1e-6, rtol=1e-6)
    assert all((b[:32] == -77).all() and (b[-32:] == -77).all()
               for b in buffers.values())
    print(json.dumps(dict(
        rows=rows, context=context, layout=layout, layers=layers,
        block_size=bs, page_stride_bytes=cache[0].stride(0) * cache.element_size(),
        cache_layout="page-major eleven-column slab",
        strict_native_parity=True, exact_selected_sets=True, guards=True,
        bitexact=bitexact, us_per_layer=timing,
        median_us={name: statistics.median(values) for name, values in timing.items()},
        scope="Isolated score only; not model throughput or KV-capacity evidence",
    )), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 131072])
    parser.add_argument("--block-size", type=int, default=4608)
    parser.add_argument("--variants", nargs="+", choices=["query2", "query4",
                                                         "split2", "split4"],
                        default=["query2", "query4"])
    parser.add_argument("--layouts", nargs="+",
                        choices=["grouped", "interleaved", "mixed"],
                        default=["grouped", "interleaved"])
    args = parser.parse_args()
    if min(args.rows) < 1 or min(args.contexts) - max(args.rows) < 2048:
        parser.error("Require positive rows and at least512complete pools per row")
    if args.block_size < 64 or args.block_size % 64:
        parser.error("Block size must be a positive multiple of64")
    for context in args.contexts:
        for rows in args.rows:
            for layout in args.layouts:
                measure(rows, context, layout, tuple(dict.fromkeys(args.variants)),
                        args.block_size)


if __name__ == "__main__":
    main()
