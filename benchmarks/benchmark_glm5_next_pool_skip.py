# SPDX-License-Identifier: Apache-2.0
"""Isolated eleven-layer score/top-k/expand timing, not model throughput.

Run only on an idle GPU, after test_pool_skip_candidate.py passes on CUDA.
The full 1M-context logits stride is retained even for short visible lengths.
"""

import argparse
import json
import statistics

import torch
from triton.testing import do_bench_cudagraph

from benchmarks.glm5_next_pool_layout import packed_indexer_cache
from benchmarks.glm5_next_pool_skip_candidate import select_without_trivial_scores
from vllm.model_executor.layers.glm5_next_indexer import _pooled_select


@torch.no_grad()
def measure(rows, context, mixed, block_size=4608):
    torch.manual_seed(103000 + rows + context)
    layers, bs, columns = 11, block_size, 262144
    pages = (context + bs - 1) // bs
    cache = packed_indexer_cache(layers, 3 * pages, bs, "cuda")
    table = torch.randperm(3 * pages, device="cuda").int().view(3, pages)
    q = torch.randn(layers, rows, 32, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(layers, rows, 32, device="cuda")
    ids = torch.arange(rows, device="cuda", dtype=torch.int32)
    req = ids % 3
    visible = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    if mixed:
        visible.copy_(torch.where(ids % 2 == 0, min(context, 2051), context))
    ape = torch.zeros(4, 128, device="cuda")
    logits = {name: torch.full((layers, rows, columns), float("nan"), device="cuda")
              for name in ("native", "skip")}
    buffers = {name: torch.full((layers * rows * 2080 + 64,), -77, device="cuda",
                                dtype=torch.int32) for name in logits}
    outputs = {name: buf[32:-32].view(layers, rows, 2080)
               for name, buf in buffers.items()}
    selected = torch.empty(layers, rows, 512, device="cuda", dtype=torch.int32)

    def run(name):
        for layer in range(layers):
            if name == "native":
                _pooled_select(q[layer], weights[layer], ape, cache[layer], table,
                               req, visible, logits[name][layer], columns, bs,
                               128**-0.5, 512, outputs[name][layer], 4)
            else:
                select_without_trivial_scores(
                    q[layer], weights[layer], cache[layer], table, req, visible,
                    logits[name][layer], selected[layer], outputs[name][layer],
                )

    short = visible // 4 <= 512

    def check():
        assert torch.equal(outputs["skip"].sort().values,
                           outputs["native"].sort().values)
        assert torch.equal(outputs["skip"][:, short], outputs["native"][:, short])
        assert torch.isnan(logits["skip"][:, short]).all()
        assert all((buf[:32] == -77).all() and (buf[-32:] == -77).all()
                   for buf in buffers.values())
        valid = ((torch.arange(columns, device="cuda")[None, :] < visible[:, None] // 4)
                 & ~short[:, None])
        torch.testing.assert_close(logits["skip"][:, valid], logits["native"][:, valid],
                                   atol=0, rtol=0)

    for name in outputs:
        run(name)
    check()
    timing = {name: [] for name in outputs}
    for repeat in range(5):
        for name in (list(outputs) if repeat % 2 == 0 else list(reversed(outputs))):
            timing[name].append(do_bench_cudagraph(lambda name=name: run(name), rep=100)
                                * 1000 / layers)
    check()
    print(json.dumps(dict(
        rows=rows, context=context, mixed=mixed, layers=layers,
        block_size=bs, page_stride_bytes=cache[0].stride(0) * cache.element_size(),
        cache_layout="page-major eleven-column slab",
        full_context_logit_columns=columns, us_per_layer=timing,
        median_us={n: statistics.median(v) for n, v in timing.items()},
        long_logits_bitexact=True, expanded_sets_exact=True,
        short_expanded_order_exact=True, skipped_logits_untouched=True,
        guards=True, scope="Isolated selector only; no serving TPS claim",
    )), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[1000, 2051, 2052, 131072])
    parser.add_argument("--block-size", type=int, default=4608,
                        help="Registered compact TP8 indexer uses4608, not MLA576")
    args = parser.parse_args()
    if min(args.rows) < 1 or min(args.contexts) < 4 or max(args.contexts) > 1048576:
        parser.error("Require positive rows and contexts between4 and1048576")
    if args.block_size < 64 or args.block_size % 64:
        parser.error("Block size must be a positive multiple of64")
    for rows in args.rows:
        for context in args.contexts:
            measure(rows, context, False, args.block_size)
        if rows > 1:
            measure(rows, max(args.contexts), True, args.block_size)


if __name__ == "__main__":
    main()
