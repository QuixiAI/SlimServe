#!/usr/bin/env python3
"""CUDA-graph timing of the actual pooled-indexer serving kernels (micro only)."""

import argparse
import hashlib
import json
from functools import partial

import torch
import triton
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers import glm5_next_indexer as indexer
from vllm.model_executor.layers.glm5_next_indexer import (
    _POOL_PROGRAMS,
    _ROW_DIM,
    _pooled_logits_kernel,
    _pooled_select,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[1000, 4096, 32768])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--schedule", choices=["tensor", "one_pool"], default="tensor")
    args = parser.parse_args()
    candidate = _pooled_logits_kernel
    if args.schedule == "one_pool":
        from glm5_next_pool_candidate import pooled_logits_one_pool

        candidate = pooled_logits_one_pool
    torch.manual_seed(42)
    results = []
    for rows in args.rows:
        for context in args.contexts:
            device = "cuda"
            block_size, heads, dim, kp, ksel = 64, 32, 128, 4, 512
            max_pools = 1048576 // kp
            nblocks = triton.cdiv(context, block_size)
            q = torch.randn(rows, heads, dim, device=device, dtype=torch.bfloat16)
            weights = torch.randn(rows, heads, device=device) * heads**-0.5
            ape = torch.randn(kp, dim, device=device)
            # Include a gap between physical pages, as in packed tier slabs.
            backing = torch.randn(
                rows * nblocks,
                2,
                block_size,
                _ROW_DIM,
                device=device,
                dtype=torch.bfloat16,
            )
            cache = backing[:, 0]
            bt = torch.arange(rows * nblocks, device=device, dtype=torch.int32).view(
                rows, -1
            )
            row_req = torch.arange(rows, device=device, dtype=torch.int32)
            visible = torch.full((rows,), context, device=device, dtype=torch.int32)
            logits = torch.empty(rows, max_pools, device=device)
            output = torch.empty(rows, 2080, device=device, dtype=torch.int32)

            score = partial(
                candidate[(rows, _POOL_PROGRAMS)],
                q,
                weights,
                ape,
                cache,
                bt,
                row_req,
                visible,
                logits,
                max_pools,
                bt.stride(0),
                cache.stride(0),
                dim**-0.5,
                BLOCK_SIZE=block_size,
                H=heads,
                D=dim,
                KP=kp,
                ROW=_ROW_DIM,
                BLOCK_P=16,
            )
            select = partial(
                _pooled_select,
                q,
                weights,
                ape,
                cache,
                bt,
                row_req,
                visible,
                logits,
                max_pools,
                block_size,
                dim**-0.5,
                ksel,
                output,
                kp,
            )

            indexer._pooled_logits_kernel = _pooled_logits_kernel
            select()
            reference_logits = logits[:, : context // kp].clone()
            reference_indices = output.clone().sort(dim=1).values
            indexer._pooled_logits_kernel = candidate
            score()
            select()
            torch.testing.assert_close(
                logits[:, : context // kp], reference_logits, atol=1e-3, rtol=2e-3
            )
            assert torch.equal(output.sort(dim=1).values, reference_indices), (
                "selected sets changed"
            )
            # do_bench_cudagraph uses a fresh stream. Publish random inputs
            # and block tables before its warmup (otherwise it can race the
            # default stream's initialization and dereference garbage pages).
            torch.cuda.synchronize()
            results.append(
                dict(
                    rows=rows,
                    context=context,
                    schedule=args.schedule,
                    max_logit_error=(logits[:, : context // kp] - reference_logits)
                    .abs()
                    .max()
                    .item(),
                    logits_us=1000 * do_bench_cudagraph(score, rep=20),
                    select_us=1000 * do_bench_cudagraph(select, rep=20),
                    logits_sha256=hashlib.sha256(
                        logits[:, : context // kp].contiguous().cpu().numpy().tobytes()
                    ).hexdigest(),
                    indices_sha256=hashlib.sha256(
                        output.cpu().numpy().tobytes()
                    ).hexdigest(),
                )
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
