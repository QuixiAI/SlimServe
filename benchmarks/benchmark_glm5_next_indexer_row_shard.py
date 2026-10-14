# SPDX-License-Identifier: Apache-2.0
"""Quarantined TP indexer row-sharding comparison; no serving imports this file.

Run with torchrun on otherwise idle GPUs. Each query still scores every
visible pool with the unchanged retained kernel. NCCL gathers only selected
pool indices. This measures score + top-k + communication, excluding common
cache updates, query projections and pool-to-token expansion. Eleven disjoint
cache layers exceed A100 L2 at 131K, but shorter contexts can remain L2-hot
unlike serving with intervening weight reads. Short contexts are measured too:
the collective can cost more than the replicated work it removes.
"""

import argparse
import json
import os
import statistics
from datetime import timedelta


def row_shard(rows, world, rank):
    if rows < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError("positive rows/world and an in-range rank are required")
    local_rows = (rows + world - 1) // world
    start = rank * local_rows
    return local_rows, start, min(start + local_rows, rows), local_rows * world


def measure(rows, context, *, layers, repeats, steps):
    import torch
    import torch.distributed as dist

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.glm5_next_pool_cache import cached_pool_logits

    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    # Below one query per rank, retain the unchanged replicated path.
    shard = rows >= world and world > 1
    local_rows, start, stop, padded_rows = row_shard(rows, world, rank)
    if not shard:
        local_rows, start, stop, padded_rows = rows, 0, rows, rows
    bs, heads, dim, ksel, max_pools = 4608, 32, 128, 512, 262144
    pages = (context + bs - 1) // bs
    generator = torch.Generator().manual_seed(4300 + rows + context)

    def random(shape, dtype):
        # CPU construction gives identical replicas on every rank.
        return torch.randn(shape, generator=generator, dtype=dtype).to(device)

    cache_bank = random((layers, pages, 2, bs, 64), torch.bfloat16)
    caches = [cache_bank[layer, :, 0] for layer in range(layers)]
    q = random((layers, padded_rows, heads, dim), torch.bfloat16)
    weights = random((layers, padded_rows, heads), torch.float32) * 0.1
    table = torch.stack((torch.arange(pages), torch.arange(pages).flip(0))).to(
        device=device, dtype=torch.int32
    )
    requests = torch.arange(padded_rows, device=device, dtype=torch.int32) % 2
    lengths_cpu = torch.full((padded_rows,), context, dtype=torch.int32)
    lengths_cpu[rows:] = 0
    visible = lengths_cpu.to(device)
    full_logits = torch.empty(rows, max_pools, device=device)
    local_logits = torch.empty(local_rows, max_pools, device=device)
    full_selected = torch.empty(layers, rows, ksel, device=device, dtype=torch.int32)
    local_selected = torch.empty(
        layers, local_rows, ksel, device=device, dtype=torch.int32
    )
    gathered = torch.empty(layers, padded_rows, ksel, device=device, dtype=torch.int32)
    full_zeros = torch.zeros(rows, device=device, dtype=torch.int32)
    local_zeros = torch.zeros(local_rows, device=device, dtype=torch.int32)

    def select(layer, lo, n, logits, selected, zeros):
        length = visible[lo : lo + n]
        cached_pool_logits(
            q[layer, lo : lo + n],
            weights[layer, lo : lo + n],
            caches[layer],
            table,
            requests[lo : lo + n],
            length,
            logits,
        )
        counts = torch.div(length, 4, rounding_mode="floor").to(torch.int32)
        ops.top_k_per_row_prefill(
            logits,
            zeros,
            counts,
            selected,
            n,
            logits.stride(0),
            logits.stride(1),
            ksel,
        )

    def baseline():
        for layer in range(layers):
            select(layer, 0, rows, full_logits, full_selected[layer], full_zeros)

    def candidate():
        for layer in range(layers):
            select(
                layer,
                start,
                local_rows,
                local_logits,
                local_selected[layer],
                local_zeros,
            )
            if shard:
                dist.all_gather_into_tensor(gathered[layer], local_selected[layer])

    # A fixed iteration count is essential: independent time-budget loops
    # could issue different numbers of collectives and deadlock the ranks.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            baseline()
            candidate()
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    dist.barrier()
    graphs = {}
    for name, fn in (("replicated", baseline), ("row_sharded", candidate)):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        graphs[name] = graph
        dist.barrier()

    # Dynamic values under unchanged graph shapes: every rank sees the same
    # CPU-generated query and lengths. Compare valid logits exactly, and
    # compare selected sets because native top-k output order is unspecified.
    for replay in range(4):
        q.copy_(random(q.shape, q.dtype))
        weights.copy_(random(weights.shape, weights.dtype) * 0.1)
        lengths_cpu[:rows] = context
        if replay % 2:
            lengths_cpu[:rows:2] = max(4, context // 2)
        visible.copy_(lengths_cpu)
        graphs["replicated"].replay()
        graphs["row_sharded"].replay()
        actual = gathered[:, :rows] if shard else local_selected
        torch.testing.assert_close(
            actual.sort(dim=-1).values,
            full_selected.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )
        # Logits workspace is deliberately shared across layers, just as in
        # serving. Check every layer separately outside the timed graph.
        for layer in range(layers):
            select(layer, 0, rows, full_logits, full_selected[layer], full_zeros)
            select(
                layer,
                start,
                local_rows,
                local_logits,
                local_selected[layer],
                local_zeros,
            )
            for local_row, global_row in enumerate(range(start, stop)):
                count = int(lengths_cpu[global_row]) // 4
                torch.testing.assert_close(
                    local_logits[local_row, :count],
                    full_logits[global_row, :count],
                    rtol=0,
                    atol=0,
                )
        dist.barrier()

    lengths_cpu[:rows] = context
    visible.copy_(lengths_cpu)
    timing = {name: [] for name in graphs}
    for repeat in range(repeats):
        order = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in order:
            dist.barrier()
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(steps):
                graphs[name].replay()
            end.record()
            end.synchronize()
            duration = torch.tensor(
                begin.elapsed_time(end) * 1000 / steps / layers, device=device
            )
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            timing[name].append(duration.item())
    return {
        "rows": rows,
        "context": context,
        "world_size": world,
        "sharding_active": shard,
        "layers": layers,
        "disjoint_cache_bytes_per_rank": cache_bank.numel() * cache_bank.element_size(),
        "pooled_key_working_bytes_per_rank": layers * (context // 4) * 128 * 2,
        "cache_residency_caveat": "Short contexts may be L2-hot; serving A/B required.",
        "selected_gather_bytes_per_layer": padded_rows * ksel * 4 if shard else 0,
        "exact_valid_logits": True,
        "exact_selected_sets": True,
        "changed_input_graph_replays": 4,
        "max_rank_us_per_layer": timing,
        "median_us": {
            name: statistics.median(values) for name, values in timing.items()
        },
        "serving_speedup_claim": False,
    }


def main():
    import torch
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument(
        "--contexts", type=int, nargs="+", default=[1024, 16384, 131072]
    )
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if min(args.rows + args.contexts + [args.layers, args.repeats, args.steps]) < 1:
        parser.error("all sizes and iteration counts must be positive")
    if max(args.contexts) > 1048576:
        parser.error("context exceeds the retained 1M workspace")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    try:
        for context in args.contexts:
            for rows in args.rows:
                result = measure(
                    rows,
                    context,
                    layers=args.layers,
                    repeats=args.repeats,
                    steps=args.steps,
                )
                if dist.get_rank() == 0:
                    print(json.dumps(result), flush=True)
                dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
