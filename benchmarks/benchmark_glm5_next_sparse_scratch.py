# SPDX-License-Identifier: Apache-2.0
"""Isolated sparse-TC allocation experiment; run each arm in a fresh process.

The CUDA graph allocator may already reuse private temporary buffers. Measure
that control instead of claiming eleven layers imply eleven live scratch sets.
This is not serving KV capacity: real-profile allocation and TPS must follow.
"""

import argparse
import json
import statistics

import torch

from benchmarks.glm5_next_sparse_scratch_candidate import (
    SparseTCWorkspace,
)
from benchmarks.glm5_next_sparse_scratch_candidate import (
    shared_sparse_tc_nope as sparse_tc_nope,
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(58032)
    device = torch.device("cuda", torch.cuda.current_device())
    layers, max_rows, width, pages, bs = 11, 32, 2080, 8, 576
    cache = torch.randn(layers, pages, 2, bs, 512, device=device, dtype=torch.bfloat16)
    q = torch.randn(layers, max_rows, 8, 512, device=device, dtype=torch.bfloat16) * 0.2
    table = torch.arange(pages, device=device, dtype=torch.int32).repeat(max_rows, 1)
    indices = torch.arange(width, device=device, dtype=torch.int32).repeat(max_rows, 1)
    lengths = torch.full((max_rows,), width, device=device, dtype=torch.int32)
    observed = torch.empty_like(q)
    before_owner = torch.cuda.memory_allocated()
    owner = SparseTCWorkspace(max_rows, width, 8, device) if args.shared else None

    def run(rows, workspace):
        for layer in range(layers):
            result = sparse_tc_nope(
                q[layer, :rows], cache[layer, :, 1], table[:rows], indices[:rows],
                lengths[:rows], 1 / 16, split=32 if rows < 8 else 128,
                workspace=workspace,
            )
            observed[layer, :rows].copy_(result)

    for rows in (32, 16, 8, 1):
        run(rows, owner)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before_graph = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for rows in (32, 16, 8, 1):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            run(rows, owner)
        graphs[rows] = graph
    torch.cuda.synchronize()
    memory = dict(
        before_owner_allocated=before_owner,
        before_graph_allocated=before_graph,
        after_graph_allocated=torch.cuda.memory_allocated(),
        after_graph_reserved=torch.cuda.memory_reserved(),
        capture_peak_allocated=torch.cuda.max_memory_allocated(),
        explicit_workspace_bytes=0 if owner is None else owner._storage.numel() * 4,
    )
    # Same fixed-address owner across graph shapes; all outputs are observed
    # before a later layer or a differently-shaped graph can overwrite them.
    for replay in range(4):
        q.normal_(std=0.2)
        lengths.fill_((0, 37, 2048, 2080)[replay])
        indices[:, 5::7] = -1
        for rows in (1, 32, 8, 16):
            observed.fill_(float("nan"))
            if owner is not None:
                owner._storage.fill_(float("nan"))
            graphs[rows].replay()
            actual = observed[:, :rows].clone()
            run(rows, None)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, observed[:, :rows], atol=0, rtol=0)
    lengths.fill_(width)
    timings = {rows: [] for rows in graphs}
    for repeat in range(5):
        for rows in (list(graphs) if repeat % 2 == 0 else list(reversed(graphs))):
            graph = graphs[rows]
            for _ in range(10):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            timings[rows].append(start.elapsed_time(end) * 1000 / 100 / layers)
    print(json.dumps(dict(
        shared=args.shared, layers=layers, memory=memory, changed_replay_exact=True,
        per_layer_us={rows: dict(repeats=values, median=statistics.median(values))
                      for rows, values in timings.items()},
        scope="Isolated shared CUDA graph pool; not serving capacity or throughput",
    ), indent=2))


if __name__ == "__main__":
    main()
