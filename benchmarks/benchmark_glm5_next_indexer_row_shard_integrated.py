# SPDX-License-Identifier: Apache-2.0
"""Compare serving and quarantined gather paths with live TP groups/graphs.

Includes native score/top-k, rank-order gather and pool-to-token expansion.
Still excludes query projections, cache updates and model serving. Run alone
on eight GPUs; no monkeypatched communicator or Python context-length gate.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

os.environ["VLLM_LOGGING_STREAM"] = "ext://sys.stderr"

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.glm5_next_indexer import _pooled_select


@torch.no_grad()
def measure(rows, context, selector=_pooled_select, gather="serving"):
    rank = dist.get_rank()
    cpu_group = get_tp_group().cpu_group

    def barrier():
        # Do not initialize a second GPU communicator merely for benchmark
        # coordination; that would hide its memory cost in the candidate.
        dist.barrier(group=cpu_group)

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        return dict(free_bytes=free, total_bytes=total,
                    allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved())

    memory_snapshots = {"before_case": memory()}
    if rank == 0:
        print(json.dumps(dict(event="case_start", rows=rows, context=context)),
              file=sys.stderr, flush=True)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    layers, bs, width, max_pools = 11, 576, 2080, 262144
    pages = (context + bs - 1) // bs
    generator = torch.Generator().manual_seed(59000 + rows + context)

    def random(shape, dtype):
        return torch.randn(shape, dtype=dtype, generator=generator).to(device)

    cache = random((layers, pages, 2, bs, 64), torch.bfloat16)
    q = random((layers, rows, 32, 128), torch.bfloat16)
    weights = random((layers, rows, 32), torch.float32) * 0.1
    ape = torch.zeros(4, 128, device=device)
    table = torch.stack((torch.arange(pages), torch.arange(pages).flip(0))).to(
        device=device, dtype=torch.int32
    )
    requests = (torch.arange(rows, device=device, dtype=torch.int32) * 3 + 1) % 2
    visible = torch.full((rows,), context, dtype=torch.int32, device=device)
    logits = torch.empty(rows, max_pools, device=device)
    backing = {name: torch.full((layers, rows * width + 64), -77,
                               dtype=torch.int32, device=device)
               for name in ("replicated", "row_sharded")}
    outputs = {name: [b[32:-32].view(rows, width) for b in buf]
               for name, buf in backing.items()}
    enabled = rows in (16, 32)
    memory_snapshots["before_warmup"] = memory()

    def run(name):
        for layer in range(layers):
            selector(
                q[layer], weights[layer], ape, cache[layer, :, 1], table,
                requests, visible, logits, max_pools, bs, 128**-0.5, 512,
                outputs[name][layer], 4,
                row_shard=enabled and name == "row_sharded",
            )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run("replicated")
            run("row_sharded")
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    barrier()
    memory_snapshots["after_warmup"] = memory()
    graphs = {}
    for name in outputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run(name)
        graphs[name] = graph
        barrier()
    memory_snapshots["after_capture"] = memory()

    for replay in range(5):
        q.copy_(random(q.shape, q.dtype))
        weights.copy_(random(weights.shape, weights.dtype) * 0.1)
        table.copy_(torch.stack((torch.arange(pages), torch.arange(pages).flip(0)))
                    .to(device=device, dtype=torch.int32).flip(replay % 2))
        requests.copy_((torch.arange(rows, device=device).int() + replay) % 2)
        lengths = [context - 3 if replay % 2 else context] * rows
        if replay == 1:
            lengths = [min(context, (0, 1, 7, 1023)[i % 4]) for i in range(rows)]
        elif replay == 2:
            lengths[rows // 2 + 1:] = [0] * (rows - (rows // 2 + 1))
        elif replay == 4:
            lengths = [0] * rows
        visible.copy_(torch.tensor(lengths, dtype=torch.int32, device=device))
        for arrays in outputs.values():
            for out in arrays:
                out.fill_(-99)
        graphs["replicated"].replay()
        graphs["row_sharded"].replay()
        for actual, expected in zip(outputs["row_sharded"], outputs["replicated"]):
            torch.testing.assert_close(actual.sort().values, expected.sort().values,
                                       atol=0, rtol=0)
            valid = (actual == -1) | ((actual >= 0) & (actual < visible[:, None]))
            assert valid.all()
        assert all((buf[:, :32] == -77).all() and (buf[:, -32:] == -77).all()
                   for buf in backing.values())
        barrier()
    visible.fill_(context)
    timing = {name: [] for name in graphs}
    for repeat in range(5):
        for name in (list(graphs) if repeat % 2 == 0 else list(reversed(graphs))):
            barrier()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graphs[name].replay()
            end.record()
            end.synchronize()
            duration = torch.tensor(start.elapsed_time(end) * 1000 / 50 / layers)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX, group=cpu_group)
            timing[name].append(duration.item())
    for actual, expected in zip(outputs["row_sharded"], outputs["replicated"]):
        torch.testing.assert_close(actual.sort().values, expected.sort().values,
                                   atol=0, rtol=0)
    assert all((buf[:, :32] == -77).all() and (buf[:, -32:] == -77).all()
               for buf in backing.values())
    memory_by_rank = [None] * get_tp_group().world_size
    dist.all_gather_object(memory_by_rank, memory_snapshots, group=cpu_group)
    if rank == 0:
        print(json.dumps(dict(
            rows=rows, context=context, layers=layers, sharding_active=enabled,
            world_size=get_tp_group().world_size, changed_graph_replays=5,
            exact_expanded_sets=True, output_guards=True,
            group="live vLLM TP group", gather=gather,
            memory_by_rank=memory_by_rank,
            max_rank_us_per_layer=timing,
            median_us={name: statistics.median(values)
                       for name, values in timing.items()},
            scope="Serving selector only, including expansion; not model TPS",
        )), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gather",
                        choices=("serving", "process-group", "existing-pynccl"),
                        default="serving")
    args = parser.parse_args()
    selector = _pooled_select
    if args.gather != "serving":
        from benchmarks.glm5_next_indexer_gather_candidate import (
            pooled_select_existing_tp,
            pooled_select_process_group,
        )

        selector = (pooled_select_existing_tp if args.gather == "existing-pynccl"
                    else pooled_select_process_group)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    assert int(os.environ["WORLD_SIZE"]) == 8
    torch.cuda.set_device(local_rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=8))
    with set_current_vllm_config(config):
        init_distributed_environment(8, rank, "env://", local_rank,
                                     timeout=timedelta(minutes=5))
        initialize_model_parallel(8)
        try:
            for context in (1024, 131072):
                for rows in (1, 8, 16, 32):
                    measure(rows, context, selector, args.gather)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == "__main__":
    main()
