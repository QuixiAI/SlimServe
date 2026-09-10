# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: remove general expert alignment for one non-EP token.

Top-k contains distinct experts. Each expert therefore has exactly one
real row and B-1 padding rows; sorting experts is unnecessary for Marlin.
Keep immutable row/padding metadata and view the router's current IDs.
No serving import, global cache, weight repack or new GEMM is introduced.
"""

import argparse
import functools
import json
import statistics

import torch

from vllm.model_executor.layers.fused_moe.singleton_alignment import SingletonAlignment


@torch.no_grad()
def measure(experts=288, intermediate=256):
    from triton.testing import do_bench_cudagraph

    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
        fused_marlin_moe,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        nvfp4_marlin_process_global_scale,
        nvfp4_marlin_process_scales,
    )
    from vllm.scalar_type import scalar_types

    torch.manual_seed(8053)
    hidden, topk, block = 4096, 8, 8
    x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randperm(experts, device="cuda", dtype=torch.int32)[:topk].view(1, -1)
    weights = torch.rand(1, topk, device="cuda")
    weights.mul_(2.5 / weights.sum())
    owner = SingletonAlignment(topk, block, x.device)

    def packed(k, n):
        # Random native-layout nibbles, not a dequantized weight cache.
        w = torch.randint(
            -(2**31),
            2**31 - 1,
            (experts, k // 16, n * 2),
            device="cuda",
            dtype=torch.int32,
        )
        scales = marlin_permute_scales(
            torch.ones(k // 16, n, device="cuda", dtype=torch.float16), k, n, 16
        )
        scales, _ = nvfp4_marlin_process_scales(
            scales, scale_factor=1.0, a_dtype=torch.bfloat16
        )
        scales = scales.unsqueeze(0).expand(experts, -1, -1).contiguous()
        global_scale = nvfp4_marlin_process_global_scale(
            torch.full((experts,), 0.01, device="cuda"), torch.bfloat16
        )
        return w, scales, global_scale

    w1, s1, g1 = packed(hidden, 2 * intermediate)
    w2, s2, g2 = packed(intermediate, hidden)
    workspace = marlin_make_workspace_new(x.device, 4)
    scratch13 = torch.empty(
        topk * max(hidden, 2 * intermediate), device=x.device, dtype=x.dtype
    )
    scratch2 = torch.empty(topk, intermediate, device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)

    def call(fast, routed_ids=None):
        routed_ids = ids if routed_ids is None else routed_ids
        return fused_marlin_moe(
            hidden_states=x,
            w1=w1,
            w2=w2,
            bias1=None,
            bias2=None,
            w1_scale=s1,
            w2_scale=s2,
            topk_weights=weights,
            topk_ids=routed_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_scale1=g1,
            global_scale2=g2,
            workspace=workspace,
            intermediate_cache13=scratch13,
            intermediate_cache2=scratch2,
            output=output,
            singleton_alignment=owner if fast else None,
        )

    expected = call(False).clone()
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected) > 0
    torch.testing.assert_close(call(True), expected, atol=0, rtol=0)
    for _ in range(3):
        call(True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = call(True)
    for _ in range(16):
        x.normal_()
        ids.copy_(
            torch.randperm(experts, device="cuda", dtype=torch.int32)[:topk].view(1, -1)
        )
        weights.uniform_()
        weights.mul_(2.5 / weights.sum())
        graph.replay()
        snapshot = actual.clone()
        torch.testing.assert_close(snapshot, call(False), atol=0, rtol=0)
    # A single repeated route fits its weights in A100 L2 and exaggerates
    # launch savings relative to a real multi-layer model. Cycle disjoint
    # expert sets within each timed graph: default 192 MiB of packed weights.
    route_count = min(16, experts // topk)
    route_bank = torch.randperm(experts, device="cuda", dtype=torch.int32)[
        : route_count * topk
    ].view(route_count, 1, topk)
    routes = list(route_bank.unbind(0))

    def cycle(fast):
        for route in routes:
            call(fast, route)

    arms = {
        "generic_alignment": functools.partial(cycle, False),
        "singleton_metadata": functools.partial(cycle, True),
    }
    timings = {name: [] for name in arms}
    for repeat in range(5):
        for name in list(arms) if repeat % 2 == 0 else list(reversed(arms)):
            timings[name].append(
                1000 * do_bench_cudagraph(arms[name], rep=100) / route_count
            )
    return dict(
        tokens=1,
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
        topk=topk,
        exact_parity=True,
        changed_input_replays=16,
        disjoint_routes=route_count,
        timed_packed_weight_bytes=(w1.nbytes + w2.nbytes)
        * route_count
        * topk
        // experts,
        us=timings,
        medians_us={name: statistics.median(t) for name, t in timings.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--intermediate", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(measure(args.experts, args.intermediate)), flush=True)
