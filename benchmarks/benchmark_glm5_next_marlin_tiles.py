# SPDX-License-Identifier: Apache-2.0
"""Isolated TP8 NVFP4 MoE GEMM tile sweep; never imported by serving.

Uses the owned Marlin op's existing tuning arguments, native packed weights,
and expert-aligned rows. No weight repack, new kernel, or serving dispatch.
Run only with exclusive GPUs. Finite but non-bit-exact candidates are reported
as diagnostics, NOT as qualified replacements. Full MoE/model gates follow.
"""

import argparse
import json
import statistics
import sys

import torch


def configurations():
    return [(-1, -1, -1)] + [
        (k, n, occupancy)
        for k, n in ((128, 128), (64, 256), (64, 128), (128, 64))
        for occupancy in (1, 2, 3, 4)
    ]


def projection_configurations(leg, include_direct=False):
    if leg not in ("gate_up", "down"):
        raise ValueError("unknown projection")
    extra = [(0, 64, 0), (0, 128, 0)] if leg == "down" and include_direct else []
    return configurations() + extra


def route_bank(rows, layers, pattern, seed=53):
    if rows not in (1, 8, 16, 32) or layers < 1:
        raise ValueError("expected c1/8/16/32 and positive layer count")
    if pattern not in ("disjoint", "shared", "uniform"):
        raise ValueError("unknown route pattern")
    generator = torch.Generator().manual_seed(seed)
    routes = []
    for layer in range(layers):
        if pattern == "uniform":
            ids = torch.stack(
                [torch.randperm(288, generator=generator)[:8] for _ in range(rows)]
            )
        else:
            # Rotate hot experts across layers so c1/shared routes do not
            # turn a whole model step into one tiny L2-resident expert set.
            count = rows * 8 if pattern == "disjoint" else 8
            ids = ((torch.arange(count) + layer * 8 + seed) % 288).view(-1, 8)
            if pattern == "shared":
                ids = ids.expand(rows, 8)
        routes.append(ids)
    return torch.stack(routes).to(torch.int32)


def route_summary(ids):
    counts = torch.bincount(ids.flatten().long(), minlength=288)
    return dict(
        active_experts=int((counts > 0).sum()),
        real_rows=ids.numel(),
        padded_rows=int(((counts + 7) // 8 * 8).sum()),
        max_rows_per_expert=int(counts.max()),
    )


def marlin_partition_plan(
    padded_rows, size_k, size_n, thread_k, thread_n, blocks_per_sm, sms=108
):
    """Mirror native DP + two-tile stream-K scheduling for group-size16.

    Counts are derived from supplied routing, not inferred from concurrency.
    They do not predict elapsed time or imply every allocated scratch byte
    is accessed. Only multi-CTA tail tiles enter global reduction.
    """
    if (
        padded_rows < 0
        or padded_rows % 8
        or min(size_k, size_n, thread_k, thread_n, blocks_per_sm, sms) <= 0
        or size_k % thread_k
        or size_n % thread_n
    ):
        raise ValueError("invalid aligned Marlin geometry")
    total = padded_rows // 8 * (size_n // thread_n)
    grid = sms * blocks_per_sm
    tail, dp_iters = total, 0
    if total > grid:
        tail = total % grid
        if tail * 3 <= grid:
            tail += grid
        dp_iters = (total - tail) // grid
    k_tiles = size_k // thread_k
    iters = (k_tiles * tail + grid - 1) // grid
    parts = [
        ((tile + 1) * k_tiles - 1) // iters - (tile * k_tiles) // iters + 1
        for tile in range(tail)
    ]
    return dict(
        total_tiles=total,
        full_k_dp_tiles=dp_iters * grid,
        tail_tiles=tail,
        multi_cta_tiles=sum(p > 1 for p in parts),
        tail_cta_fragments=sum(parts),
        grid_ctas=grid,
        k_tiles=k_tiles,
        tail_iters_per_cta=iters,
    )


@torch.inference_mode()
def measure(rows, leg, pattern, layers, repeats, bench_ms, include_direct=False):
    from triton.testing import do_bench_cudagraph

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
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

    k, n = (4096, 512) if leg == "gate_up" else (256, 4096)
    routed_rows = rows * 8
    input_rows = rows if leg == "gate_up" else routed_rows
    torch.manual_seed(53000 + rows + k)
    packed = torch.randint(
        -(2**31), 2**31 - 1, (288, k // 16, n * 2), dtype=torch.int32, device="cuda"
    )
    raw_scales = torch.randint(-4, 3, (288 * k // 16, n), device="cuda")
    raw_scales = torch.exp2(raw_scales.float()).half()
    scales, _ = nvfp4_marlin_process_scales(
        marlin_permute_scales(raw_scales, k, n, 16),
        scale_factor=1.0,
        a_dtype=torch.bfloat16,
    )
    scales = scales.view(288, k // 16, n)
    del raw_scales
    global_scale = nvfp4_marlin_process_global_scale(
        torch.full((288,), 0.01, device="cuda"), torch.bfloat16
    )
    workspace = marlin_make_workspace_new(torch.device("cuda", 0), 4)
    x = torch.empty(layers, input_rows, k, device="cuda", dtype=torch.bfloat16)
    ids = route_bank(rows, layers, pattern).cuda()
    weights = torch.empty(layers, rows, 8, device="cuda", dtype=torch.float32)
    alignment = [
        moe_align_block_size(i, 8, 288, ignore_invalid_experts=True) for i in ids
    ]
    backing = torch.full(
        (layers, routed_rows * n + 64), -71.0, device="cuda", dtype=torch.bfloat16
    )
    outputs = [b[32:-32].view(routed_rows, n) for b in backing]

    def refill(replay):
        torch.manual_seed(70000 + rows + replay)
        x.normal_()
        weights.uniform_()
        weights.mul_(2.5 / weights.sum(-1, keepdim=True))
        ids.copy_(route_bank(rows, layers, pattern, seed=53 + replay))
        for layer in range(layers):
            fresh = moe_align_block_size(
                ids[layer], 8, 288, ignore_invalid_experts=True
            )
            for destination, source in zip(alignment[layer], fresh):
                destination.copy_(source)

    def run(config):
        thread_k, thread_n, occupancy = config
        for layer in range(layers):
            sorted_ids, experts, padded = alignment[layer]
            if thread_k == 0:
                from benchmarks.glm5_next_nvfp4_down_candidate import direct_down

                assert leg == "down"
                direct_down(
                    x[layer],
                    packed,
                    scales,
                    global_scale,
                    sorted_ids,
                    experts,
                    padded,
                    weights[layer],
                    out=outputs[layer],
                    tile=thread_n,
                )
                continue
            ops.moe_wna16_marlin_gemm(
                x[layer],
                outputs[layer],
                packed,
                None,
                scales,
                None,
                global_scale,
                None,
                None,
                None,
                workspace,
                sorted_ids,
                experts,
                padded,
                weights[layer],
                moe_block_size=8,
                top_k=8 if leg == "gate_up" else 1,
                mul_topk_weights=leg == "down",
                b_q_type=scalar_types.float4_e2m1f,
                size_m=input_rows,
                size_n=n,
                size_k=k,
                is_k_full=True,
                use_atomic_add=False,
                use_fp32_reduce=True,
                is_zp_float=False,
                thread_k=thread_k,
                thread_n=thread_n,
                blocks_per_sm=occupancy,
            )

    automatic = configurations()[0]
    valid, records = [], {}
    for config in projection_configurations(leg, include_direct):
        print(
            json.dumps(
                dict(
                    event="arm_start",
                    rows=rows,
                    leg=leg,
                    pattern=pattern,
                    config=config,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        refill(0)
        try:
            run(config)
            torch.cuda.synchronize()
        except RuntimeError as error:
            # Only explicit host-side launch-configuration rejection is
            # skippable. CUDA faults and unexpected errors abort the sweep.
            if not any(
                s in str(error)
                for s in ("Invalid thread config:", "Unsupported shapes:")
            ):
                raise
            records[str(config)] = dict(status="unsupported", error=str(error))
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(config)
        exact, maximum = True, 0.0
        for replay in range(4):
            refill(replay)
            run(automatic)
            expected = [o.clone() for o in outputs]
            # Restore this arm's CUDA function attributes before replay;
            # different occupancy choices alter shared-memory allowances.
            run(config)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            for output, reference in zip(outputs, expected):
                assert torch.isfinite(output).all() and torch.isfinite(reference).all()
                assert torch.count_nonzero(reference) > 0
                exact = exact and torch.equal(output, reference)
                maximum = max(
                    maximum, (output.float() - reference.float()).abs().max().item()
                )
            assert (backing[:, :32] == -71).all() and (backing[:, -32:] == -71).all()
        if config == automatic:
            assert exact, "automatic baseline was not bit-exact across replay"
        records[str(config)] = dict(
            status="measured", bit_exact=exact, max_absolute_error=maximum, us=[]
        )
        if config[0] > 0:
            records[str(config)]["synthetic_first_layer_scheduler"] = (
                marlin_partition_plan(
                    route_summary(route_bank(rows, layers, pattern)[0])["padded_rows"],
                    k,
                    n,
                    *config,
                )
            )
        print(
            json.dumps(
                dict(
                    event="arm_replay_result",
                    rows=rows,
                    leg=leg,
                    pattern=pattern,
                    config=config,
                    bit_exact=exact,
                    max_absolute_error=maximum,
                )
            ),
            file=sys.stderr,
            flush=True,
        )
        valid.append(config)
        del graph
    refill(0)
    for repeat in range(repeats):
        for config in valid if repeat % 2 == 0 else reversed(valid):
            # Fresh capture per arm restores its function attributes. Route
            # alignment/preparation is excluded equally from every arm.
            us = (
                do_bench_cudagraph(lambda cfg=config: run(cfg), rep=bench_ms)
                * 1000
                / layers
            )
            records[str(config)]["us"].append(us)
            snapshot = [output.clone() for output in outputs]
            run(config)
            for output, saved in zip(outputs, snapshot):
                torch.testing.assert_close(output, saved, atol=0, rtol=0)
            assert (backing[:, :32] == -71).all() and (backing[:, -32:] == -71).all()
            print(
                json.dumps(
                    dict(
                        event="timed_replay_stable",
                        rows=rows,
                        leg=leg,
                        pattern=pattern,
                        config=config,
                        repeat=repeat,
                        microseconds=us,
                    )
                ),
                file=sys.stderr,
                flush=True,
            )
    for record in records.values():
        if record["status"] == "measured":
            record["median_us"] = statistics.median(record["us"])
    return dict(
        rows=rows,
        leg=leg,
        size_k=k,
        size_n=n,
        pattern=pattern,
        layers=layers,
        repeats=repeats,
        bench_ms=bench_ms,
        changed_replays=4,
        timed_replay_stability_checked=True,
        route_summary=route_summary(route_bank(rows, layers, pattern)[0]),
        route_distribution="synthetic; not captured model routing",
        touched_packed_weight_and_scale_bytes=(packed.nbytes + scales.nbytes)
        * route_bank(rows, layers, pattern).unique().numel()
        // 288,
        weight_bank_bytes=packed.nbytes + scales.nbytes + global_scale.nbytes,
        arms=records,
        direct_down_enabled=include_direct and leg == "down",
        config_encoding=(
            "(Ktile,Ntile,CTAs/SM); (-1,-1,-1)=auto, (0,Ntile,0)=direct down"
        ),
        scope="One MoE GEMM only; excludes alignment, SwiGLU, reduction and serving",
        quality_gate="Bit-exact flag is explicit; non-bit-exact is NOT qualified",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 16, 32], choices=[1, 8, 16, 32]
    )
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--bench-ms", type=int, default=100)
    parser.add_argument(
        "--direct-down",
        action="store_true",
        help="also gate/time the quarantined fixed-K down kernels",
    )
    args = parser.parse_args()
    if min(args.layers, args.repeats, args.bench_ms) < 1:
        parser.error("positive benchmark sizes required")
    for rows in args.rows:
        for pattern in ("disjoint", "shared", "uniform"):
            for leg in ("gate_up", "down"):
                print(
                    json.dumps(
                        measure(
                            rows,
                            leg,
                            pattern,
                            args.layers,
                            args.repeats,
                            args.bench_ms,
                            args.direct_down,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
