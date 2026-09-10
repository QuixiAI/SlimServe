# SPDX-License-Identifier: Apache-2.0
"""Exclusive-GPU gate for precomputed KDA chunk metadata (not serving TPS).

Each timed step has fresh cu_seqlens objects shared across several layer
calls, reflecting per-step metadata construction and within-step reuse. Do
not time just a permanent cu_seqlens object: that measures a cache hit and
hides the GPU-to-CPU synchronization under investigation. Both arms clone V
because the unchanged KDA kernel can overwrite that input with its output.
"""

import argparse
import json
import statistics
import time

import torch

from benchmarks.glm5_next_kda_metadata_candidate import chunk_kda_precomputed
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices


@torch.no_grad()
def measure(lengths, layers=34, groups=4, steps=5, repeats=3):
    torch.manual_seed(9910 + sum(lengths))
    heads, dim = 8, 128
    tokens = sum(lengths)
    cpu_cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32
    )
    cu = cpu_cu.cuda()
    chunks = prepare_chunk_indices(cpu_cu, 64).cuda()
    q, k, original_v, raw = (
        torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    )
    beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads * dim, device="cuda") * 0.1
    initial = torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.1
    initial_before = initial.clone()

    def call(use_precomputed, sequence_starts):
        kwargs = dict(
            q=q,
            k=k,
            v=original_v.clone(),
            raw_g=raw,
            raw_beta=beta,
            A_log=a_log,
            g_bias=bias,
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            lower_bound=-5.0,
            cu_seqlens=sequence_starts,
        )
        if use_precomputed:
            return chunk_kda_precomputed(**kwargs, chunk_indices=chunks)
        return chunk_kda_with_fused_gate(**kwargs)

    for _ in range(4):
        q.normal_()
        k.normal_()
        original_v.normal_()
        raw.normal_()
        beta.normal_()
        sequence_starts = cu.clone()
        expected, expected_state = call(False, sequence_starts)
        actual, actual_state = call(True, sequence_starts)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
        torch.testing.assert_close(initial, initial_before, atol=0, rtol=0)

    times = {"rebuild_from_gpu_lengths": [], "reuse_cpu_prepared_table": []}
    for repeat in range(repeats):
        order = [False, True] if repeat % 2 == 0 else [True, False]
        for precomputed in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(steps):
                sequence_starts = [cu.clone() for _ in range(groups)]
                for layer in range(layers):
                    call(precomputed, sequence_starts[layer % groups])
            torch.cuda.synchronize()
            name = (
                "reuse_cpu_prepared_table"
                if precomputed
                else "rebuild_from_gpu_lengths"
            )
            times[name].append((time.perf_counter() - start) * 1e6 / steps)
    return dict(
        lengths=lengths,
        heads=heads,
        dim=dim,
        layers=layers,
        groups=groups,
        steps=steps,
        changed_input_checks=4,
        exact_output_and_state=True,
        metadata_cpu_preparation_excluded_both_arms=True,
        wall_us_per_step=times,
        median_wall_us_per_step={k: statistics.median(v) for k, v in times.items()},
        scope="Synthetic KDA pipeline only; not graph or serving qualification",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layers", type=int, default=34)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.layers, args.groups, args.steps, args.repeats) < 1:
        p.error("positive sizes required")
    for lengths in (
        [1],
        [1] * 8,
        [1] * 16,
        [1] * 32,
        [63, 64, 65, 127, 128, 129],
        [1024],
        [8192],
    ):
        print(json.dumps(measure(lengths, **vars(args))), flush=True)


if __name__ == "__main__":
    main()
