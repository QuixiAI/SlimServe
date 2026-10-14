# SPDX-License-Identifier: Apache-2.0
"""Isolated unchanged KDA kernel with and without the serving output copy.

Run only when serving has released the GPUs. No serving code is imported
from this module. Exact state/output checks accompany graph timing.
"""

import json

import torch
from triton.testing import do_bench_cudagraph

from benchmarks.glm5_next_kda_direct_output import direct_output
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)


@torch.no_grad()
def measure(rows):
    torch.manual_seed(9900 + rows)
    heads, d, layers = 8, 128, 16
    packed = torch.randn(rows, 3 * heads * d, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(1, rows, heads, d, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(1, rows, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads * d, device="cuda") * 0.1
    indices = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    initial = torch.randn(layers, rows + 1, heads, d, d, device="cuda")
    reference_state, direct_state = initial.clone(), initial.clone()
    ref_out = torch.empty(
        layers, 1, rows, heads, d, device="cuda", dtype=torch.bfloat16
    )
    direct_out = torch.empty_like(ref_out)

    def baseline():
        for layer in range(layers):
            result, _ = fused_recurrent_kda_packed_decode(
                packed, gate, beta, a_log, bias, -5.0, reference_state[layer], indices
            )
            ref_out[layer].copy_(result)

    def candidate():
        for layer in range(layers):
            direct_output(
                packed,
                gate,
                beta,
                a_log,
                bias,
                -5.0,
                direct_state[layer],
                indices,
                direct_out[layer],
            )

    baseline()
    candidate()
    torch.cuda.synchronize()
    reference_graph, direct_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(reference_graph):
        baseline()
    with torch.cuda.graph(direct_graph):
        candidate()
    reference_state.copy_(initial)
    direct_state.copy_(initial)
    for _ in range(8):
        packed.normal_()
        gate.normal_()
        beta.normal_()
        reference_graph.replay()
        direct_graph.replay()
        torch.testing.assert_close(direct_out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(direct_state, reference_state, rtol=0, atol=0)
    times = {"allocate_copy": [], "direct_output": []}
    arms = {"allocate_copy": baseline, "direct_output": candidate}
    for repeat in range(5):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            times[name].append(1000 * do_bench_cudagraph(arms[name], rep=100) / layers)
    return dict(
        rows=rows,
        heads=heads,
        layers=layers,
        state_bytes=initial.numel() * initial.element_size(),
        changed_input_replays=8,
        output_and_state_bit_exact=True,
        us=times,
    )


if __name__ == "__main__":
    for rows in (1, 8, 16, 32):
        print(json.dumps(measure(rows)), flush=True)
