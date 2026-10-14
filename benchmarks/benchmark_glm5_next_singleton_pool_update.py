# SPDX-License-Identifier: Apache-2.0
"""Byte-exact, alternating CUDA-graph timing of c1 compact-cache update.

Run as a module from the repo root after the serving engine releases GPUs:
  python -m benchmarks.benchmark_glm5_next_singleton_pool_update
"""

import json

import torch
from triton.testing import do_bench_cudagraph

from vllm.model_executor.layers.glm5_next_pool_cache import update_pool_cache


@torch.no_grad()
def measure(bs, phase):
    torch.manual_seed(9500 + bs + phase)
    storage = torch.randn(3, 2, bs, 64, device="cuda", dtype=torch.bfloat16)
    alternative = storage.clone()
    old, new = storage[:, 0], alternative[:, 0]
    packed = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([bs + 8 + phase], device="cuda", dtype=torch.int64)
    ape = torch.randn(4, 128, device="cuda")

    def baseline():
        update_pool_cache(packed, slots, ape, old)

    def candidate():
        update_pool_cache(packed, slots, ape, new, singleton_fused=True)

    baseline()
    candidate()
    assert torch.equal(storage, alternative)
    samples = {"baseline": [], "fused": []}
    arms = {"baseline": baseline, "fused": candidate}
    for repeat in range(5):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            samples[name].append(1000 * do_bench_cudagraph(arms[name], rep=100))
    assert torch.equal(storage, alternative)
    return dict(block_size=bs, phase=phase, cache_exact=True, us=samples)


if __name__ == "__main__":
    for bs in (64, 4608):
        for phase in range(4):
            print(json.dumps(measure(bs, phase)), flush=True)
