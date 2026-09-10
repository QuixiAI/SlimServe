# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse, large-vocabulary sampling must be invariant to logit offsets."""

from types import SimpleNamespace

import pytest
import torch

from vllm.quixicore import quixicore_ops
from vllm.v1.worker.gpu.sample.gumbel import stateless_uniform_2d

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


def test_sparse_rejection_sampling_matches_cpu_under_large_logit_offsets():
    requests, vocab, steps = 32, 129280, 5
    support = torch.tensor([7, 17000, 65539, 90001, vocab - 1])
    values = torch.tensor([-0.25, -0.5, -0.75, -1.0, -1.25])
    seeds = torch.arange(42, 42 + requests, dtype=torch.int64)
    positions = torch.full((requests,), 1333, dtype=torch.int64)
    u = stateless_uniform_2d(seeds, positions ^ (1 << 41), 1).squeeze(1)
    expected = support[torch.searchsorted(values.softmax(0).cumsum(0), u)]
    cu = torch.arange(requests + 1, dtype=torch.int32, device="mps")
    mapping = torch.randperm(requests, generator=torch.Generator().manual_seed(9))
    state_seeds = torch.empty_like(seeds)
    state_seeds[mapping] = seeds
    args = (
        None,
        torch.zeros(requests, dtype=torch.int32, device="mps"),
        cu,
        positions.to("mps"),
        mapping.to(dtype=torch.int32, device="mps"),
        torch.ones(requests, device="mps"),
        state_seeds.to("mps"),
        steps,
        vocab,
    )
    for offset in (0.0, 20000.0, -20000.0):
        logits = torch.full((requests, vocab), -torch.inf)
        logits[:, support] = values + offset
        gpu_logits = logits.to("mps")
        for _ in range(8):
            sampled, counts = quixicore_ops.qwen38_rejection_sample(gpu_logits, *args)
            torch.testing.assert_close(
                counts.cpu(), torch.ones(requests, dtype=torch.int32)
            )
            torch.testing.assert_close(sampled[:, 0].cpu(), expected)
            assert (sampled[:, 1:].cpu() == -1).all()


def test_seeded_first_token_uses_same_sampler_alone_and_in_mixed_batch():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        _rejection_sample_mps,
    )

    device = "mps"
    seed = torch.tensor([42, 81], dtype=torch.int64, device=device)
    temp = torch.ones(2, device=device)

    def plain(logits, batch):
        tokens = gumbel_sample(
            logits, batch.mapping, temp, seed, batch.positions, apply_temperature=False
        )
        return SimpleNamespace(
            sampled_token_ids=tokens[:, None], num_sampled=None, num_rejected=None
        )

    def verifier(logits, batch, draft_logits):
        tokens, counts = _rejection_sample_mps(
            logits,
            draft_logits,
            torch.zeros(len(logits), dtype=torch.int32, device=device),
            batch.cu,
            batch.positions,
            batch.mapping,
            temp,
            seed,
            5,
            logits.shape[1],
        )
        return SimpleNamespace(
            sampled_token_ids=tokens, num_sampled=counts, num_rejected=None
        )

    runner = SimpleNamespace(
        model=SimpleNamespace(compute_logits=lambda x: x),
        _nan_watch=lambda *a: None,
        thinking_budget_state=None,
        sampler=plain,
        rejection_sampler=verifier,
        speculator=SimpleNamespace(draft_logits=None),
    )
    # The first request has no draft tokens in either batch. Its neighbor
    # contributes five, which formerly switched the first request's RNG.
    first = torch.linspace(-1, 1, 1024, device=device)[None, :]
    outputs = []
    for mixed in (False, True):
        rows = 7 if mixed else 1
        batch = SimpleNamespace(
            num_draft_tokens=5 if mixed else 0,
            logits_indices=torch.arange(rows, device=device),
            mapping=torch.arange(2 if mixed else 1, dtype=torch.int32, device=device),
            cu=torch.tensor(
                [0, 1, 7] if mixed else [0, 1], dtype=torch.int32, device=device
            ),
            positions=torch.tensor(
                [1333] + list(range(90, 96)) if mixed else [1333],
                dtype=torch.int64,
                device=device,
            ),
        )
        logits = first.expand(rows, -1).contiguous()
        output, _, _ = GPUModelRunner.sample(runner, logits, batch, None)
        outputs.append(output.sampled_token_ids[0, 0].item())
    assert outputs[0] == outputs[1]
