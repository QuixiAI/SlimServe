# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling constraints retain CPU semantics across expanded Metal rows."""

import pytest
import torch

from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


@pytest.mark.parametrize("vocab", [17, 8197])
def test_logit_bias_matches_cpu_with_expansion_padding_and_minimum_boundary(vocab):
    torch.manual_seed(123)
    raw = torch.randn(5, vocab + 13)
    expected = raw.clone()
    gpu_raw = raw.to("mps")
    # Two speculative positions for request 2, one on either side of min_len.
    mapping = torch.tensor([2, 0, 2, -1, 1], dtype=torch.int64)
    positions = torch.tensor([10, 9, 11, 5, 2], dtype=torch.int32)
    allowed_count = torch.tensor([3, 0, 2], dtype=torch.int32)
    allowed_ids = torch.tensor([[1, 3, 5], [0, 0, 0], [2, 7, 0]], dtype=torch.int32)
    bias_count = torch.tensor([2, 1, 2], dtype=torch.int32)
    bias_ids = torch.tensor([[1, 4], [8, 0], [2, 7]], dtype=torch.int32)
    biases = torch.tensor([[3.0, 4.0], [-2.0, 0.0], [1.0, -7.0]])
    minimum = torch.tensor([11, 0, 12], dtype=torch.int32)
    stop_count = torch.tensor([1, 0, 1], dtype=torch.int32)
    stop_ids = torch.tensor([[3], [0], [7]], dtype=torch.int32)
    for row, req in enumerate(mapping.tolist()):
        if req < 0:
            continue
        values = expected[row, :vocab]
        count = int(allowed_count[req])
        if count:
            ids = allowed_ids[req, :count].long()
            original = values[ids].clone()
            values.fill_(-torch.inf)
            values[ids] = original
        for i in range(int(bias_count[req])):
            values[int(bias_ids[req, i])] += biases[req, i]
        if positions[row] + 1 < minimum[req]:
            values[stop_ids[req, : int(stop_count[req])].long()] = -torch.inf
    apply_logit_bias(
        gpu_raw[:, :vocab],
        *[
            t.to("mps")
            for t in (
                mapping,
                positions,
                allowed_count,
                allowed_ids,
                bias_count,
                bias_ids,
                biases,
                minimum,
                stop_count,
                stop_ids,
            )
        ],
    )
    assert torch.equal(gpu_raw.cpu(), expected)


def test_logit_bias_full_allowlist_saves_before_clearing_logits():
    vocab = 2053
    raw = torch.arange(vocab, dtype=torch.float32).repeat(2, 1)
    expected = torch.full_like(raw, -torch.inf)
    ids = torch.arange(0, 2048, 2, dtype=torch.int32)[None, :]
    expected[:, ids[0].long()] = raw[:, ids[0].long()]
    gpu = raw.to("mps")
    apply_logit_bias(
        gpu,
        *[
            t.to("mps")
            for t in (
                torch.tensor([0, 0], dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int64),
                torch.tensor([1024], dtype=torch.int32),
                ids,
                torch.tensor([0], dtype=torch.int32),
                ids,
                ids.float(),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                ids[:, :1],
            )
        ],
    )
    assert torch.equal(gpu.cpu(), expected)
