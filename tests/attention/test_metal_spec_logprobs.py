# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative logprobs map accepted tokens to the correct ragged rows."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


@pytest.mark.parametrize("logits_mode", [False, True])
def test_ragged_accepted_logprobs_match_cpu_with_padded_rows(logits_mode):
    sampler = RejectionSampler.__new__(RejectionSampler)
    sampler.sampler = SimpleNamespace(
        logprobs_mode="raw_logits" if logits_mode else "raw_logprobs"
    )
    torch.manual_seed(7)
    raw = torch.randn(8, 44)
    logits = raw[:, :37]
    samples = torch.tensor([[11, 22, -1, -1, 90], [-1] * 5, [3, 4, 5, -1, 90]])
    gpu_samples = samples.to("mps")[:, :4]
    counts = torch.tensor([2, 0, 3], dtype=torch.int32, device="mps")
    offsets = np.array([0, 3, 4, 8], dtype=np.int32)
    result = sampler._get_logprobs_tensors(
        gpu_samples,
        counts,
        raw.to("mps")[:, :37],
        torch.from_numpy(offsets).to("mps"),
        offsets,
        3,
    )
    expected_ids = torch.tensor([11, 22, 0, 0, 3, 4, 5, 0])
    ids = result.logprob_token_ids.cpu()
    torch.testing.assert_close(ids[:, 0], expected_ids)
    values = logits if logits_mode else logits.log_softmax(-1)
    torch.testing.assert_close(
        result.logprobs.cpu(), values.gather(1, ids), atol=2e-6, rtol=2e-6
    )
    ranks = (logits >= logits.gather(1, expected_ids[:, None])).sum(-1)
    torch.testing.assert_close(result.selected_token_ranks.cpu(), ranks)
    assert result.cu_num_generated_tokens == offsets.tolist()
    torch.testing.assert_close(gpu_samples.cpu(), samples[:, :4])
