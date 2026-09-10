# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The rejection verifier must receive the distribution actually sampled."""

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


@pytest.mark.parametrize("per_row_column", [False, True])
def test_gumbel_writes_processed_draft_logits_to_persistent_state(per_row_column):
    source = torch.linspace(-3, 3, 2 * 37).view(2, 37)
    mapping = torch.tensor([3, 1], dtype=torch.int32, device="mps")
    temperatures = torch.tensor([1, 0.5, 1, 2], device="mps")
    seeds = torch.arange(42, 46, dtype=torch.int64, device="mps")
    positions = torch.tensor([1333, 2000], dtype=torch.int64, device="mps")
    cols = torch.tensor([2, 0] if per_row_column else 2, device="mps")
    backing = torch.full((4, 5, 41), -123.0, device="mps")
    output = backing[:, :3, :37]
    sampled = gumbel_sample(
        source.to("mps"),
        mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
        output_processed_logits=output,
        output_processed_logits_col=cols,
    )
    expected = torch.full((4, 5, 41), -123.0)
    expected[mapping.cpu().long(), cols.cpu(), :37] = (
        source / torch.tensor([2, 0.5])[:, None]
    )
    torch.testing.assert_close(backing.cpu(), expected)
    assert sampled.shape == (2,)
