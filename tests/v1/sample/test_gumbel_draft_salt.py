# SPDX-License-Identifier: Apache-2.0
"""A drafter's Gumbel draw never shares its noise vector with the target's draw
at the same (seed, position): the drafting salt moves it to a disjoint stream."""

import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import DRAFT_NOISE_SALT, gumbel_sample


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_drafting_salt_moves_the_draw_to_a_disjoint_stream():
    torch.manual_seed(0)
    rows, vocab = 64, 4096
    device = torch.device("cuda")
    # Nearly flat logits: the noise decides the draw.
    logits = torch.randn(rows, vocab, device=device) * 0.1
    idx = torch.arange(rows, device=device, dtype=torch.int32)
    temperature = torch.ones(rows, device=device)
    seeds = torch.arange(1000, 1000 + rows, device=device, dtype=torch.int64)
    pos = torch.arange(10, 10 + rows, device=device, dtype=torch.int64)

    def draw(p, drafting):
        return gumbel_sample(
            logits.clone(), idx, temperature, seeds, p, True, is_drafting=drafting
        )

    target = draw(pos, False)
    draft = draw(pos, True)
    # The salted draw is exactly the target-side draw at the salted key ...
    assert torch.equal(draft, draw(pos + DRAFT_NOISE_SALT, False))
    # ... and differs from the target's draw at the same position.
    assert not torch.equal(target, draft)
    # Deterministic: the same key reproduces the same draft.
    assert torch.equal(draft, draw(pos, True))
