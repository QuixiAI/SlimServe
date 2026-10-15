# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded placeholder drafts (-1) are rejected outright.

The scheduler pads a newcomer whose prompt has one token left to the batch's
speculative width so the step keeps its full cudagraph, and marks those rows
with draft id -1. The runner has no drafts for that request yet, so the rows
must verify as rejections and the single emitted token must come from the
target distribution, on every verification path: the Triton Leviathan and
block-verification kernels, the native CUDA kernels, with and without draft
logits. Before this, block verification with a stale draft-logits row
accepted the placeholders and emitted three token-0 ("!") characters.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

VOCAB = 8192
N_SPEC = 3
MAX_REQS = 4
PEAK = 1234  # the token the peaked target row makes near-certain
REAL_DRAFT = [77, 88, 99]


def _run(
    target_logits,
    draft_logits,
    draft_sampled,
    cu_num_logits,
    idx_mapping,
    temp,
    use_block_verification,
):
    device = target_logits.device
    num_logits = target_logits.shape[0]
    cu = cu_num_logits.tolist()
    expanded_idx = torch.cat(
        [
            torch.full((cu[i + 1] - cu[i],), int(idx_mapping[i]), dtype=torch.int32)
            for i in range(len(cu) - 1)
        ]
    ).to(device)
    expanded_pos = torch.cat(
        [torch.arange(cu[i + 1] - cu[i], dtype=torch.int32) for i in range(len(cu) - 1)]
    ).to(device)
    sampled, num_sampled = rejection_sample(
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        torch.arange(1000, 1000 + num_logits, dtype=torch.int64, device=device),
        idx_mapping,
        expanded_idx,
        expanded_pos,
        torch.full((MAX_REQS,), temp, dtype=torch.float32, device=device),
        torch.arange(MAX_REQS, dtype=torch.int64, device=device) * 7919,
        N_SPEC,
        use_block_verification=use_block_verification,
    )
    return sampled, num_sampled


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("use_block_verification", [False, True])
@pytest.mark.parametrize("draft_kind", [None, "stale", "neg_inf"])
@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_placeholder_drafts_are_rejected(use_block_verification, draft_kind, temp):
    torch.manual_seed(0)
    device = "cuda"
    # Two requests: the padded newcomer first, a request with real drafts second
    # (so the placeholder rows sit in the middle of the flattened logits).
    num_logits = 2 * (N_SPEC + 1)
    target = torch.randn(num_logits, VOCAB, device=device) * 2.0
    target[0].fill_(-30.0)
    target[0, PEAK] = 30.0  # the newcomer's only real row: near-certain PEAK
    # Real drafts that the second request's target rows make near-certain, so
    # the request accepts everything and covers the mixed-batch indexing.
    for i, tok in enumerate(REAL_DRAFT):
        target[N_SPEC + 1 + i].fill_(-30.0)
        target[N_SPEC + 1 + i, tok] = 30.0
    draft_sampled = torch.tensor(
        [5, -1, -1, -1, 6] + REAL_DRAFT, dtype=torch.int32, device=device
    )
    cu_num_logits = torch.tensor(
        [0, N_SPEC + 1, num_logits], dtype=torch.int32, device=device
    )
    idx_mapping = torch.tensor([2, 1], dtype=torch.int32, device=device)

    draft = None
    if draft_kind is not None:
        # Slot 2 (the newcomer) holds whatever the previous occupant left.
        draft = torch.randn(MAX_REQS, N_SPEC, VOCAB, device=device) * 2.0
        if draft_kind == "neg_inf":
            draft[2].fill_(float("-inf"))
        else:
            draft[2].fill_(-30.0)
            draft[2, :, 0] = 30.0  # stale row that would accept token 0
        for i, tok in enumerate(REAL_DRAFT):
            draft[1, i].fill_(-30.0)
            draft[1, i, tok] = 30.0

    sampled, num_sampled = _run(
        target,
        draft,
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        temp,
        use_block_verification,
    )
    assert int(num_sampled[0]) == 1, (num_sampled.tolist(), sampled.tolist())
    assert int(sampled[0, 0]) == PEAK, sampled[0].tolist()
    assert int(num_sampled[1]) == N_SPEC + 1, (num_sampled.tolist(), sampled.tolist())
    assert sampled[1, :N_SPEC].tolist() == REAL_DRAFT
    assert 0 <= int(sampled[1, N_SPEC]) < VOCAB
