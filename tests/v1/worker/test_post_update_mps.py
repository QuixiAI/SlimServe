# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MPS post_update in-place masked scatter (vllm/v1/worker/gpu/input_batch.py).

The tensorized MPS branch writes sampled tokens into all_token_ids by
redirecting masked lanes to flat cell 0, where they re-write that cell's
own value. These tests pin the contract the redirect relies on: valid
rows receive exactly their sampled tokens, and every other cell of
all_token_ids — cell 0 included — is bit-unchanged.
"""

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import post_update

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires Apple Metal (MPS)",
)

DEV = "mps"


def _run_case(seed: int) -> None:
    torch.manual_seed(seed)
    max_reqs, row_capacity = 6, 32
    num_reqs, slots = 4, 3
    vocab_size = 64

    # req 1 is skipped (negative mapping), req 2 samples zero tokens, and
    # req 0's state row is index 0 so the scratch cell's own row is live.
    idx_mapping = torch.tensor([3, -1, 0, 5], dtype=torch.int32, device=DEV)
    num_sampled = torch.tensor([2, 3, 0, 1], dtype=torch.int32, device=DEV)
    num_rejected = torch.tensor([1, 0, 3, 2], dtype=torch.int32, device=DEV)
    sampled_tokens = torch.randint(
        0, vocab_size, (num_reqs, slots), dtype=torch.int32, device=DEV
    )
    query_start_loc = torch.tensor([0, 3, 7, 11, 14, 14], dtype=torch.int32, device=DEV)

    # Prompt lengths >= 1 so no valid write can target flat cell 0.
    total_len = torch.tensor([4, 9, 1, 7, 2, 5], dtype=torch.int32, device=DEV)
    all_token_ids = torch.randint(
        0, vocab_size, (max_reqs, row_capacity), dtype=torch.int32, device=DEV
    )
    num_computed_tokens = torch.randint(
        1, 20, (max_reqs,), dtype=torch.int32, device=DEV
    )
    last_sampled = torch.randint(
        0, vocab_size, (max_reqs, 1), dtype=torch.int64, device=DEV
    )
    bin_counts = torch.zeros((max_reqs, vocab_size), dtype=torch.int32, device=DEV)

    before_tokens = all_token_ids.cpu().clone()
    before_total = total_len.cpu().clone()
    before_computed = num_computed_tokens.cpu().clone()
    before_last = last_sampled.cpu().clone()

    post_update(
        idx_mapping=idx_mapping,
        num_computed_tokens=num_computed_tokens,
        last_sampled_tokens=last_sampled,
        output_bin_counts=bin_counts,
        sampled_tokens=sampled_tokens,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        query_start_loc=query_start_loc,
        all_token_ids=all_token_ids,
        total_len=total_len,
    )

    got_tokens = all_token_ids.cpu()
    sampled_cpu = sampled_tokens.cpu()
    idx_cpu = idx_mapping.cpu()
    counts_cpu = num_sampled.cpu()
    qsl_cpu = query_start_loc.cpu().to(torch.int64)
    rej_cpu = num_rejected.cpu().to(torch.int64)

    # Reference: apply the documented per-request semantics with plain loops.
    exp_tokens = before_tokens.clone()
    exp_total = before_total.clone()
    exp_computed = before_computed.clone()
    exp_last = before_last.clone()
    exp_bins = torch.zeros((max_reqs, vocab_size), dtype=torch.int32)
    for b in range(num_reqs):
        state = int(idx_cpu[b])
        if state < 0:
            continue
        count = int(counts_cpu[b])
        start = int(before_total[state])
        for j in range(count):
            tok = int(sampled_cpu[b, j])
            exp_tokens[state, start + j] = tok
            exp_bins[state, tok] += 1
        if count > 0:
            exp_last[state, 0] = int(sampled_cpu[b, count - 1])
        exp_total[state] += count
        query_len = int(qsl_cpu[b + 1] - qsl_cpu[b])
        exp_computed[state] += query_len - int(rej_cpu[b])

    assert torch.equal(got_tokens, exp_tokens), (
        "all_token_ids mismatch (stray writes or missing sampled tokens)"
    )
    assert torch.equal(total_len.cpu(), exp_total)
    assert torch.equal(num_computed_tokens.cpu(), exp_computed)
    assert torch.equal(last_sampled.cpu(), exp_last)
    assert torch.equal(bin_counts.cpu(), exp_bins)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_post_update_masked_scatter(seed: int) -> None:
    _run_case(seed)
