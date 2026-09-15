# SPDX-License-Identifier: Apache-2.0
"""The prompt tail of a prefix-cache replay (P-1 tokens hit, one prompt
token left) runs as a decode row: the V2 dispatcher keeps the FULL decode
graph for a uniform batch of such rows and the GDN builder counts them as
decodes, while fresh rows and wider tails stay on the prefill path."""

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import promote_prompt_tail_rows
from vllm.v1.worker.gpu.cudagraph_utils import uniform_token_count_with_prefills


def _guard(uniform, scheduled, computed, prompt):
    req_ids = [f"r{i}" for i in range(len(scheduled))]
    return uniform_token_count_with_prefills(
        uniform,
        dict(zip(req_ids, scheduled)),
        {r: i for i, r in enumerate(req_ids)},
        np.array(computed, dtype=np.int32),
        np.array(prompt, dtype=np.int32),
    )


@pytest.mark.parametrize("count", [1, 8, 16])
def test_prompt_tail_rows_keep_the_decode_graph(count):
    # Every request hit 999 of its 1000 prompt tokens: one token left each.
    assert _guard(1, [1] * count, [999] * count, [1000] * count) == 1


def test_prompt_tails_mixed_with_decodes_keep_the_decode_graph():
    computed = [1200, 1200, 999, 32767]
    prompt = [1000, 1000, 1000, 32768]
    assert _guard(1, [1] * 4, computed, prompt) == 1


@pytest.mark.parametrize("count", [1, 8])
def test_fresh_one_token_prompts_stay_eager(count):
    assert _guard(1, [1] * count, [0] * count, [1] * count) is None


def test_a_mid_prompt_one_token_chunk_stays_eager():
    # One token computed after a preemption, more than one token left.
    assert _guard(1, [1], [500], [1000]) is None


def test_a_spec_width_tail_stays_eager():
    # A 4-token tail chunk completing the prompt at spec width 4.
    assert _guard(4, [4, 4], [996, 1200], [1000, 1000]) is None


def test_decode_batches_and_non_uniform_batches_pass_through():
    assert _guard(1, [1, 1], [1300, 1300], [1000, 1000]) == 1
    assert _guard(None, [1, 40], [1300, 960], [1000, 1000]) is None


def test_gdn_promotes_only_one_token_tails_with_prior_state():
    is_prefilling = torch.tensor([True, True, True, False])
    query_start_loc = torch.tensor([0, 1, 2, 42, 43])  # 1, 1, 40, 1 tokens
    seq_lens = torch.tensor([1000, 1, 1000, 1300])  # tail, fresh, chunk, decode
    promoted = promote_prompt_tail_rows(is_prefilling, query_start_loc, seq_lens)
    assert promoted is not None
    assert promoted.tolist() == [False, True, True, False]
    # The input mask is left untouched.
    assert is_prefilling.tolist() == [True, True, True, False]


def test_gdn_promotion_handles_unpadded_masks_and_no_op_batches():
    is_prefilling = torch.tensor([True, False])
    query_start_loc = torch.tensor([0, 1, 2, 2, 2])  # two padded rows
    seq_lens = torch.tensor([1000, 1300, 0, 0])
    promoted = promote_prompt_tail_rows(is_prefilling, query_start_loc, seq_lens)
    assert promoted is not None and promoted.tolist() == [False, False]
    assert promote_prompt_tail_rows(None, query_start_loc, seq_lens) is None
    assert promote_prompt_tail_rows(is_prefilling, query_start_loc, None) is None
    fresh = torch.tensor([True, False])
    assert (
        promote_prompt_tail_rows(fresh, query_start_loc, torch.tensor([1, 1300, 0, 0]))
        is None
    )
