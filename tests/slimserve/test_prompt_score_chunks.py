# SPDX-License-Identifier: Apache-2.0
import weakref

import pytest
import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.prompt_logprobs import MODES, gather_prompt_logprobs


class CPUSampler:
    """Same public row operations; compiled GPU rank kernel tested separately."""

    @staticmethod
    def compute_logprobs(logits):
        return logits.log_softmax(-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(scores, count, targets):
        values, indices = torch.topk(scores, count, dim=-1)
        target_scores = scores.gather(-1, targets[:, None])
        return LogprobsTensors(
            torch.cat((targets[:, None], indices), -1).to(torch.int32),
            torch.cat((target_scores, values), -1),
            (scores >= target_scores).sum(-1),
        )


def reference(logits, targets, count, mode, sampler=CPUSampler):
    scores = (
        logits.float() if mode.endswith("logits") else sampler.compute_logprobs(logits)
    )
    return sampler.gather_logprobs(scores, count, targets)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 16, 639, 1024, 1025, 2051])
@pytest.mark.parametrize("count", [0, 5])
def test_exact_row_chunks_preserve_scores_topk_ranks_and_inputs(
    mode, dtype, rows, count
):
    source = torch.randn(
        rows + 2, 23, generator=torch.Generator().manual_seed(530910)
    ).to(dtype)
    original = source.clone()
    # Padded noncontiguous view exercises vocabulary trimming and offset rows.
    logits = source[1:-1, 2:19]
    targets = torch.arange(rows, dtype=torch.int64) % logits.shape[1]
    expected = reference(logits, targets, count, mode)
    actual = gather_prompt_logprobs(logits, targets, count, mode, sampler=CPUSampler)
    assert actual.cu_num_generated_tokens is None
    for a, b in zip(actual[:3], expected[:3]):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    assert torch.equal(source, original)


@pytest.mark.parametrize("mode", MODES)
def test_equal_logits_preserve_topk_ties_and_inclusive_rank(mode):
    logits = torch.ones(1025, 17, dtype=torch.bfloat16)
    targets = torch.arange(1025) % 17
    actual = gather_prompt_logprobs(logits, targets, 17, mode, sampler=CPUSampler)
    expected = reference(logits, targets, 17, mode)
    assert all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3]))
    assert torch.equal(actual.selected_token_ranks, torch.full((1025,), 17))


def test_previous_score_matrix_is_released_before_next_chunk():
    shapes, refs = [], []

    class Observed(CPUSampler):
        @staticmethod
        def compute_logprobs(piece):
            assert all(r() is None for r in refs)
            scores = CPUSampler.compute_logprobs(piece)
            shapes.append(tuple(scores.shape))
            refs.append(weakref.ref(scores))
            return scores

    result = gather_prompt_logprobs(
        torch.ones(2051, 17),
        torch.zeros(2051, dtype=torch.int64),
        0,
        "raw_logprobs",
        sampler=Observed,
    )
    assert shapes == [(1024, 17), (1024, 17), (3, 17)]
    assert all(r() is None for r in refs)
    assert result.logprobs.shape == (2051, 1)


@pytest.mark.parametrize(
    "change",
    [
        "rows",
        "columns",
        "dtype",
        "targets",
        "target_dtype",
        "count",
        "count_bool",
        "mode",
        "chunk",
        "chunk_bool",
    ],
)
def test_invalid_inputs_never_launch_sampler(change):
    logits, targets, count, mode, chunk = (
        torch.ones(2, 3),
        torch.zeros(2, dtype=torch.int64),
        0,
        "raw_logprobs",
        1024,
    )
    if change == "rows":
        logits = logits[:0]
    elif change == "columns":
        logits = logits[:, :0]
    elif change == "dtype":
        logits = logits.long()
    elif change == "targets":
        targets = targets[:1]
    elif change == "target_dtype":
        targets = targets.int()
    elif change == "count":
        count = -1
    elif change == "count_bool":
        count = True
    elif change == "mode":
        mode = "unknown"
    elif change == "chunk":
        chunk = 0
    else:
        chunk = True
    with pytest.raises(ValueError, match="invalid prompt-score"):
        gather_prompt_logprobs(
            logits, targets, count, mode, sampler=object(), chunk_rows=chunk
        )


def test_full_vocabulary_result_is_requested_output_not_unbounded_scratch():
    logits, targets = torch.zeros(33, 17), torch.zeros(33, dtype=torch.int64)
    result = gather_prompt_logprobs(
        logits, targets, 17, "raw_logits", sampler=CPUSampler, chunk_rows=16
    )
    assert result.logprobs.shape == (33, 18)
    assert result.logprob_token_ids.shape == (33, 18)


def test_sampler_failure_propagates_without_fallback():
    class Broken(CPUSampler):
        @staticmethod
        def gather_logprobs(*args):
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        gather_prompt_logprobs(
            torch.ones(1025, 17),
            torch.zeros(1025, dtype=torch.int64),
            0,
            "raw_logprobs",
            sampler=Broken,
        )
