# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar
from vllm.v1.worker.gpu import structured_outputs
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as dspark_speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.structured_output import (
    DraftGrammarBatch,
    DraftStructuredOutputState,
)


class _Grammar(StructuredOutputGrammar):
    def __init__(self, allowed_by_position: list[set[int]]) -> None:
        self.allowed_by_position = allowed_by_position
        self.position = 0

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        del request_id
        position = self.position
        for token in tokens:
            if token not in self.allowed_by_position[position]:
                return False
            position += 1
        self.position = position
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        accepted = []
        for position, token in enumerate(tokens, self.position):
            if token not in self.allowed_by_position[position]:
                break
            accepted.append(token)
        return accepted

    def rollback(self, num_tokens: int) -> None:
        self.position -= num_tokens

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].zero_()
        for token in self.allowed_by_position[self.position]:
            bitmask[batch_index, token // 32] |= 1 << (token % 32)

    def is_terminated(self) -> bool:
        return self.position == len(self.allowed_by_position)

    def reset(self) -> None:
        self.position = 0


class _Backend:
    @staticmethod
    def allocate_token_bitmask(num_rows: int) -> torch.Tensor:
        return torch.zeros((num_rows, 1), dtype=torch.int32)


class _MaskingWorker:
    def __init__(self) -> None:
        self.target_token_ids: torch.Tensor | None = None

    def apply_grammar_bitmask_rows(
        self,
        logits: torch.Tensor,
        rows: list[int],
        bitmask: np.ndarray,
        target_token_ids: torch.Tensor | None = None,
    ) -> None:
        self.target_token_ids = target_token_ids
        for mask_row, logits_row in enumerate(rows):
            for column in range(logits.shape[1]):
                token = (
                    column
                    if target_token_ids is None
                    else int(target_token_ids[column])
                )
                allowed = (int(bitmask[mask_row, token // 32]) >> (token % 32)) & 1
                if not allowed:
                    logits[logits_row, column] = float("-inf")


def test_draft_grammar_batch_masks_advances_and_rolls_back() -> None:
    first = _Grammar([{1}, {3}])
    second = _Grammar([{2}, {4}])
    worker = _MaskingWorker()
    batch = DraftGrammarBatch(
        SimpleNamespace(backend=_Backend()),
        worker,
        rows=[0, 2],
        request_ids=["first", "second"],
        grammars=[first, second],
    )

    logits = torch.zeros((3, 8))
    batch.apply(logits)

    assert torch.isfinite(logits[0]).nonzero().flatten().tolist() == [1]
    assert torch.isfinite(logits[1]).all()
    assert torch.isfinite(logits[2]).nonzero().flatten().tolist() == [2]

    batch.advance(torch.tensor([1, 7, 2]))
    assert first.position == 1
    assert second.position == 1

    next_logits = torch.zeros((3, 8))
    batch.apply(next_logits)
    assert torch.isfinite(next_logits[0]).nonzero().flatten().tolist() == [3]
    assert torch.isfinite(next_logits[2]).nonzero().flatten().tolist() == [4]

    batch.rollback()
    assert first.position == 0
    assert second.position == 0
    assert batch.advancements == [0, 0]


def test_draft_grammar_batch_disables_rejected_mirror_without_raising() -> None:
    grammar = _Grammar([{1}, {2}])
    rejected: list[tuple[str, str]] = []
    batch = DraftGrammarBatch(
        SimpleNamespace(backend=_Backend()),
        _MaskingWorker(),
        rows=[0],
        request_ids=["request"],
        grammars=[grammar],
        on_reject=lambda req_id, reason: rejected.append((req_id, reason)),
    )

    # Simulate disagreement between the applied mask and the local matcher.
    batch.advance(torch.tensor([7]))

    assert batch.enabled == [False]
    assert rejected == [("request", "draft matcher rejected its masked token 7")]
    logits = torch.zeros((1, 8))
    batch.apply(logits)
    assert torch.isfinite(logits).all()


def test_worker_masks_reduced_draft_vocabulary_by_target_id(monkeypatch) -> None:
    current_stream = MagicMock()
    copy_stream = MagicMock()
    worker = object.__new__(structured_outputs.StructuredOutputsWorker)
    # The local torch build reports MPS as the active accelerator even in
    # CPU-only unit tests. Mark this worker as MPS so the production path does
    # not request pinned host memory from the unavailable allocator.
    worker.device = torch.device("mps")
    worker.copy_stream = copy_stream
    worker.grammar_bitmask = torch.zeros((2, 1), dtype=torch.int32)
    worker.logits_indices = torch.zeros(2, dtype=torch.int32)

    monkeypatch.setattr(
        structured_outputs.torch.accelerator,
        "current_stream",
        lambda _device: current_stream,
    )
    monkeypatch.setattr(
        structured_outputs,
        "stream",
        lambda *_args: contextlib.nullcontext(),
    )

    def copy_to_device(value, out):
        source = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        return out.copy_(source)

    monkeypatch.setattr(structured_outputs, "async_copy_to_gpu", copy_to_device)

    logits = torch.zeros((2, 3))
    grammar_bitmask = np.array([[1 << 5]], dtype=np.int32)
    worker.apply_grammar_bitmask_rows(
        logits,
        mapping=[1],
        grammar_bitmask=grammar_bitmask,
        target_token_ids=torch.tensor([2, 5, 7]),
    )

    assert torch.isfinite(logits[0]).all()
    assert torch.isfinite(logits[1]).nonzero().flatten().tolist() == [1]
    current_stream.wait_stream.assert_called_once_with(copy_stream)
    copy_stream.wait_stream.assert_called_once_with(current_stream)


class _DraftModel:
    def __init__(self, logits: torch.Tensor, target_ids: torch.Tensor) -> None:
        self.logits = logits
        self.target_ids = target_ids

    def compute_draft_logits(self, _hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits.clone()

    @staticmethod
    def markov_embed(tokens: torch.Tensor) -> torch.Tensor:
        return tokens

    def markov_bias(self, _tokens: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(self.logits)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return self.target_ids[draft_ids]


def test_dspark_full_vocab_model_does_not_require_mapping_attribute(
    monkeypatch,
) -> None:
    model = SimpleNamespace()
    monkeypatch.setattr(
        dspark_speculator,
        "load_dspark_model",
        lambda _target, _config: model,
    )
    speculator = object.__new__(DSparkSpeculator)
    speculator.vllm_config = object()
    speculator.draft_logits = None
    speculator._draft_target_ids = None
    speculator._d2t_scatter_index = None
    speculator._draft_scatter_buf = None

    loaded = speculator.load_draft_model(object(), set())

    assert loaded is model
    assert speculator._draft_target_ids is None


def test_dspark_chooses_grammar_valid_reduced_vocab_token() -> None:
    speculator = object.__new__(DSparkSpeculator)
    speculator.num_speculative_steps = 1
    speculator.sample_indices = torch.tensor([0])
    speculator.sample_idx_mapping = torch.tensor([0])
    speculator.sample_pos = torch.tensor([1])
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([9]))
    speculator._anchor_idx = torch.tensor([0])
    speculator.model = _DraftModel(
        logits=torch.tensor([[10.0, 9.0, 8.0]]),
        target_ids=torch.tensor([2, 5, 7]),
    )
    speculator.draft_logits = None
    speculator.draft_tokens = torch.zeros((1, 1), dtype=torch.int64)
    speculator._draft_target_ids = torch.tensor([2, 5, 7])
    speculator.draft_grammar = MagicMock()

    def allow_only_target_five(logits, target_token_ids=None) -> None:
        assert target_token_ids.tolist() == [2, 5, 7]
        logits[:, target_token_ids != 5] = float("-inf")

    speculator.draft_grammar.apply.side_effect = allow_only_target_five

    speculator._sample_sequential(1, torch.zeros((1, 1)))

    assert speculator.draft_tokens.tolist() == [[5]]
    speculator.draft_grammar.advance.assert_called_once()
    assert speculator.draft_grammar.advance.call_args.args[0].tolist() == [5]


class _MirrorRequest:
    def __init__(self, request_id: str, grammar: _Grammar) -> None:
        self.request_id = request_id
        self.structured_output_request = SimpleNamespace(grammar=grammar)
        self.output: list[int] = []

    def append_output_token_ids(self, tokens: list[int]) -> None:
        self.output.extend(tokens)


def test_verified_tokens_follow_authoritative_mask_rows() -> None:
    constrained_grammar = _Grammar([{12}, {13}])
    reasoning_grammar = _Grammar([{21}])
    constrained = _MirrorRequest("constrained", constrained_grammar)
    reasoning = _MirrorRequest("reasoning", reasoning_grammar)

    state = object.__new__(DraftStructuredOutputState)
    state.manager = SimpleNamespace(backend=_Backend())
    state.worker = _MaskingWorker()
    state.requests = {
        "constrained": constrained,
        "reasoning": reasoning,
    }
    state.draft_grammars = {
        "constrained": _Grammar([{12}, {13}]),
        "reasoning": _Grammar([{21}]),
    }
    state.verified_tokens = {
        "constrained": [],
        "reasoning": [],
    }
    state.draft_active = set()
    state.draft_disabled = set()
    input_batch = SimpleNamespace(
        req_ids=["constrained", "reasoning", "unstructured"],
        cu_num_logits_np=np.array([0, 2, 4, 5], dtype=np.int32),
    )
    # The constrained request crosses the reasoning boundary between its two
    # rows. The reasoning request remains unrestricted. All-ones packed masks
    # are the scheduler's unrestricted sentinel.
    grammar_output = GrammarOutput(
        structured_output_request_ids=["constrained", "reasoning"],
        grammar_bitmask=np.array([[-1], [0], [-1], [-1]], dtype=np.int32),
    )

    state.advance_verified(
        input_batch,
        sampled_token_ids=torch.tensor([[99, 12], [20, 21], [30, 31]]),
        num_sampled=torch.tensor([2, 2, 1]),
        grammar_output=grammar_output,
    )

    assert constrained.output == [99, 12]
    assert reasoning.output == [20, 21]
    assert constrained_grammar.position == 1
    assert reasoning_grammar.position == 0
    assert state.draft_grammars["constrained"].position == 1
    assert state.draft_grammars["reasoning"].position == 0
    assert state.draft_active == {"constrained"}

    draft = state.begin_draft(input_batch)
    assert draft is not None
    assert draft.rows == [0]
    assert draft.request_ids == ["constrained"]
    assert draft.grammars == [state.draft_grammars["constrained"]]


def test_unconstrained_rows_delay_grammar_aware_drafting() -> None:
    grammar = _Grammar([{12}])
    request = _MirrorRequest("request", grammar)
    state = object.__new__(DraftStructuredOutputState)
    state.manager = SimpleNamespace(backend=_Backend())
    state.worker = _MaskingWorker()
    state.requests = {"request": request}
    state.draft_grammars = {"request": _Grammar([{12}])}
    state.verified_tokens = {"request": []}
    state.draft_active = set()
    state.draft_disabled = set()
    input_batch = SimpleNamespace(
        req_ids=["request"],
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
    )

    state.advance_verified(
        input_batch,
        sampled_token_ids=torch.tensor([[99]]),
        num_sampled=torch.tensor([1]),
        grammar_output=GrammarOutput(["request"], np.array([[-1]], dtype=np.int32)),
    )
    assert state.begin_draft(input_batch) is None

    state.advance_verified(
        input_batch,
        sampled_token_ids=torch.tensor([[12]]),
        num_sampled=torch.tensor([1]),
        grammar_output=GrammarOutput(["request"], np.array([[0]], dtype=np.int32)),
    )
    assert state.draft_active == {"request"}


def test_worker_mirror_rejection_disables_drafting_without_raising() -> None:
    grammar = _Grammar([{7}])
    request = _MirrorRequest("request", grammar)
    state = object.__new__(DraftStructuredOutputState)
    state.manager = SimpleNamespace(backend=_Backend())
    state.worker = _MaskingWorker()
    state.requests = {"request": request}
    state.draft_grammars = {"request": _Grammar([{7}])}
    state.verified_tokens = {"request": []}
    state.draft_active = set()
    state.draft_disabled = set()
    input_batch = SimpleNamespace(
        req_ids=["request"],
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
    )

    state.advance_verified(
        input_batch,
        sampled_token_ids=torch.tensor([[12]]),
        num_sampled=torch.tensor([1]),
        grammar_output=GrammarOutput(["request"], np.array([[0]], dtype=np.int32)),
    )

    assert state.draft_disabled == {"request"}
    assert state.begin_draft(input_batch) is None


def test_draft_matcher_recovers_from_rollback_drift() -> None:
    state = object.__new__(DraftStructuredOutputState)
    draft_grammar = _Grammar([{12}, {13}])
    # Simulate a speculative rollback that left the matcher one token behind.
    draft_grammar.position = 0
    state.draft_grammars = {"request": draft_grammar}
    state.verified_tokens = {"request": [12]}
    state.draft_active = {"request"}
    state.draft_disabled = set()

    assert state._advance_draft_mirror("request", [13])

    assert draft_grammar.position == 2
    assert state.verified_tokens["request"] == [12]
