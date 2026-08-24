# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side grammar state for grammar-aware speculative drafting."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_types import StructuredOutputGrammar

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import GrammarOutput, NewRequestData
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

logger = init_logger(__name__)


class DraftGrammarBatch:
    """Temporary grammar states advanced only through one draft block."""

    def __init__(
        self,
        manager: StructuredOutputManager,
        worker: StructuredOutputsWorker,
        rows: list[int],
        request_ids: list[str],
        grammars: list[StructuredOutputGrammar],
        on_reject: Callable[[str, str], None] | None = None,
    ) -> None:
        backend = manager.backend
        assert backend is not None
        self.worker = worker
        self.rows = rows
        self.request_ids = request_ids
        self.grammars = grammars
        self.on_reject = on_reject
        self.bitmask = backend.allocate_token_bitmask(len(rows))
        self.advancements = [0] * len(rows)
        self.enabled = [True] * len(rows)

    def apply(
        self,
        logits: torch.Tensor,
        target_token_ids: torch.Tensor | None = None,
    ) -> None:
        active = [index for index, enabled in enumerate(self.enabled) if enabled]
        if not active:
            return
        for index in active:
            grammar = self.grammars[index]
            grammar.fill_bitmask(self.bitmask, index)
        rows = [self.rows[index] for index in active]
        bitmask = self.bitmask.numpy()
        if len(active) != len(self.rows):
            bitmask = bitmask[active]
        self.worker.apply_grammar_bitmask_rows(
            logits,
            rows,
            bitmask,
            target_token_ids=target_token_ids,
        )

    def advance(self, sampled_target_ids: torch.Tensor) -> None:
        row_indices = torch.tensor(
            self.rows, dtype=torch.int64, device=sampled_target_ids.device
        )
        tokens = sampled_target_ids.index_select(0, row_indices).cpu().tolist()
        for index, (request_id, grammar, token) in enumerate(
            zip(self.request_ids, self.grammars, tokens)
        ):
            if not self.enabled[index]:
                continue
            if not grammar.accept_tokens(request_id, [token]):
                reason = f"draft matcher rejected its masked token {token}"
                if self.on_reject is not None:
                    self.on_reject(request_id, reason)
                else:
                    logger.warning(
                        "Disabling draft grammar for request %s: %s",
                        request_id,
                        reason,
                    )
                self.enabled[index] = False
                continue
            self.advancements[index] += 1

    def rollback(self) -> None:
        for grammar, count in zip(self.grammars, self.advancements):
            if not count:
                continue
            grammar.rollback(count)
        self.advancements = [0] * len(self.rows)


class DraftStructuredOutputState:
    """Mirror scheduler grammar state for grammar-aware draft sampling.

    The scheduler's grammar bitmask is authoritative about the reasoning-to-
    structured-output boundary. Re-running the reasoning parser in the worker
    is not equivalent under speculative decoding: the scheduler may constrain
    only a suffix of the sampled rows. This mirror therefore advances only for
    rows that were actually constrained by the scheduler.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        worker: StructuredOutputsWorker,
    ) -> None:
        self.manager = StructuredOutputManager(vllm_config)
        # Drafting occurs in the worker's synchronous request-admission path;
        # completing compilation here avoids a first-token race with drafting.
        self.manager._use_async_grammar_compilation = False
        self.worker = worker
        self.requests: dict[str, Request] = {}
        # Verified target tokens and speculative proposals must not share one
        # matcher. Some backends only guarantee bounded rollback for the
        # speculative matcher; keeping the verified matcher isolated prevents
        # rollback drift from poisoning the next target token.
        self.draft_grammars: dict[str, StructuredOutputGrammar] = {}
        self.verified_tokens: dict[str, list[int]] = {}
        self.draft_active: set[str] = set()
        self.draft_disabled: set[str] = set()

    def add_request(self, data: NewRequestData) -> None:
        sampling_params = data.sampling_params
        if sampling_params is None or sampling_params.structured_outputs is None:
            return
        request = Request(
            request_id=data.req_id,
            prompt_token_ids=data.prompt_token_ids,
            sampling_params=sampling_params,
            pooling_params=data.pooling_params,
        )
        self.manager.grammar_init(request)
        structured = request.structured_output_request
        assert structured is not None
        grammar = structured.grammar
        if isinstance(grammar, Exception):
            raise grammar
        if not isinstance(grammar, StructuredOutputGrammar):
            raise RuntimeError(f"grammar for request {data.req_id} is not ready")
        self.requests[data.req_id] = request
        draft_grammar = self.manager._create_grammar(request)
        self.draft_grammars[data.req_id] = draft_grammar
        self.verified_tokens[data.req_id] = []

    def remove_request(self, req_id: str) -> None:
        self.requests.pop(req_id, None)
        self.draft_grammars.pop(req_id, None)
        self.verified_tokens.pop(req_id, None)
        self.draft_active.discard(req_id)
        self.draft_disabled.discard(req_id)

    @staticmethod
    def _is_unconstrained(mask: np.ndarray) -> bool:
        # StructuredOutputManager represents the reasoning / unrestricted row
        # as a packed vocabulary mask with every word set to all ones.
        return bool(np.all(mask == -1))

    def _disable(self, req_id: str, reason: str) -> None:
        if req_id not in self.draft_disabled:
            logger.warning(
                "Disabling grammar-aware speculative drafting for request %s: %s. "
                "Target-token grammar enforcement remains active.",
                req_id,
                reason,
            )
        self.draft_disabled.add(req_id)
        self.draft_active.discard(req_id)

    def _accept_verified(self, req_id: str, tokens: list[int]) -> bool:
        request = self.requests[req_id]
        structured = request.structured_output_request
        assert structured is not None
        grammar = structured.grammar
        assert isinstance(grammar, StructuredOutputGrammar)

        if grammar.accept_tokens(req_id, tokens):
            return True

        # A worker mirror must never turn a scheduler-valid target token into
        # an engine-wide failure. Try one full replay, then stop constraining
        # this request's drafts and leave validation to the scheduler.
        grammar.reset()
        history = [*self.verified_tokens[req_id], *tokens]
        if not history or grammar.accept_tokens(req_id, history):
            return True
        self._disable(
            req_id,
            f"verified matcher rejected scheduler-constrained tokens {tokens} "
            f"after history {self.verified_tokens[req_id]}",
        )
        return False

    def _advance_draft_mirror(self, req_id: str, tokens: list[int]) -> bool:
        draft_grammar = self.draft_grammars[req_id]
        if draft_grammar.validate_tokens(
            tokens
        ) == tokens and draft_grammar.accept_tokens(req_id, tokens):
            return True

        # Recover a speculative matcher whose rollback did not return to the
        # verified state. Replaying target-verified history is correctness-
        # preserving and is only paid on the exceptional recovery path.
        draft_grammar.reset()
        history = self.verified_tokens[req_id]
        if history and not draft_grammar.accept_tokens(req_id, history):
            self._disable(
                req_id,
                f"draft matcher could not replay verified history {history}",
            )
            return False
        if draft_grammar.validate_tokens(
            tokens
        ) == tokens and draft_grammar.accept_tokens(req_id, tokens):
            return True
        self._disable(
            req_id,
            f"draft matcher rejected scheduler-constrained tokens {tokens} "
            "after replaying verified history",
        )
        return False

    @staticmethod
    def _mask_rows_by_request(
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> dict[str, np.ndarray]:
        if grammar_output is None:
            return {}

        req_id_to_index = {
            req_id: index for index, req_id in enumerate(input_batch.req_ids)
        }
        cu_num_logits = input_batch.cu_num_logits_np
        rows_by_request: dict[str, np.ndarray] = {}
        offset = 0
        for req_id in grammar_output.structured_output_request_ids:
            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue
            num_rows = int(cu_num_logits[req_index + 1] - cu_num_logits[req_index])
            rows_by_request[req_id] = grammar_output.grammar_bitmask[
                offset : offset + num_rows
            ]
            offset += num_rows
        return rows_by_request

    def advance_verified(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        grammar_output: GrammarOutput | None,
    ) -> None:
        mask_rows = self._mask_rows_by_request(input_batch, grammar_output)
        counts = num_sampled.cpu().tolist()
        rows = sampled_token_ids.cpu().tolist()
        for req_id, count, row in zip(input_batch.req_ids, counts, rows):
            request = self.requests.get(req_id)
            if request is None or count <= 0:
                continue
            new_tokens = row[:count]
            request.append_output_token_ids(new_tokens)
            request_masks = mask_rows.get(req_id)
            if request_masks is None:
                continue

            if count > len(request_masks):
                self._disable(
                    req_id,
                    f"sampled {count} tokens but received only "
                    f"{len(request_masks)} authoritative mask rows",
                )
                continue

            grammar_tokens = [
                token
                for token, mask in zip(new_tokens, request_masks[:count])
                if not self._is_unconstrained(mask)
            ]
            if not grammar_tokens or req_id in self.draft_disabled:
                continue
            if not self._accept_verified(req_id, grammar_tokens):
                continue
            self.verified_tokens[req_id].extend(grammar_tokens)
            if self._advance_draft_mirror(req_id, grammar_tokens):
                # Deliberately activate only after observing a constrained
                # target row. At a reasoning boundary this can cost one draft
                # block, but it cannot admit an unconstrained draft as if it
                # had been grammar-checked.
                self.draft_active.add(req_id)

    def begin_draft(self, input_batch: InputBatch) -> DraftGrammarBatch | None:
        rows: list[int] = []
        request_ids: list[str] = []
        grammars: list[StructuredOutputGrammar] = []
        for row, req_id in enumerate(input_batch.req_ids):
            if req_id not in self.draft_active or req_id in self.draft_disabled:
                continue
            grammar = self.draft_grammars[req_id]
            if grammar.is_terminated():
                continue
            rows.append(row)
            request_ids.append(req_id)
            grammars.append(grammar)
        if not rows:
            return None
        return DraftGrammarBatch(
            self.manager,
            self.worker,
            rows,
            request_ids,
            grammars,
            on_reject=self._disable,
        )

    def shutdown(self) -> None:
        self.requests.clear()
        self.draft_grammars.clear()
        self.verified_tokens.clear()
        self.draft_active.clear()
        self.draft_disabled.clear()
        self.manager.clear_backend()
