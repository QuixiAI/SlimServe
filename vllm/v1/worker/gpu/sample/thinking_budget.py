# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thinking-token budget enforcement for the V2 model runner.

Qwen3.8-class reasoning models can think nearly without bound even at low
reasoning effort, so the per-request ``thinking_token_budget`` is a hard
functional requirement, not a tuning convenience. This implementation is
GPU-resident and rides the same in-place logits-forcing seam as the grammar
bitmask, so it is async-safe and works unchanged under speculative decoding.

State machine (llama.cpp ``common/reasoning-budget.cpp`` extended with an
operator-directed wrap-up nudge, 2026-08-30):

    IDLE -> COUNTING -> [WAITING_NUDGE] -> NUDGING -> COUNTING
                     -> [WAITING_UTF8]  -> FORCING -> DONE -> COUNTING ...

- IDLE / DONE: passthrough, watching for the start-marker sequence. A new
  start marker RE-ARMS the machine with a fresh budget (a response may hold
  several reasoning blocks; each gets its own window).
- COUNTING: one budget unit per accepted token; a natural end-marker match
  transitions to DONE before that token is charged. Partial-marker prefix
  tokens are charged like ordinary tokens (reference semantics).
- NUDGING: at ~85-90%% of the budget the configured wrap-up message is
  INJECTED (forced token-by-token, uncharged), then counting resumes - the
  model reacts to the nudge with its remaining budget. Once per block.
- WAITING_NUDGE / WAITING_UTF8: the trigger token's bytes end mid-UTF-8
  codepoint; generation continues unforced until the codepoint closes, so
  neither the nudge nor the close ever severs a multi-byte character.
- FORCING: the budget is a HARD cutoff - at exhaustion the end marker is
  forced token-by-token (all other logits -> -inf), then DONE.

Under speculative decoding the machine is SIMULATED across each request's
verify span, so the first violating draft position is forced while a draft
that closes the block naturally within budget is left for rejection
sampling to validate.

Shape-static by construction: state lives in fixed per-slot tensors, the
forced sequences are padded tensors walked by per-request cursors, and all
updates are masked writes - boolean advanced indexing would give
data-dependent shapes, which drain the GPU queue per step on MPS.

Markers may tokenize to multiple tokens: matching uses a dense KMP DFA
(``[seq_len, vocab]`` transition table, one gather per token), the tensor
analogue of the reference's Aho-Corasick matcher for the single-sequence
case - including correct handling of self-overlapping sequences.
"""

import math

import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch

_NEG_INF = float("-inf")

# States (int8 tensor values).
_IDLE = 0
_COUNTING = 1
_WAITING_UTF8 = 2
_FORCING = 3
_DONE = 4
_NUDGING = 5
_WAITING_NUDGE = 6


def reasoning_budget_supported(reasoning_config) -> bool:
    """V2 enforcement needs resolved, non-empty start/end marker sequences."""
    if reasoning_config is None or not reasoning_config.enabled:
        return False
    starts = reasoning_config.reasoning_start_token_ids or []
    ends = reasoning_config.reasoning_end_token_ids or []
    return len(starts) >= 1 and len(ends) >= 1


def _find_last_seq(arr: np.ndarray, seq: list[int]) -> int:
    """Index of the last occurrence of ``seq`` in ``arr``, or -1."""
    n, m = arr.size, len(seq)
    if m == 0 or n < m:
        return -1
    if m == 1:
        hits = np.flatnonzero(arr == seq[0])
        return int(hits[-1]) if hits.size else -1
    target = np.asarray(seq)
    windows = np.lib.stride_tricks.sliding_window_view(arr, m)
    hits = np.flatnonzero((windows == target).all(axis=1))
    return int(hits[-1]) if hits.size else -1


def _kmp_dfa(seq: list[int], vocab_size: int) -> np.ndarray:
    """Dense KMP transition table: ``dfa[state, token] -> next state``.

    States 0..len(seq)-1 are match progress; a transition to ``len(seq)``
    is a completed match (the caller resets to 0). Exact-match equivalent
    of the reference's Aho-Corasick for a single pattern, with correct
    self-overlap handling.
    """
    m = len(seq)
    dfa = np.zeros((m, vocab_size), dtype=np.int32)
    if seq[0] < vocab_size:
        dfa[0, seq[0]] = 1
    fail = 0  # state after shifting off the first matched token
    for s in range(1, m):
        dfa[s, :] = dfa[fail, :]
        tok = seq[s]
        if tok < vocab_size:
            fail_next = int(dfa[fail, tok])
            dfa[s, tok] = s + 1
            fail = fail_next
        else:
            fail = 0
    return dfa


class ThinkingBudgetState:
    """Per-request GPU state for the reasoning-budget machine.

    Host-side bookkeeping is limited to which rows carry a budget at all
    (the fast skip check) and the one-time prompt scan at admission; all
    per-step accounting runs on GPU tensors so the async pipeline is never
    drained.
    """

    def __init__(
        self,
        max_num_reqs: int,
        device: torch.device,
        reasoning_config,
        vocab_size: int,
    ) -> None:
        assert reasoning_budget_supported(reasoning_config)
        self.device = device
        self.vocab_size = vocab_size

        starts = [int(t) for t in reasoning_config.reasoning_start_token_ids]
        ends = [int(t) for t in reasoning_config.reasoning_end_token_ids]
        nudge = [
            int(t)
            for t in getattr(
                reasoning_config, "thinking_budget_message_token_ids", None
            )
            or []
        ]
        self.ls = len(starts)
        self.le = len(ends)
        self.ln = len(nudge)
        self._starts_list = starts
        self._ends_list = ends
        frac = float(
            getattr(reasoning_config, "thinking_budget_nudge_fraction", 0.85) or 0.0
        )
        self.nudge_enabled = self.ln > 0 and 0.0 < frac < 1.0
        self.nudge_fraction = frac
        # The hard close forces the end marker only; the message is injected
        # earlier as the nudge so the model can still react to it.
        self.close_seq = torch.tensor(ends, dtype=torch.long, device=device)
        self.nudge_seq = (
            torch.tensor(nudge, dtype=torch.long, device=device)
            if self.ln
            else torch.zeros(1, dtype=torch.long, device=device)
        )
        self.start_dfa = torch.from_numpy(_kmp_dfa(starts, vocab_size)).to(device)
        self.end_dfa = torch.from_numpy(_kmp_dfa(ends, vocab_size)).to(device)

        # Dense per-token "ends mid-UTF-8-codepoint" flags, indexed by token
        # id; an empty config list leaves the hold inert (all complete).
        incomplete = [
            int(t)
            for t in getattr(reasoning_config, "incomplete_utf8_token_ids", None) or []
            if 0 <= int(t) < vocab_size
        ]
        if incomplete:
            table = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            table[torch.tensor(incomplete, dtype=torch.long, device=device)] = True
            self._incomplete_t: torch.Tensor | None = table
        else:
            self._incomplete_t = None

        # -1 = no budget for this row (feature inert).
        self.budget_np = np.full(max_num_reqs, -1, dtype=np.int64)
        self.budget = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.remaining = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.state = torch.zeros(max_num_reqs, dtype=torch.int8, device=device)
        self.force_pos = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.nudge_pos = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.nudged = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        # Remaining-budget level at which the nudge fires; -1 disables.
        self.nudge_rem = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=device
        )
        self.start_prog = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.end_prog = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # GPU mirror of budget_np >= 0, so per-row gathers never touch host
        # state. Staged alongside the other admission writes.
        self.active = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Admission writes staged host-side, applied in one H2D per step.
        self._staged: list[tuple[int, int, int, int, bool]] = []

    # ------------------------------------------------------------- admission

    def _nudge_rem_for(self, budget: int) -> int:
        if not self.nudge_enabled or budget <= 0:
            return -1
        # Fires once consumption reaches ceil(fraction * budget), expressed
        # as a floor on the remaining count so the GPU check is a compare.
        return budget - int(math.ceil(self.nudge_fraction * budget))

    def add_request(
        self,
        req_idx: int,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams,
    ) -> None:
        budget = sampling_params.thinking_token_budget
        if budget is None or budget < 0:
            # None = unset; -1 = the explicit unlimited opt-out. The chat
            # protocol resolves -1 to None, but SamplingParams built
            # directly (offline LLM API) can carry it here -- staging it as
            # an active budget of -1 would force the close on the first
            # in-think token.
            self.budget_np[req_idx] = -1
            # Clear the GPU active flag so a previous occupant's budget
            # cannot leak into this row's masking.
            self._staged.append((req_idx, 0, _IDLE, -1, False))
            return
        state0 = _IDLE
        if prompt_token_ids:
            arr = np.asarray(prompt_token_ids)
            last_start = _find_last_seq(arr, self._starts_list)
            last_end = _find_last_seq(arr, self._ends_list)
            if last_start > last_end:
                # Prompt ends inside an open reasoning block: generation
                # starts counting with the full budget (prompt tokens are
                # never charged). Budget <= 0 promotes straight to FORCING,
                # mirroring the reference's init promotion.
                state0 = _COUNTING if budget > 0 else _FORCING
        self.budget_np[req_idx] = budget
        self._staged.append(
            (req_idx, budget, state0, self._nudge_rem_for(budget), True)
        )

    def apply_staged_writes(self) -> None:
        if not self._staged:
            return
        idx = torch.tensor(
            [s[0] for s in self._staged], dtype=torch.long, device=self.device
        )
        bud = torch.tensor(
            [s[1] for s in self._staged], dtype=torch.int32, device=self.device
        )
        st = torch.tensor(
            [s[2] for s in self._staged], dtype=torch.int8, device=self.device
        )
        nrem = torch.tensor(
            [s[3] for s in self._staged], dtype=torch.int32, device=self.device
        )
        act = torch.tensor(
            [s[4] for s in self._staged], dtype=torch.bool, device=self.device
        )
        zeros = torch.zeros_like(bud)
        self.budget[idx] = bud
        self.remaining[idx] = bud
        self.state[idx] = st
        self.force_pos[idx] = zeros
        self.nudge_pos[idx] = zeros
        self.nudged[idx] = torch.zeros_like(act)
        self.nudge_rem[idx] = nrem
        self.start_prog[idx] = zeros
        self.end_prog[idx] = zeros
        self.active[idx] = act
        self._staged.clear()

    # ------------------------------------------------------------- internals

    def _batch_is_inert(self, input_batch: InputBatch) -> bool:
        return not bool(np.any(self.budget_np[input_batch.idx_mapping_np] >= 0))

    def _token_complete(self, t: torch.Tensor) -> torch.Tensor:
        """Per-token UTF-8 tail-completeness flags (True = complete)."""
        if self._incomplete_t is None:
            return torch.ones_like(t, dtype=torch.bool)
        return ~self._incomplete_t[torch.clamp(t, 0, self.vocab_size - 1)]

    def _accept(self, s: dict[str, torch.Tensor], t: torch.Tensor,
                valid: torch.Tensor) -> None:
        """One reference ``accept(token)`` transition, vectorized, in place.

        ``s`` holds the working state tensors (st, rem, fpos, npos, nudged,
        nrem, sprog, eprog, budget). ``valid`` gates every effect: rows
        where it is False keep all state unchanged.
        """
        st = s["st"]
        watch = valid & ((st == _IDLE) | (st == _DONE))
        counting = valid & (st == _COUNTING)
        waiting = valid & (st == _WAITING_UTF8)
        waitn = valid & (st == _WAITING_NUDGE)
        nudging = valid & (st == _NUDGING)
        forcing = valid & (st == _FORCING)
        complete = self._token_complete(t)
        tc = torch.clamp(t, 0, self.vocab_size - 1)

        # --- IDLE/DONE: watch for a start marker; match re-arms fresh.
        sprog_raw = self.start_dfa[s["sprog"].to(torch.long), tc]
        s_match = watch & (sprog_raw >= self.ls)
        s["sprog"] = torch.where(
            watch,
            torch.where(s_match, torch.zeros_like(sprog_raw), sprog_raw),
            s["sprog"],
        )

        # --- The end matcher runs while counting, waiting, or forcing the
        # close (reference tracks it during FORCING for end-match reporting);
        # it does not run through the injected nudge text.
        in_end_watch = counting | waiting | waitn | forcing
        eprog_raw = self.end_dfa[s["eprog"].to(torch.long), tc]
        e_match = in_end_watch & (eprog_raw >= self.le)
        s["eprog"] = torch.where(
            in_end_watch,
            torch.where(e_match, torch.zeros_like(eprog_raw), eprog_raw),
            s["eprog"],
        )
        nat_done = e_match & (counting | waiting | waitn)

        # --- COUNTING: charge, then exhaustion / nudge routing (skipped on
        # a natural end match: the reference breaks before the decrement).
        charge = counting & ~nat_done
        s["rem"] = s["rem"] - charge.to(s["rem"].dtype)
        exhausted = charge & (s["rem"] <= 0)
        nudge_trig = (
            charge
            & ~exhausted
            & ~s["nudged"]
            & (s["nrem"] >= 0)
            & (s["rem"] <= s["nrem"])
        )
        to_forcing_c = exhausted & complete
        to_waiting = exhausted & ~complete
        to_nudging_c = nudge_trig & complete
        to_waitn = nudge_trig & ~complete

        # --- WAITING_*: a complete token releases the held transition
        # (unless the end marker matched naturally on this very token).
        to_forcing_w = waiting & ~nat_done & complete
        to_nudging_w = waitn & ~nat_done & complete

        # --- NUDGING: advance the message cursor; at the end, counting
        # resumes and the nudge is spent for this block.
        s["npos"] = s["npos"] + nudging.to(s["npos"].dtype)
        nudge_done = nudging & (s["npos"] >= self.ln)

        # --- FORCING: advance the close cursor; sequence end -> DONE.
        s["fpos"] = s["fpos"] + forcing.to(s["fpos"].dtype)
        forced_done = forcing & (s["fpos"] >= self.le)

        # --- Start-marker match activates counting with a fresh budget and
        # a fresh nudge; budget <= 0 promotes straight to FORCING.
        s["rem"] = torch.where(s_match, s["budget"], s["rem"])
        s["nudged"] = torch.where(s_match, torch.zeros_like(s["nudged"]), s["nudged"])
        st_new = torch.where(
            s_match,
            torch.where(
                s["budget"] > 0,
                torch.full_like(st, _COUNTING),
                torch.full_like(st, _FORCING),
            ),
            st,
        )
        st_new = torch.where(nat_done, torch.full_like(st, _DONE), st_new)
        st_new = torch.where(
            to_forcing_c | to_forcing_w, torch.full_like(st, _FORCING), st_new
        )
        st_new = torch.where(to_waiting, torch.full_like(st, _WAITING_UTF8), st_new)
        st_new = torch.where(
            to_nudging_c | to_nudging_w, torch.full_like(st, _NUDGING), st_new
        )
        st_new = torch.where(to_waitn, torch.full_like(st, _WAITING_NUDGE), st_new)
        st_new = torch.where(nudge_done, torch.full_like(st, _COUNTING), st_new)
        st_new = torch.where(forced_done, torch.full_like(st, _DONE), st_new)
        s["st"] = st_new

        # Cursor and matcher resets on the transitions the reference resets.
        enter_forcing = to_forcing_c | to_forcing_w | (s_match & (s["budget"] <= 0))
        enter_nudging = to_nudging_c | to_nudging_w
        s["fpos"] = torch.where(
            enter_forcing | forced_done, torch.zeros_like(s["fpos"]), s["fpos"]
        )
        s["npos"] = torch.where(
            enter_nudging | nudge_done, torch.zeros_like(s["npos"]), s["npos"]
        )
        s["nudged"] = torch.where(
            nudge_done, torch.ones_like(s["nudged"]), s["nudged"]
        )
        s["eprog"] = torch.where(
            enter_forcing | enter_nudging | to_waiting | to_waitn | s_match | nudge_done,
            torch.zeros_like(s["eprog"]),
            s["eprog"],
        )
        s["sprog"] = torch.where(
            nat_done | forced_done, torch.zeros_like(s["sprog"]), s["sprog"]
        )

    def _forced_token(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        """The token forced for the current state (0 where passthrough)."""
        close_tok = self.close_seq[
            torch.clamp(s["fpos"].to(torch.long), max=self.le - 1)
        ]
        nudge_tok = self.nudge_seq[
            torch.clamp(s["npos"].to(torch.long), max=max(self.ln - 1, 0))
        ]
        return torch.where(s["st"] == _NUDGING, nudge_tok, close_tok)

    # ------------------------------------------------------------ step hooks

    def apply_mask(self, logits: torch.Tensor, input_batch: InputBatch) -> None:
        """Force rows of exhausted or nudging blocks, in place.

        Runs where the grammar bitmask runs: on the target logits before the
        plain sampler or the rejection sampler consumes them. Row local
        position 0's input token is the previous step's last accepted token
        (already accounted by update_state); positions >= 1 carry this
        step's draft tokens, which are folded through the state machine
        positionally here.
        """
        if self._batch_is_inert(input_batch):
            return
        tok = input_batch.input_ids[input_batch.logits_indices]  # [rows]
        slots = input_batch.idx_mapping  # [num_reqs] -> req slot

        # Per-REQUEST simulated state; consumption accumulates across a
        # request's verify span, not per row.
        s = {
            "st": self.state[slots],
            "rem": self.remaining[slots],
            "fpos": self.force_pos[slots],
            "npos": self.nudge_pos[slots],
            "nudged": self.nudged[slots],
            "nrem": self.nudge_rem[slots],
            "sprog": self.start_prog[slots],
            "eprog": self.end_prog[slots],
            "budget": self.budget[slots],
        }
        req_active = self.active[slots]

        # Span geometry is host-known (cu_num_logits_np): row indices per
        # local position are computed on the host, so nothing here syncs.
        cu = input_batch.cu_num_logits_np[: input_batch.num_reqs + 1]
        spans = np.diff(cu)
        max_local = int(spans.max()) - 1 if spans.size else 0
        num_rows = tok.shape[0]
        force = torch.zeros(num_rows, dtype=torch.bool, device=self.device)
        ftok = torch.zeros(num_rows, dtype=torch.long, device=self.device)

        def _is_forcing(st: torch.Tensor) -> torch.Tensor:
            return (st == _FORCING) | (st == _NUDGING)

        # Local position 0: the input token is the previous step's last
        # accepted token, already folded into state by update_state.
        rows0 = torch.from_numpy(cu[:-1].astype(np.int64)).to(self.device)
        force[rows0] = req_active & _is_forcing(s["st"])
        ftok[rows0] = self._forced_token(s)

        for pos in range(1, max_local + 1):
            sel = np.flatnonzero(spans > pos)
            rows_p = torch.from_numpy((cu[sel] + pos).astype(np.int64)).to(self.device)
            reqs_p = torch.from_numpy(sel.astype(np.int64)).to(self.device)
            t_p = tok[rows_p]
            sub = {k: v[reqs_p] for k, v in s.items()}
            self._accept(sub, t_p, torch.ones_like(t_p, dtype=torch.bool))
            for k in s:
                s[k][reqs_p] = sub[k]
            force[rows_p] = req_active[reqs_p] & _is_forcing(sub["st"])
            ftok[rows_p] = self._forced_token(sub)

        # Shape-static masked writes: whole-row -inf where forced, then the
        # forced column restored to 0. The scatter runs over every row with
        # a fixed [num_rows] shape; unforced rows write back their own value.
        logits.masked_fill_(force.unsqueeze(1), _NEG_INF)
        rows_all = torch.arange(num_rows, device=self.device)
        cur = logits[rows_all, ftok]
        logits[rows_all, ftok] = torch.where(force, torch.zeros_like(cur), cur)

    def update_state(
        self,
        input_batch: InputBatch,
        sampled_token_ids: torch.Tensor,  # [num_reqs, W]
        num_sampled: torch.Tensor,  # [num_reqs]
    ) -> None:
        """Fold this step's ACCEPTED tokens into the per-request state."""
        if self._batch_is_inert(input_batch):
            return
        slots = input_batch.idx_mapping  # [num_reqs] -> req slot
        s = {
            "st": self.state[slots],
            "rem": self.remaining[slots],
            "fpos": self.force_pos[slots],
            "npos": self.nudge_pos[slots],
            "nudged": self.nudged[slots],
            "nrem": self.nudge_rem[slots],
            "sprog": self.start_prog[slots],
            "eprog": self.end_prog[slots],
            "budget": self.budget[slots],
        }
        width = sampled_token_ids.shape[1]
        for p in range(width):
            self._accept(s, sampled_token_ids[:, p], p < num_sampled)
        self.state[slots] = s["st"]
        self.remaining[slots] = s["rem"]
        self.force_pos[slots] = s["fpos"]
        self.nudge_pos[slots] = s["npos"]
        self.nudged[slots] = s["nudged"]
        self.start_prog[slots] = s["sprog"]
        self.end_prog[slots] = s["eprog"]
