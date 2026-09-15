# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-affinity routing for the data-parallel load balancer.

Each DP replica owns its own prefix cache and KV tiers; the balancer alone
decides which replica a request lands on. Pure least-loaded routing sends a
continued conversation to the replica without its prefix roughly half the
time, and every such miss is a full re-prefill of the history. This router
keeps an approximate, bounded record of which prompt blocks each replica
has been sent and charges a request's routing by the prefill it would cost
on each replica:

    cost(engine) = prompt_tokens - matched_prefix_tokens(engine)
                   + load_tokens * load_score(engine)

``load_score`` is the balancer's existing ``waiting * 4 + running`` and
``load_tokens`` says how much prefill one queued request is worth, so with
no recorded prefix the choice collapses to least-loaded, and a long cached
history outweighs a moderate load imbalance but not a severe one.

Affinity is sticky by construction: once a conversation has a long history
on a replica no request-count imbalance can move it, so a chance split of
the first turns (5 sessions on one replica, 3 on the other) would stay for
the life of the sessions and churn one replica's KV pool while the other
idles (WildChat leg, 2026-09-13). Two rules keep the placement balanced:

* a ``recent`` footprint term charges each replica with the prompt tokens
  it was routed over the last ``recent_window`` requests (long-context
  sessions dominate it), weighted by ``recent_permille``; new
  conversations therefore go to the replica carrying less live context,
  and a continued conversation migrates only when the imbalance exceeds
  its own re-prefill cost;
* only the replica with the LONGEST recorded match is credited with the
  prefix. After a migration the old replica's record still matches the
  older part of the chain, but crediting it would let the session bounce
  back and forth as the footprints drift; with a single home the way back
  costs a full re-prefill again.

Block hashes are chained like the engine's prefix cache (parent hash +
block tokens, seeded by ``cache_salt``) but computed locally, so only
self-consistency matters; the engine's own hash algorithm is not involved.
Multimodal placeholders hash as their placeholder token ids.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict, deque
from collections.abc import Sequence

_DIGEST = 16


def prompt_block_hashes(
    token_ids: Sequence[int], block_size: int, cache_salt: str | None = None
) -> list[bytes]:
    """Chained hashes of every FULL block of ``token_ids``."""
    n_full = len(token_ids) // block_size
    if n_full == 0:
        return []
    parent = (cache_salt or "").encode()
    hashes: list[bytes] = []
    for b in range(n_full):
        block = array("q", token_ids[b * block_size : (b + 1) * block_size])
        parent = hashlib.blake2b(
            parent + block.tobytes(), digest_size=_DIGEST
        ).digest()
        hashes.append(parent)
    return hashes


class PrefixAffinityRouter:
    """Bounded per-engine memory of routed prompt blocks."""

    def __init__(
        self,
        num_engines: int,
        block_size: int,
        capacity_blocks: int = 262_144,
        load_tokens: int = 2048,
        recent_window: int = 64,
        recent_permille: int = 100,
    ):
        assert block_size > 0 and capacity_blocks > 0
        self.block_size = block_size
        self.capacity = capacity_blocks
        self.load_tokens = load_tokens
        self.recent_window = max(recent_window, 0)
        self.recent_permille = max(recent_permille, 0)
        self._seen: list[OrderedDict[bytes, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]
        # Prompt tokens routed per engine over the last ``recent_window``
        # requests: the live-context footprint of each replica.
        self._recent: deque[tuple[int, int]] = deque()
        self._recent_tokens: list[int] = [0] * num_engines
        # Routing statistics for the periodic log line.
        self.routed = 0
        self.routed_with_match = 0
        self.matched_blocks_total = 0
        self.prompt_blocks_total = 0
        self.followed_affinity = 0
        self.migrated = 0

    @property
    def num_engines(self) -> int:
        return len(self._seen)

    def resize(self, num_engines: int) -> None:
        while len(self._seen) < num_engines:
            self._seen.append(OrderedDict())
            self._recent_tokens.append(0)
        del self._seen[num_engines:]
        del self._recent_tokens[num_engines:]
        if len(self._recent_tokens) < num_engines:
            self._recent_tokens.extend([0] * (num_engines - len(self._recent_tokens)))

    def recent_tokens(self, engine: int) -> int:
        return self._recent_tokens[engine]

    def _note_routed(self, engine: int, n_tokens: int) -> None:
        if self.recent_window == 0:
            return
        self._recent.append((engine, n_tokens))
        self._recent_tokens[engine] += n_tokens
        while len(self._recent) > self.recent_window:
            old_engine, old_tokens = self._recent.popleft()
            if old_engine < len(self._recent_tokens):
                self._recent_tokens[old_engine] -= old_tokens

    def matched_blocks(self, engine: int, hashes: Sequence[bytes]) -> int:
        seen = self._seen[engine]
        n = 0
        for h in hashes:
            if h not in seen:
                break
            n += 1
        return n

    def record(self, engine: int, hashes: Sequence[bytes]) -> None:
        seen = self._seen[engine]
        for h in hashes:
            if h in seen:
                seen.move_to_end(h)
            else:
                seen[h] = None
        while len(seen) > self.capacity:
            seen.popitem(last=False)

    def choose(
        self,
        prompt_token_ids: Sequence[int],
        cache_salt: str | None,
        load_scores: Sequence[int],
        start_index: int = 0,
    ) -> tuple[int, int]:
        """Pick an engine; returns ``(engine_index, matched_blocks)``.

        Ties resolve in scan order from ``start_index`` so the caller's
        rotation keeps working when nothing distinguishes the engines.
        The chosen engine is recorded as holding the prompt's blocks.
        """
        num_engines = len(load_scores)
        self.resize(num_engines)
        hashes = prompt_block_hashes(prompt_token_ids, self.block_size, cache_salt)
        n_tokens = len(prompt_token_ids)
        order = [(start_index + i) % num_engines for i in range(num_engines)]
        matched_by_engine = [
            self.matched_blocks(idx, hashes) if hashes else 0 for idx in order
        ]
        max_matched = max(matched_by_engine) if matched_by_engine else 0
        # Only the longest match is a home; see the module docstring.
        home = order[matched_by_engine.index(max_matched)] if max_matched else -1
        best, best_cost, best_matched = 0, None, 0
        for idx, matched in zip(order, matched_by_engine):
            credit = matched if idx == home else 0
            cost = (
                n_tokens
                - credit * self.block_size
                + self.load_tokens * load_scores[idx]
                + (self._recent_tokens[idx] * self.recent_permille) // 1000
            )
            if best_cost is None or cost < best_cost:
                best, best_cost, best_matched = idx, cost, credit
        if hashes:
            self.record(best, hashes)
        self._note_routed(best, n_tokens)
        self.routed += 1
        self.prompt_blocks_total += len(hashes)
        if max_matched:
            self.routed_with_match += 1
            self.matched_blocks_total += best_matched
            if best == home:
                self.followed_affinity += 1
            else:
                self.migrated += 1
        return best, best_matched

    def stats(self) -> str:
        return (
            f"routed={self.routed} with_prefix={self.routed_with_match} "
            f"followed={self.followed_affinity} migrated={self.migrated} "
            f"matched_blocks={self.matched_blocks_total}/{self.prompt_blocks_total} "
            f"recent_tokens={self._recent_tokens}"
        )
