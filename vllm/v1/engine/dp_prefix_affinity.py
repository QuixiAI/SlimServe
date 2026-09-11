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

Block hashes are chained like the engine's prefix cache (parent hash +
block tokens, seeded by ``cache_salt``) but computed locally, so only
self-consistency matters; the engine's own hash algorithm is not involved.
Multimodal placeholders hash as their placeholder token ids.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict
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
    ):
        assert block_size > 0 and capacity_blocks > 0
        self.block_size = block_size
        self.capacity = capacity_blocks
        self.load_tokens = load_tokens
        self._seen: list[OrderedDict[bytes, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]
        # Routing statistics for the periodic log line.
        self.routed = 0
        self.routed_with_match = 0
        self.matched_blocks_total = 0
        self.prompt_blocks_total = 0
        self.followed_affinity = 0

    @property
    def num_engines(self) -> int:
        return len(self._seen)

    def resize(self, num_engines: int) -> None:
        while len(self._seen) < num_engines:
            self._seen.append(OrderedDict())
        del self._seen[num_engines:]

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
        best, best_cost, best_matched, max_matched = 0, None, 0, 0
        for i in range(num_engines):
            idx = (start_index + i) % num_engines
            matched = self.matched_blocks(idx, hashes) if hashes else 0
            max_matched = max(max_matched, matched)
            cost = (
                n_tokens
                - matched * self.block_size
                + self.load_tokens * load_scores[idx]
            )
            if best_cost is None or cost < best_cost:
                best, best_cost, best_matched = idx, cost, matched
        if hashes:
            self.record(best, hashes)
        self.routed += 1
        self.prompt_blocks_total += len(hashes)
        if max_matched:
            self.routed_with_match += 1
            self.matched_blocks_total += best_matched
            if best_matched >= max_matched:
                self.followed_affinity += 1
        return best, best_matched

    def stats(self) -> str:
        return (
            f"routed={self.routed} with_prefix={self.routed_with_match} "
            f"followed={self.followed_affinity} matched_blocks="
            f"{self.matched_blocks_total}/{self.prompt_blocks_total}"
        )
