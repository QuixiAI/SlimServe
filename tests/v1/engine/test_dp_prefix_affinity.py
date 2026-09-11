# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-affinity routing for the DP load balancer."""

from vllm.v1.engine.dp_prefix_affinity import (
    PrefixAffinityRouter,
    prompt_block_hashes,
)

BLOCK = 4


def _prompt(n: int, seed: int = 1) -> list[int]:
    return [seed * 1000 + i for i in range(n)]


def test_block_hashes_are_chained_and_ignore_partial_tail():
    a = prompt_block_hashes(_prompt(9), BLOCK)
    b = prompt_block_hashes(_prompt(11), BLOCK)
    assert len(a) == 2 and a == b  # same two full blocks, tails ignored
    c = prompt_block_hashes(_prompt(12), BLOCK)
    assert c[:2] == a and len(c) == 3
    # A different first block changes every later hash (chained).
    d = prompt_block_hashes([7] + _prompt(12)[1:], BLOCK)
    assert d[0] != c[0] and d[1] != c[1]
    # cache_salt seeds the chain.
    assert prompt_block_hashes(_prompt(8), BLOCK, "salt") != a


def test_no_history_routes_least_loaded_in_scan_order():
    r = PrefixAffinityRouter(2, BLOCK)
    eng, matched = r.choose(_prompt(8), None, [3, 1])
    assert (eng, matched) == (1, 0)
    # Ties resolve from start_index.
    r = PrefixAffinityRouter(2, BLOCK)
    assert r.choose(_prompt(8, seed=2), None, [0, 0], start_index=1)[0] == 1
    assert r.choose(_prompt(8, seed=3), None, [0, 0], start_index=0)[0] == 0


def test_continued_conversation_follows_its_prefix():
    r = PrefixAffinityRouter(2, BLOCK, load_tokens=4)
    turn1 = _prompt(40)
    assert r.choose(turn1, None, [0, 0])[0] == 0
    # Turn 2 extends turn 1; engine 0 is busier but holds 10 blocks (40
    # tokens) of the prefix, worth 10 queued requests at load_tokens=4.
    turn2 = turn1 + _prompt(12, seed=9)
    eng, matched = r.choose(turn2, None, [5, 0])
    assert (eng, matched) == (0, 10)
    # The new blocks are now remembered on engine 0 as well.
    assert r.matched_blocks(0, prompt_block_hashes(turn2, BLOCK)) == 13


def test_severe_imbalance_overrides_affinity():
    r = PrefixAffinityRouter(2, BLOCK, load_tokens=4)
    turn1 = _prompt(40)
    r.choose(turn1, None, [0, 0])
    # 40 cached tokens are worth 10 load points; engine 0 is 20 ahead.
    eng, _ = r.choose(turn1 + _prompt(4, seed=9), None, [20, 0])
    assert eng == 1


def test_unrelated_prompts_spread_by_load():
    r = PrefixAffinityRouter(2, BLOCK)
    assert r.choose(_prompt(8, seed=1), None, [0, 0])[0] == 0
    assert r.choose(_prompt(8, seed=2), None, [1, 0])[0] == 1


def test_capacity_evicts_oldest_blocks():
    r = PrefixAffinityRouter(1, BLOCK, capacity_blocks=3)
    hashes = prompt_block_hashes(_prompt(20), BLOCK)  # 5 blocks
    r.record(0, hashes)
    assert r.matched_blocks(0, hashes) == 0  # first two evicted -> chain breaks
    assert r.matched_blocks(0, hashes[2:]) == 3  # the retained tail is intact
    assert hashes[2] in r._seen[0] and hashes[0] not in r._seen[0]


def test_resize_follows_engine_count():
    r = PrefixAffinityRouter(2, BLOCK)
    r.choose(_prompt(8), None, [0, 0, 0])
    assert r.num_engines == 3
    r.resize(1)
    assert r.num_engines == 1


def test_short_prompt_without_a_full_block_is_least_loaded():
    r = PrefixAffinityRouter(2, BLOCK)
    eng, matched = r.choose(_prompt(3), None, [2, 1])
    assert (eng, matched) == (1, 0)
    assert all(len(s) == 0 for s in r._seen)


def test_dp_lb_client_routes_through_the_prefix_router():
    """The balancer's request path uses the router and keeps its
    bookkeeping (in-flight map, local waiting bump, rotation)."""
    from types import SimpleNamespace

    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    client = object.__new__(DPLBAsyncMPClient)
    client.core_engines = ["eng0", "eng1"]
    client.lb_engines = [[0, 0], [0, 0]]
    client.eng_start_index = 0
    client.client_count = 1
    client.reqs_in_flight = {}
    client.prefix_router = PrefixAffinityRouter(2, BLOCK, load_tokens=4)

    def req(rid, tokens):
        return SimpleNamespace(
            request_id=rid,
            data_parallel_rank=None,
            pooling_params=None,
            prompt_token_ids=tokens,
            cache_salt=None,
        )

    turn1 = _prompt(40)
    assert client.get_core_engine_for_request(req("a", turn1)) == "eng0"
    assert client.reqs_in_flight["a"] == "eng0"
    assert client.lb_engines[0][0] == 1  # local waiting bump
    # An unrelated prompt goes to the less loaded engine 1.
    assert client.get_core_engine_for_request(req("b", _prompt(40, 7))) == "eng1"
    # Turn 2 of conversation "a" follows its prefix to engine 0 even though
    # engine 0 now reports more load.
    client.lb_engines = [[2, 1], [0, 0]]
    assert (
        client.get_core_engine_for_request(req("c", turn1 + _prompt(8, 9)))
        == "eng0"
    )
    # Explicit data_parallel_rank still wins.
    r = req("d", turn1)
    r.data_parallel_rank = 1
    assert client.get_core_engine_for_request(r) == "eng1"
