# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache hits when one full-attention group's block exceeds the
Mamba ("align") block.

GLM-5.3-Flash with the compact indexer cache: the MLA group and the KDA
state group both use the 1088-token block the platform derived from the
KDA page, but the compact indexer page is 8x smaller per token, so the
page-size unification scales its block to 8704 tokens. Chunk ends (where
align-mode Mamba states are materialized) still land on 1088-token
boundaries, so hits must be found at hash-block (1088) granularity with
a fine-grained partial hit in the 8704-token indexer group; requiring
alignment to the 8704-token LCM instead left every prefix that does not
happen to end on a multiple of 8704 with zero hits.

With the DFlash2 speculator the KDA page grows, the block becomes 1152
tokens (indexer 9216), the drafter adds a sliding-window group and the
scheduler treats the drafter as EAGLE-like (one block backed off, one
hash unit dropped from every hit). The sliding-window manager must then
cache every block that a hash-aligned hit can consult, not only the
tails of scheduler-block-sized segments.

With a finer ``prefix_match_unit`` (the rtx6000 record hashes every 64
tokens) the KDA group is coarser than the hash block too, so the prompt's
partial tail sits on the replay boundary (``num_prompt_tokens - 1``): a
repeated prompt hits everything but its last token and recomputes that one
token as a decode-shaped step, and a longer prompt sharing the prefix finds
the same tail through its root hash.
"""

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    prompt_tail_boundary,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

KDA_PAGE = 1_085_440  # bytes of KDA state per block without speculation
KDA_PAGE_SPEC = 1_122_304  # with the DFlash2 speculator's extra state


class Config:
    """The engine's block geometry for one serving mode."""

    def __init__(
        self,
        spec: bool,
        compact: bool = True,
        match_unit: int | None = None,
        eagle_drop: bool | None = None,
    ):
        self.spec = spec
        # The EAGLE last-unit drop: on for hidden-state drafters that pair
        # state i with token i+1 (EAGLE, MTP); DFlash's hits are exact.
        self.eagle_drop = spec if eagle_drop is None else eagle_drop
        # attention block = smallest multiple of 64 whose 1024 B/token page
        # covers the KDA page (interface.py hybrid block alignment).
        page = KDA_PAGE_SPEC if spec else KDA_PAGE
        self.block = 64 * -(-page // (64 * 1024))
        self.indexer_block = self.block * (8 if compact else 1)
        self.scheduler_block = max(self.block, self.indexer_block)
        # prefix_match_unit: the hash block, else the attention block.
        self.hash_block = match_unit or self.block
        # max_num_batched_tokens 8192, minus the speculator's lookahead.
        self.budget = 8160 if spec else 8192


def make_manager(cfg: Config) -> KVCacheManager:
    page = cfg.block * 1024
    mla = MLAAttentionSpec(
        block_size=cfg.block, num_kv_heads=1, head_size=512, dtype=torch.bfloat16
    )
    indexer = MLAAttentionSpec(
        block_size=cfg.indexer_block,
        num_kv_heads=1,
        head_size=512 * cfg.block // cfg.indexer_block,
        dtype=torch.bfloat16,
    )
    kda = MambaSpec(
        shapes=(((KDA_PAGE_SPEC if cfg.spec else KDA_PAGE) // 4,),),
        dtypes=(torch.float32,),
        block_size=cfg.block,
        page_size_padded=page,
        mamba_cache_mode="align",
    )
    groups = [
        KVCacheGroupSpec(["mla"], mla),
        KVCacheGroupSpec(["indexer"], indexer),
        KVCacheGroupSpec(["kda"], kda),
    ]
    if cfg.spec:
        # DFlash2's five sliding-window layers: 2 kv heads per rank x 128 x
        # K,V x bf16 = the MLA group's 1024 B/token, so the same block.
        drafter = SlidingWindowSpec(
            block_size=cfg.block,
            num_kv_heads=2,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=2048,
        )
        groups.append(KVCacheGroupSpec(["draft_swa"], drafter))
    assert len({g.kv_cache_spec.page_size_bytes for g in groups}) == 1
    config = KVCacheConfig(num_blocks=2048, kv_cache_tensors=[], kv_cache_groups=groups)
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=1 << 20,
        scheduler_block_size=cfg.scheduler_block,
        hash_block_size=cfg.hash_block,
        enable_caching=True,
        use_eagle=cfg.eagle_drop,
    )


def mamba_aligned_chunk(
    cfg: Config, manager: KVCacheManager, request: Request, start: int, num_new: int
) -> int:
    """Scheduler._mamba_block_aligned_split without shared prefixes."""
    if start >= max(request.num_prompt_tokens, request.num_tokens - 1):
        return num_new
    block = cfg.block
    last_cache_position = request.num_tokens - request.num_tokens % block
    if cfg.eagle_drop:
        last_cache_position = max(last_cache_position - block, 0)
    end = start + num_new
    if end < last_cache_position:
        end = end // block * block
    next_boundary = (start // block + 1) * block
    tail_boundary = (
        prompt_tail_boundary(
            request.num_prompt_tokens,
            cfg.hash_block,
            manager.coordinator.prompt_tail_replay,
        )
        if cfg.hash_block < cfg.scheduler_block
        else 0
    )
    stops = [
        next_boundary
        if start % block != 0 and next_boundary <= last_cache_position
        else 0,
        last_cache_position,
        tail_boundary
        if 0 < tail_boundary < request.num_prompt_tokens
        and tail_boundary != last_cache_position
        else 0,
    ]
    end = min((s for s in stops if start < s < end), default=end)
    return max(end - start, 0)


def run_request(
    cfg: Config,
    manager: KVCacheManager,
    request: Request,
    decode_steps=1,
    chunks: list[int] | None = None,
):
    """Prefill in scheduler-sized chunks, then decode; returns the hit length.
    ``chunks`` collects the prefill chunk sizes when given."""
    lookahead = 4 if cfg.spec else 0
    manager.new_step_starts()
    computed_blocks, num_hit, _ = manager.get_computed_blocks(request)
    num_computed = num_hit
    first = True
    while num_computed < request.num_tokens:
        if not first:
            manager.new_step_starts()
        num_new = mamba_aligned_chunk(
            cfg,
            manager,
            request,
            num_computed,
            min(cfg.budget, request.num_tokens - num_computed),
        )
        if chunks is not None:
            chunks.append(num_new)
        assert num_new > 0, (num_computed, request.num_tokens)
        if first:
            blocks = manager.allocate_slots(
                request,
                num_new,
                num_new_computed_tokens=num_hit,
                new_computed_blocks=computed_blocks,
                num_lookahead_tokens=lookahead,
            )
            request.num_computed_tokens = num_hit
            first = False
        else:
            blocks = manager.allocate_slots(
                request, num_new, num_lookahead_tokens=lookahead
            )
        assert blocks is not None
        num_computed += num_new
        request.num_computed_tokens = num_computed
    for step in range(decode_steps):
        request.append_output_token_ids(7 + step)
        manager.new_step_starts()
        assert (
            manager.allocate_slots(request, 1, num_lookahead_tokens=lookahead)
            is not None
        )
        request.num_computed_tokens += 1
    return num_hit


def make_request(request_id: str, num_tokens: int, hasher) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=[(i * 7919) % 1000 for i in range(num_tokens)],
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=hasher,
    )


def expected_hit(cfg: Config, num_tokens: int) -> int:
    full = (num_tokens - 1) // cfg.block
    if cfg.eagle_drop:
        # The EAGLE-style drop recomputes the last block before the tail.
        full = max(full - 1, 0)
    return full * cfg.block


def make_hasher(cfg: Config):
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    return get_request_block_hasher(cfg.hash_block, hash_fn)


@pytest.mark.parametrize("spec", [False, True], ids=["no-spec", "dflash2"])
@pytest.mark.parametrize("compact", [False, True], ids=["raw", "compact"])
@pytest.mark.parametrize("num_tokens", [512, 4096, 32768, 131072])
def test_repeat_prompt_hits_at_hash_granularity(spec, compact, num_tokens):
    cfg = Config(spec, compact)
    hasher = make_hasher(cfg)
    manager = make_manager(cfg)

    cold = make_request("cold", num_tokens, hasher)
    assert run_request(cfg, manager, cold) == 0
    manager.free(cold)

    warm = make_request("warm", num_tokens, hasher)
    hit = run_request(cfg, manager, warm, decode_steps=4)
    assert hit == expected_hit(cfg, num_tokens)
    manager.free(warm)


def test_compact_hit_keeps_a_private_indexer_tail_while_the_source_lives():
    cfg = Config(spec=False)
    hasher = make_hasher(cfg)
    manager = make_manager(cfg)

    source = make_request("source", 32768, hasher)
    run_request(cfg, manager, source)
    twin = make_request("twin", 32768, hasher)
    assert run_request(cfg, manager, twin, decode_steps=4) == 30 * cfg.block
    src_blocks = manager.get_block_ids("source")
    twin_blocks = manager.get_block_ids("twin")
    # Full MLA blocks are shared; the indexer's partial 8704-token tail is a
    # copy-on-write block private to the twin so the two decodes cannot
    # overwrite each other's rows.
    assert src_blocks[0][:30] == twin_blocks[0][:30]
    assert src_blocks[1][:3] == twin_blocks[1][:3]
    assert src_blocks[1][3] != twin_blocks[1][3]
    manager.free(source)
    manager.free(twin)


@pytest.mark.parametrize("spec", [False, True], ids=["no-spec", "dflash2"])
@pytest.mark.parametrize("num_tokens", [1000, 1088, 1089, 4096, 32768])
def test_repeat_prompt_hits_its_replay_boundary_under_a_finer_match_unit(
    num_tokens, spec
):
    """prefix_match_unit 64 (the rtx6000 record): the tail sits on the replay
    boundary, so an exact repeat leaves exactly one token to compute and the
    cold prefill ends a chunk there. With the DFlash2 drafter its sliding-window
    group takes part in the fine-grained hit and no unit is dropped (its draft
    context is complete at every boundary), so the replay works under
    speculation too."""
    cfg = Config(spec=spec, match_unit=64, eagle_drop=False)
    hasher = make_hasher(cfg)
    manager = make_manager(cfg)
    assert manager.coordinator.prompt_tail_replay

    cold = make_request("cold", num_tokens, hasher)
    chunks: list[int] = []
    assert run_request(cfg, manager, cold, chunks=chunks) == 0
    assert sum(chunks[:-1]) == num_tokens - 1 and chunks[-1] == 1
    manager.free(cold)

    warm = make_request("warm", num_tokens, hasher)
    chunks = []
    assert (
        run_request(cfg, manager, warm, decode_steps=4, chunks=chunks) == num_tokens - 1
    )
    assert chunks == [1]
    manager.free(warm)


def test_a_longer_prompt_finds_the_shared_prompt_tail_through_its_root():
    cfg = Config(spec=False, match_unit=64)
    hasher = make_hasher(cfg)
    manager = make_manager(cfg)

    source = make_request("source", 1000, hasher)
    run_request(cfg, manager, source, decode_steps=300)
    manager.free(source)

    longer = Request(
        request_id="longer",
        prompt_token_ids=source.prompt_token_ids + [5] * 500,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=hasher,
    )
    chunks: list[int] = []
    # The source ran 300 tokens past its prompt, filling its first MLA block:
    # the tail entry survives that promotion for the append-only groups and
    # on a private copy of the KDA state. Resumed mid-block, the prefill
    # re-aligns to the block grid, then ends a chunk at its own tail.
    assert run_request(cfg, manager, longer, chunks=chunks) == 999
    assert chunks == [1088 - 999, 1499 - 1088, 1]
    manager.free(longer)

    unrelated = Request(
        request_id="unrelated",
        prompt_token_ids=source.prompt_token_ids[:960] + [9] * 40,
        sampling_params=SamplingParams(max_tokens=16),
        pooling_params=None,
        block_hasher=hasher,
    )
    # Same root, different tail tokens: the tail must not match. Intermediate
    # hash boundaries carry no entry of their own (a Mamba state exists only
    # where a chunk ended), so the prefix below the tail is not a hit either.
    assert run_request(cfg, manager, unrelated) == 0
    manager.free(unrelated)


def test_the_tail_index_forgets_evicted_entries():
    cfg = Config(spec=False, match_unit=64)
    hasher = make_hasher(cfg)
    manager = make_manager(cfg)
    source = make_request("source", 1000, hasher)
    run_request(cfg, manager, source)
    manager.free(source)
    # The scheduler releases the copy-on-write endpoints once the copies ran.
    _, retained = manager.take_kv_cache_block_copies()
    manager.free_blocks(retained)
    pool = manager.block_pool
    assert sum(len(t) for t in pool.tail_entries_by_root.values()) == 1
    assert manager.reset_prefix_cache()
    assert not pool.tail_entries_by_root
    again = make_request("again", 1000, hasher)
    assert run_request(cfg, manager, again) == 0
    manager.free(again)
