# SPDX-License-Identifier: Apache-2.0
"""Semantic checkpoint hints survive IPC and steer only prefill chunk ends."""

from types import SimpleNamespace

import msgspec
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.request import Request


def make_request(boundaries):
    return Request(
        request_id="semantic-test",
        prompt_token_ids=[1] * 256,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        semantic_cache_boundaries=boundaries,
    )


def test_ipc_roundtrip_preserves_semantic_hints_for_engine_request():
    wire_request = EngineCoreRequest(
        request_id="semantic-test",
        prompt_token_ids=[1] * 256,
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        semantic_cache_boundaries=[129, 65, 65, 400],
    )
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(wire_request), type=EngineCoreRequest
    )
    request = Request.from_engine_core_request(decoded, block_hasher=None)
    assert request.semantic_cache_boundaries == (65, 129)


def test_custom_request_hints_cannot_escape_prompt_or_use_boolean_offsets():
    request = make_request([True, False, -1, 0, 65, 65, 256, 257, 4.5, "32"])
    assert request.semantic_cache_boundaries == (65, 256)


def scheduler_config():
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=32),
        hash_block_size=32,
        use_eagle=False,
        mamba_partial_cache_hit=False,
    )


def test_prefill_stops_at_each_reachable_semantic_checkpoint_without_stalling():
    request = make_request([65, 129])
    chunks = []
    while request.num_computed_tokens < request.num_prompt_tokens:
        chunk = Scheduler._mamba_block_aligned_split(
            scheduler_config(), request, 256 - request.num_computed_tokens
        )
        assert chunk > 0
        chunks.append(chunk)
        request.num_computed_tokens += chunk
    assert chunks == [64, 64, 128]


@pytest.mark.parametrize("local,external", [(64, 0), (0, 64), (32, 32)])
def test_cache_hits_skip_already_computed_semantic_boundaries(local, external):
    request = make_request([65, 129])
    assert (
        Scheduler._mamba_block_aligned_split(
            scheduler_config(),
            request,
            192,
            num_new_local_computed_tokens=local,
            num_external_computed_tokens=external,
        )
        == 64
    )


def test_semantic_checkpoints_do_not_split_decode():
    request = make_request([65, 129])
    request.num_computed_tokens = 256
    assert Scheduler._mamba_block_aligned_split(scheduler_config(), request, 4) == 4
