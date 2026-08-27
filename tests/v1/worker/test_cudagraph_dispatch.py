# SPDX-License-Identifier: Apache-2.0
"""Pure dispatch-plan tests for the V2 CudaGraphManager.

Every dynamic-speculative failure so far lived in the capture-plan /
dispatch logic, not in kernels: the drafter-manager zero-division, and the
multi-family staging bug where a 60-token qlen-3 batch was mapped only to
the numerically nearest (qlen-4) graph and ran eager. These tests pin that
logic down without touching CUDA graphs themselves.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager


def _config(
    capture_sizes: list[int],
    max_capture: int,
    max_num_seqs: int,
    schedule: list[list[int]] | None,
    num_spec_tokens: int,
):
    speculative = None
    if num_spec_tokens:
        speculative = SimpleNamespace(
            num_speculative_tokens=num_spec_tokens,
            num_speculative_tokens_per_batch_size=schedule,
            uses_dynamic_speculative_decoding=lambda: schedule is not None,
        )
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=max_capture,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1, tensor_parallel_size=1
        ),
        speculative_config=speculative,
        num_speculative_tokens=num_spec_tokens,
    )


@pytest.fixture()
def manager_factory(monkeypatch):
    fake_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", lambda: fake_group)
    monkeypatch.setattr(
        cudagraph_utils.current_platform,
        "get_global_graph_pool",
        Mock(return_value=None),
    )

    def build(decode_query_len, schedule=None, num_spec_tokens=2,
              capture_sizes=None, max_capture=128, max_num_seqs=32):
        if capture_sizes is None:
            capture_sizes = [1, 2, 4, 8] + list(range(16, max_capture + 1, 8))
        mgr = CudaGraphManager(
            _config(capture_sizes, max_capture, max_num_seqs, schedule,
                    num_spec_tokens),
            torch.device("cpu"),
            CUDAGraphMode.FULL_DECODE_ONLY,
            decode_query_len,
        )
        mgr._graphs_captured = True
        return mgr

    return build


def test_static_k_uniform_batch_hits_full(manager_factory):
    mgr = manager_factory(decode_query_len=3)  # k=2
    desc = mgr.dispatch(8, 24, 3, 0)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == 3


def test_static_k_pads_up_within_family(manager_factory):
    mgr = manager_factory(decode_query_len=3)
    desc = mgr.dispatch(20, 60, 3, 0)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.num_tokens >= 60 and desc.num_tokens % 3 == 0


def test_dynamic_schedule_padding_crosses_family_boundaries(manager_factory):
    """The 60-token qlen-3 case: the nearest capture key belongs to the
    qlen-4 family, but a compatible qlen-3 graph exists two slots up."""
    schedule = [[1, 4, 3], [5, 32, 2]]
    mgr = manager_factory(decode_query_len=4, schedule=schedule,
                          num_spec_tokens=3)
    desc = mgr.dispatch(20, 60, 3, 0)
    assert desc.cg_mode == CUDAGraphMode.FULL, "qlen-3 batch fell to eager"
    assert desc.uniform_token_count == 3
    assert desc.num_tokens >= 60

    desc4 = mgr.dispatch(4, 16, 4, 0)
    assert desc4.cg_mode == CUDAGraphMode.FULL
    assert desc4.uniform_token_count == 4

    # Each family is captured only for its scheduled batch range plus the
    # k-switch margin: qlen 4 (k=3, batches 1-4) covers batches 1-8, qlen 3
    # (k=2, batches 5-32) covers batches 1-32. Outside a family's range the
    # step correctly runs eager instead of paying graph memory for shapes
    # the schedule never produces.
    for reqs in range(1, 33):
        for qlen in (3, 4):
            tokens = reqs * qlen
            if tokens > 128:
                continue
            d = mgr.dispatch(reqs, tokens, qlen, 0)
            in_range = reqs <= 8 if qlen == 4 else True
            if in_range:
                assert d.cg_mode == CUDAGraphMode.FULL, (
                    f"{reqs} reqs x qlen {qlen} ({tokens} tokens) ran eager"
                )
                assert d.uniform_token_count == qlen
            else:
                assert d.cg_mode == CUDAGraphMode.NONE, (
                    f"{reqs} reqs x qlen {qlen} captured outside its range"
                )


def test_non_uniform_batch_is_eager_in_full_decode_only(manager_factory):
    mgr = manager_factory(decode_query_len=3)
    desc = mgr.dispatch(20, 61, None, 0)
    assert desc.cg_mode == CUDAGraphMode.NONE


def test_over_ceiling_is_eager(manager_factory):
    mgr = manager_factory(decode_query_len=3, max_capture=64,
                          capture_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64])
    desc = mgr.dispatch(32, 96, 3, 0)
    assert desc.cg_mode == CUDAGraphMode.NONE


def test_drafter_manager_survives_dynamic_schedule(manager_factory):
    """The drafter's manager (query length 1, smaller than the schedule's
    implied lengths) must not derive zero/negative query lengths."""
    schedule = [[1, 4, 3], [5, 32, 2]]
    mgr = manager_factory(decode_query_len=1, schedule=schedule,
                          num_spec_tokens=3)
    desc = mgr.dispatch(8, 8, 1, 0)
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc.uniform_token_count == 1


def test_dynamic_families_stay_within_a_static_graph_budget(manager_factory):
    """Two-family capture must not blow up graph count: the qlen-4 family
    is clipped to its scheduled batches, so the total stays near the
    single-family plan (measured: unclipped two-family capture starved the
    KV pool and throttled c32 to batch 16)."""
    static = manager_factory(decode_query_len=3)
    dynamic = manager_factory(
        decode_query_len=4,
        schedule=[[1, 4, 3], [5, 32, 2]],
        num_spec_tokens=3,
    )

    def full_descs(mgr):
        return {
            d
            for descs in mgr._candidates.values()
            for d in descs
            if d.cg_mode == CUDAGraphMode.FULL
        }

    n_static = len(full_descs(static))
    n_dynamic = len(full_descs(dynamic))
    assert n_dynamic <= n_static + 6, (
        f"dynamic capture plans {n_dynamic} graphs vs {n_static} static; "
        "family ranges are not being applied"
    )
