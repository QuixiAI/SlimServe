# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated-MoE data parallelism: each replica shards experts over its own
TP group and never dispatches tokens across replicas."""

from types import SimpleNamespace

import pytest

from vllm.model_executor.layers.fused_moe import config as moe_config


@pytest.fixture
def groups(monkeypatch):
    monkeypatch.setattr(
        moe_config, "get_dp_group", lambda: SimpleNamespace(rank_in_group=1)
    )
    monkeypatch.setattr(moe_config, "get_tensor_model_parallel_rank", lambda: 3)


def _parallel(replicate: bool, ep: bool = False):
    return SimpleNamespace(
        enable_expert_parallel=ep,
        data_parallel_replicate_moe=replicate,
        all2all_backend="naive",
        enable_eplb=False,
    )


def test_default_dp_flattens_tp_across_replicas(groups):
    cfg = moe_config.FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=2, sp_size_=1,
        vllm_parallel_config=_parallel(replicate=False),
    )
    # dp rank 1, tp rank 3 -> shard 7 of 8: every replica holds half the
    # experts and the runner dispatches across replicas.
    assert (cfg.tp_size, cfg.tp_rank, cfg.dp_size, cfg.dp_rank) == (8, 7, 2, 1)
    assert not cfg.use_ep


def test_replicated_moe_keeps_experts_local(groups):
    cfg = moe_config.FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=2, sp_size_=1,
        vllm_parallel_config=_parallel(replicate=True),
    )
    assert (cfg.tp_size, cfg.tp_rank, cfg.dp_size, cfg.dp_rank) == (4, 3, 1, 0)
    assert cfg.ep_size == 1 and not cfg.use_ep


def test_replicated_moe_is_ignored_under_expert_parallel(groups):
    cfg = moe_config.FusedMoEParallelConfig.make(
        tp_size_=4, pcp_size_=1, dp_size_=2, sp_size_=1,
        vllm_parallel_config=_parallel(replicate=True, ep=True),
    )
    assert cfg.use_ep and cfg.ep_size == 8


def test_parallel_config_rejects_replicate_with_ep():
    from vllm.config.parallel import ParallelConfig

    with pytest.raises(ValueError, match="replicate"):
        ParallelConfig(
            tensor_parallel_size=1,
            data_parallel_size=2,
            enable_expert_parallel=True,
            data_parallel_replicate_moe=True,
        )
