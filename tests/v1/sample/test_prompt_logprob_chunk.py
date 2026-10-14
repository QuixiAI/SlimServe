# SPDX-License-Identifier: Apache-2.0
"""The prompt-logprob chunk is sized from free memory (a fixed 1024-token
chunk OOM-killed production on 2026-09-07) and must be identical on every TP
rank (a per-rank chunk misaligned the logits all-gather)."""
import types

import pytest
import torch

from vllm.v1.worker.gpu.sample import prompt_logprob as pl


def test_cpu_device_uses_the_cap():
    assert pl._chunk_for_budget(248320, torch.device("cpu")) == pl._MAX_CHUNK


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_chunk_follows_the_budget_and_is_bounded(monkeypatch):
    dev = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (64 << 20, 1 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 0)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: types.SimpleNamespace(world_size=1, device_group=None),
    )
    chunk = pl._chunk_for_budget(248320, dev)
    # 32 MB budget / (248320 * 16 B) = 8 tokens
    assert chunk == pl._FIRST_CHUNK
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (8 << 30, 24 << 30))
    assert pl._chunk_for_budget(248320, dev) == pl._MAX_CHUNK
