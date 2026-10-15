# SPDX-License-Identifier: Apache-2.0
"""Glm5NextPooledIndexer initialises ``packed_kw`` in ``__init__`` and
``build_packed_projection()`` (called by the model after weight load) fills
it. CPU-only: the KV-cache registration is stubbed, everything else is the
real module."""

from types import SimpleNamespace

import torch
from torch import nn

import vllm.model_executor.layers.glm5_next_indexer as idx
import vllm.model_executor.layers.linear as linear
import vllm.model_executor.parameter as parameter

HIDDEN, HEAD_DIM, N_HEADS, KP = 64, 128, 4, 4


class _StubCache(nn.Module):
    def __init__(self, head_dim, dtype, prefix, cache_config):
        super().__init__()
        self.prefix = prefix
        self.kv_cache = torch.empty(0)


def _build(monkeypatch) -> idx.Glm5NextPooledIndexer:
    monkeypatch.setattr(idx, "Glm5NextIndexerCache", _StubCache)
    monkeypatch.setattr(idx, "glm5_next_device", lambda: torch.device("cpu"))
    # Replicated projections only; no TP group is initialised on CPU.
    for mod in (linear, parameter):
        monkeypatch.setattr(mod, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(mod, "get_tensor_model_parallel_world_size", lambda: 1)
    vllm_config = SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(max_model_len=256),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
    )
    config = SimpleNamespace(
        index_n_heads=N_HEADS,
        index_head_dim=HEAD_DIM,
        index_topk=8 * KP,
        index_kpool=KP,
        index_kpool_compress=True,
        index_kpool_always_select_tail=True,
        q_lora_rank=32,
        hidden_size=HIDDEN,
    )
    topk = torch.full((4, 64), -1, dtype=torch.int32)
    with torch.device("cpu"):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            return idx.Glm5NextPooledIndexer(
                vllm_config, config, None, None, topk, prefix="layers.0.indexer"
            )
        finally:
            torch.set_default_dtype(prev)


def test_packed_kw_initialised_and_built(monkeypatch):
    indexer = _build(monkeypatch)
    # Set in __init__, before any weight load.
    assert indexer.packed_kw is None
    torch.manual_seed(0)
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.randn_like(indexer.wk.weight))
        indexer.weights_proj.weight.copy_(
            torch.randn_like(indexer.weights_proj.weight)
        )
        indexer.index_kpool_compress_gate.copy_(
            torch.randn_like(indexer.index_kpool_compress_gate)
        )
    indexer.build_packed_projection()
    packed = indexer.packed_kw
    assert packed is not None
    assert packed.shape == (2 * HEAD_DIM + N_HEADS, HIDDEN)
    assert torch.equal(packed[:HEAD_DIM], indexer.wk.weight.data)
    assert torch.equal(packed[2 * HEAD_DIM :], indexer.weights_proj.weight.data)
    # Idempotent: a second call keeps the first pack.
    indexer.build_packed_projection()
    assert indexer.packed_kw is packed
