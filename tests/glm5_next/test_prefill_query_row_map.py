# SPDX-License-Identifier: Apache-2.0
"""Regression: a KV-token map cannot be sliced as a prefill query-row map."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import glm5_next_indexer as glm
from vllm.v1.attention.backends.mla import indexer


def invoke(monkeypatch, chunk, q, cache, table_expected=None):
    rows = q.shape[0]
    md = indexer.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([4096, 8192], device=q.device, dtype=torch.int32),
        max_seq_len=8192,
        slot_mapping=torch.full((rows,), -1, device=q.device, dtype=torch.int64),
        num_decodes=0, num_decode_tokens=0, num_prefills=2, num_prefill_tokens=rows,
        prefill=indexer.DeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
    )
    monkeypatch.setattr(glm, "get_forward_context",
                        lambda: SimpleNamespace(attn_metadata={"test": md}))
    # This gate isolates selection; insertion with masked slots is unrelated.
    monkeypatch.setattr(glm, "update_pool_cache", lambda *args, **kwargs: None)
    weights = torch.randn(rows, 32, device=q.device)
    ape = torch.zeros(4, 128, device=q.device)
    output = torch.empty(rows, 2080, device=q.device, dtype=torch.int32)
    glm.glm5_next_pooled_indexer(
        q, torch.empty(rows, 256, device=q.device, dtype=torch.bfloat16),
        weights, ape, "test", cache, output,
        torch.empty(rows, 262144, device=q.device), 262144, 512, 4, 128**-0.5,
    )
    if table_expected is not None:
        selected_output = output[chunk.token_start:chunk.token_end]
        reference = torch.empty_like(selected_output)
        glm._pooled_select(q[chunk.token_start:chunk.token_end],
                           weights[chunk.token_start:chunk.token_end],
                           ape, cache, chunk.block_table, table_expected,
                           chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
                           torch.empty(reference.shape[0], 2048, device=q.device), 2048,
                           cache.shape[1], 128**-0.5, 512, reference, 4)
        assert torch.equal(selected_output.sort().values, reference.sort().values)
        assert (output[:chunk.token_start] == -1).all()
    return output


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("start,stop", [(0, 12), (5, 9)])
def test_actual_glm_op_receives_query_map_not_kv_prefix(monkeypatch, cached,
                                                      start, stop):
    lengths = torch.tensor([4096, 8192] if cached else [4, 8], dtype=torch.int32)
    cumulative = torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).int()))
    query_map = torch.tensor([0] * 4 + [1] * 8, dtype=torch.int32)[start:stop]
    row_starts = cumulative[query_map.long()]
    chunk = indexer.DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros(2, 128, dtype=torch.int32),
        cu_seqlen_ks=row_starts, cu_seqlen_ke=row_starts + 1,
        cu_seq_lens=cumulative,
        token_to_seq=torch.repeat_interleave(torch.arange(2).int(), lengths),
        total_seq_lens=int(lengths.sum()), max_seq_len=int(lengths.max()),
        token_start=0, token_end=stop - start, num_reqs=2,
        local_cu_seq_lens=cumulative,
    )
    captured = []
    monkeypatch.setattr(glm, "_pooled_select",
                        lambda *args, **kwargs: captured.append(args[5].clone()))
    q = torch.empty(stop - start, 32, 128, dtype=torch.bfloat16)
    cache = torch.empty(256, 64, 64, dtype=torch.bfloat16)
    invoke(monkeypatch, chunk, q, cache)
    assert len(captured) == 1
    assert torch.equal(captured[0], query_map)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("row_dim", [64, 256])
@pytest.mark.parametrize("start,stop", [(0, 12), (5, 9)])
@torch.no_grad()
def test_real_gpu_metadata_and_glm_selection(monkeypatch, native, row_dim, start, stop):
    torch.manual_seed(90120 + start)
    monkeypatch.setattr(indexer, "_use_native_indexer_metadata", lambda: native)
    locs_cpu = torch.tensor([0, 4, 12], dtype=torch.int32)
    lengths_cpu = torch.tensor([4096, 8192], dtype=torch.int32)
    lengths = lengths_cpu.cuda()
    table = torch.randperm(256, device="cuda").int().view(2, 128)
    chunk = indexer.build_prefill_chunk_metadata(
        0, 2, locs_cpu.cuda(), locs_cpu, lengths, lengths, lengths_cpu, table, 1,
        query_slice=slice(start, stop),
    )
    expected = torch.tensor([0] * 4 + [1] * 8, device="cuda", dtype=torch.int32)
    expected = expected[start:stop]
    # Establish the metadata contract independently of the GLM consumer.
    assert torch.equal(chunk.token_to_seq[:12], torch.zeros(12, device="cuda",
                                                          dtype=torch.int32))
    assert torch.equal(torch.index_select(chunk.token_to_seq, 0, chunk.cu_seqlen_ks),
                       expected)
    q = torch.randn(stop, 32, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(256, 2, 64, row_dim, device="cuda",
                        dtype=torch.bfloat16)[:, 1]
    invoke(monkeypatch, chunk, q, cache, expected)


def test_localized_dcp_bounds_fail_closed():
    cumulative = torch.tensor([0, 4096, 12288], dtype=torch.int32)
    chunk = SimpleNamespace(cu_seq_lens=cumulative,
                            local_cu_seq_lens=cumulative // 2)
    with pytest.raises(AssertionError, match="unsharded KV row bounds"):
        glm._prefill_query_requests(chunk)
