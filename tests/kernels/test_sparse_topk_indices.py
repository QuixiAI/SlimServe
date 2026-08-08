# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariants of the sparse-attention top-k index selection.

`top_k_per_row_decode` chooses which KV tokens the DSA sparse attention
attends to. Its output feeds an index-conversion kernel that maps an invalid
entry to physical KV slot 0 rather than skipping it, so the load-bearing
invariant is that no invalid entry ever appears in the span the attention
actually reads -- which is min(position + 1, topk_tokens) entries per row.
"""

import pytest
import torch

from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device kernel")

TOPK = 2048


def _select(logits: torch.Tensor, seq_len: int) -> torch.Tensor:
    rows = logits.shape[0]
    out = torch.empty(rows, TOPK, dtype=torch.int32, device=logits.device)
    workspace = torch.empty(
        ops.top_k_per_row_decode_workspace_size(rows, logits.shape[1], TOPK),
        dtype=torch.uint8,
        device=logits.device,
    )
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device=logits.device)
    ops.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        out,
        workspace,
        rows,
        logits.stride(0),
        logits.stride(1),
        TOPK,
    )
    return out


def _logits(kind: str, seq_len: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    base = torch.randn(2, seq_len, device="cuda", generator=generator)
    if kind == "distinct":
        return base
    if kind == "tied":  # coarse quantization: thousands share a value
        return (base * 4).round() / 4
    if kind == "sparse_relu":  # what the indexer really produces
        return torch.relu(base - 2.0)
    raise AssertionError(kind)


@pytest.mark.parametrize("seq_len", [512, 2047, 2048, 2049, 3000])
@pytest.mark.parametrize("kind", ["distinct", "tied", "sparse_relu"])
def test_no_invalid_index_inside_the_consumed_span(seq_len: int, kind: str):
    """The span the attention reads must be entirely real token indices.

    An invalid entry here would be converted to KV slot 0 and attended to with
    full weight, silently mixing an unrelated token into every request.
    """
    selected = _select(_logits(kind, seq_len), seq_len)
    consumed = selected[:, : min(seq_len, TOPK)]
    assert (consumed >= 0).all(), "invalid index would become physical KV slot 0"
    assert (consumed < seq_len).all(), "index past the end of the sequence"


@pytest.mark.parametrize("seq_len", [512, 2047])
def test_short_rows_pad_with_the_sentinel_beyond_the_consumed_span(seq_len: int):
    """Rows shorter than topk fill the tail with -1, which is never read."""
    selected = _select(_logits("distinct", seq_len), seq_len)
    assert (selected[:, seq_len:] == -1).all()


def test_distinct_logits_select_exactly_the_true_top_k():
    seq_len = 4096
    logits = _logits("distinct", seq_len)
    got = torch.sort(_select(logits, seq_len).long(), dim=1).values
    want = torch.sort(torch.topk(logits.float(), TOPK, dim=1).indices, dim=1).values
    torch.testing.assert_close(got, want)


def test_selection_is_stable_across_launches_when_no_tie_straddles_the_cut():
    """Slots are handed out by atomicAdd, so only the *set* is guaranteed.

    With distinct logits the set is reproducible. It is not when exactly-tied
    logits straddle the cut -- see the note in sampler.cu.
    """
    seq_len = 4096
    logits = _logits("distinct", seq_len)
    first = torch.sort(_select(logits, seq_len).long(), dim=1).values
    for _ in range(4):
        again = torch.sort(_select(logits, seq_len).long(), dim=1).values
        torch.testing.assert_close(again, first)


def test_long_decode_topk_is_graph_safe_across_all_model_layers():
    """Exercise the 1M-token split path as it is captured by the 78-layer model.

    All layer launches deliberately share one caller-owned workspace. Native
    temporary allocations here are unsafe because graph-pool reuse can alias a
    later layer's live buffers.
    """
    rows = 4
    seq_len = 1_048_576
    num_layers = 78
    generator = torch.Generator(device="cuda").manual_seed(1)
    logits = torch.randn(rows, seq_len, device="cuda", generator=generator)
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device="cuda")
    outputs = [
        torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
        for _ in range(num_layers)
    ]
    workspace = torch.empty(
        ops.top_k_per_row_decode_workspace_size(rows, seq_len, TOPK),
        dtype=torch.uint8,
        device="cuda",
    )

    def launch_all_layers() -> None:
        for output in outputs:
            ops.top_k_per_row_decode(
                logits,
                1,
                seq_lens,
                output,
                workspace,
                rows,
                logits.stride(0),
                logits.stride(1),
                TOPK,
            )

    launch_all_layers()  # initialize the kernel before graph capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_all_layers()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()

    expected = torch.sort(outputs[0].long(), dim=1).values
    assert (expected >= 0).all()
    assert (expected < seq_len).all()
    for output in outputs[1:]:
        torch.testing.assert_close(torch.sort(output.long(), dim=1).values, expected)
