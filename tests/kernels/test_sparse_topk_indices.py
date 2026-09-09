# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Invariants of the sparse-attention top-k index selection.

`top_k_per_row_decode` chooses which KV tokens the DSA sparse attention
attends to. Its output feeds an index-conversion kernel that maps an invalid
entry to physical KV slot 0 rather than skipping it, so the load-bearing
invariant is that no invalid entry ever appears in the span the attention
actually reads -- which is min(position + 1, topk_tokens) entries per row.
"""

import hashlib
from pathlib import Path

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


@pytest.mark.parametrize("columns", [8193, 12288, 65537, 199999])
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("per_row_lengths", [False, True])
def test_single_block_decode_changed_inputs(
    columns, topk, stride, per_row_lengths, record_property
):
    """Regression guard for both generic single-CTA decode specializations."""
    root = Path(__file__).resolve().parents[2]
    for key, path in (
        ("native_sha256", root / "vllm/_C_stable_libtorch.abi3.so"),
        ("test_source_sha256", Path(__file__)),
    ):
        with path.open("rb") as stream:
            record_property(key, hashlib.file_digest(stream, "sha256").hexdigest())
    rows, next_n = 8, 2
    backing = torch.full((rows * columns * stride + 2,), 417.0, device="cuda")
    logits = backing[1:-1:stride].view(rows, columns)
    output = torch.empty(rows, topk, dtype=torch.int32, device="cuda")
    lengths = torch.zeros(
        (4, 2) if per_row_lengths else (4,), dtype=torch.int32, device="cuda"
    )
    workspace = torch.empty(0, dtype=torch.uint8, device="cuda")

    def call():
        ops.top_k_per_row_decode(
            logits,
            next_n,
            lengths,
            output,
            workspace,
            rows,
            logits.stride(0),
            logits.stride(1),
            topk,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for phase in range(3):
        g = torch.Generator().manual_seed(2381 + phase)
        values = (
            torch.stack(
                [torch.randperm(columns, generator=g) for _ in range(rows)]
            ).float()
            if phase == 0
            else torch.randint(-2, 3, (rows, columns), generator=g).float()
        )
        if phase == 2:
            values.zero_()
            values[:, 1::2] = -0.0
        if per_row_lengths:
            host_lengths = torch.tensor(
                [0, 1, topk - 1, topk, topk + 1, columns // 2, columns - 1, columns],
                dtype=torch.int32,
            ).view(4, 2)
            visible = host_lengths.flatten()
        else:
            host_lengths = torch.tensor(
                [0, topk - 1, columns // 2, columns], dtype=torch.int32
            )
            visible = (
                (host_lengths[:, None] - 1 + torch.arange(2)[None, :])
                .clamp_min(0)
                .flatten()
            )
        defined = torch.arange(columns)[None, :] < visible[:, None]
        expected = values.masked_fill(~defined, -torch.inf).topk(topk, dim=1).values
        values[~defined] = float("nan") if phase % 2 == 0 else 123.0
        logits.copy_(values)
        lengths.copy_(host_lengths)
        before = logits.contiguous().view(torch.uint8).clone()
        for callback in (call, graph.replay):
            output.fill_(-777)
            callback()
            selected = output.cpu()
            valid = torch.arange(topk)[None, :] < visible.clamp_max(topk)[:, None]
            assert torch.equal(selected >= 0, valid)
            assert (selected[~valid] == -1).all()
            assert ((selected < visible[:, None]) | ~valid).all()
            sorted_ids = selected.sort(dim=1).values
            assert (
                (sorted_ids[:, 1:] != sorted_ids[:, :-1]) | (sorted_ids[:, 1:] == -1)
            ).all()
            scores = values.gather(1, selected.clamp_min(0).long()).masked_fill(
                ~valid, -torch.inf
            )
            assert torch.equal(scores.sort(dim=1, descending=True).values, expected)
            assert torch.equal(before, logits.contiguous().view(torch.uint8))
            assert backing[0] == 417 and backing[-1] == 417
            if stride == 2:
                assert (backing[2:-1:2] == 417).all()
