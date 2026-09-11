# SPDX-License-Identifier: Apache-2.0
"""Native cutoff policy against independent CPU score/index ordering."""

import os
import subprocess
import sys

import pytest
import torch

from slimserve.canonical_indexer import canonicalize
from vllm import _custom_ops as ops

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)),
    reason="requires SM120",
)


def oracle(logits, ends):
    """Stable CPU sort starts in ascending pool-ID order; do not perturb scores."""
    result = torch.full((len(ends), 512), -1, dtype=torch.int32)
    for row, end in enumerate(ends.tolist()):
        assert torch.isfinite(logits[row, :end]).all()
        chosen = logits[row, :end].argsort(descending=True, stable=True)[:512]
        result[row, : len(chosen)] = chosen.sort().values.int()
    return result


def matrix(rows, columns, family, phase):
    g = torch.Generator().manual_seed(8231 + phase)
    if family == "random":
        values = torch.randn(rows, columns, generator=g)
    elif family == "unique":
        values = torch.stack(
            [torch.randperm(columns, generator=g) for _ in range(rows)]
        ).float()
        values -= columns // 2
    elif family == "ties":
        # Small candidate ties, negative cutoffs, and exact ties larger than2048.
        values = torch.randint(-2, 3, (rows, columns), generator=g).float() * 0.125
    elif family == "adjacent":
        base = torch.tensor(-8.358503341674805)
        ulp = torch.nextafter(base, torch.tensor(torch.inf)) - base
        values = base + torch.randint(-2, 3, (rows, columns), generator=g) * ulp
    elif family == "zeros":
        values = torch.zeros(rows, columns)
        values[:, (phase % 2) :: 2] = -0.0
    elif family == "cutoff":
        values = torch.full((rows, columns), -8.358503341674805)
        # >2048 equal-cutoff candidates plus strictly better/worse values when
        # wide; ordered scanning must continue across chunks of nonmembers.
        ids = torch.arange(columns)
        values[:, (ids + phase) % 17 == 0] = -7.0
        values[:, (ids + phase) % 7 == 0] = -9.0
    else:
        raise AssertionError(family)
    choices = [
        0,
        1,
        min(511, columns),
        min(512, columns),
        min(513, columns),
        columns // 2,
        max(0, columns - 1),
        columns,
    ]
    ends = torch.tensor([choices[i % 8] for i in range(rows)], dtype=torch.int32)
    return values, ends


@pytest.fixture(params=["post-sort", "fused"])
def selector(request):
    return request.param


def exercise(rows, columns, family, offset, selector, ragged=True):
    # Guard both ends; storage offsets deliberately misalign vectorized input.
    backing = torch.full((rows * columns + offset + 1,), 417.0, device="cuda")
    logits = backing[offset : offset + rows * columns].view(rows, columns)
    storage = torch.full(
        (rows * 512 + offset + 1,), -987, dtype=torch.int32, device="cuda"
    )
    indices = storage[offset : offset + rows * 512].view(rows, 512)
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.zeros_like(starts)

    def call():
        operation = (
            ops.glm53_top_k_per_row_ordered
            if selector == "fused"
            else ops.glm53_top_k_per_row_prefill
        )
        operation(logits, starts, ends, indices, rows, columns, 1, 512)
        if selector == "post-sort":
            canonicalize(indices)

    logits.zero_()
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for phase in range(3):
        values, visible = matrix(rows, columns, family, phase)
        if not ragged:
            visible.fill_(columns - phase)
        expected = oracle(values, visible)
        undefined = torch.arange(columns)[None, :] >= visible[:, None]
        values[undefined] = float("nan") if phase % 2 == 0 else 123.0
        logits.copy_(values)
        ends.copy_(visible)
        before = logits.view(torch.uint8).clone()
        for callback in (call, graph.replay):
            indices.fill_(-777)
            callback()
            assert torch.equal(indices.cpu(), expected)
            assert torch.equal(before, logits.view(torch.uint8))
            assert (starts == 0).all() and torch.equal(ends.cpu(), visible)
            assert (storage[:offset] == -987).all() and storage[-1] == -987
            assert (backing[:offset] == 417).all() and backing[-1] == 417
    torch.cuda.synchronize()


@pytest.mark.parametrize(
    "columns", [160, 512, 513, 1904, 2048, 2049, 8193, 16384, 262144]
)
@pytest.mark.parametrize(
    "family", ["random", "unique", "ties", "adjacent", "zeros", "cutoff"]
)
@pytest.mark.parametrize("offset", [0, 1])
def test_native_tie_policy_changed_input_graphs(columns, family, offset, selector):
    exercise(8, columns, family, offset, selector)


@pytest.mark.parametrize("rows,columns", [(7616, 1904), (8192, 2049)])
def test_full_chunk_geometry(rows, columns, selector):
    exercise(rows, columns, "ties", 1, selector)


@pytest.mark.parametrize("rows", [1, 2, 8, 16, 64])
@pytest.mark.parametrize("columns", [1904, 262144])
def test_small_rowcounts_all_visible(rows, columns, selector):
    exercise(rows, columns, "cutoff", 1, selector, ragged=False)


@pytest.mark.skipif(torch.cuda.device_count() != 4, reason="requires all four GPUs")
@pytest.mark.parametrize("device", [0, 1, 2, 3])
def test_native_restores_foreign_current_device(device, selector):
    values, visible = matrix(8, 1904, "adjacent", 1)
    expected = oracle(values, visible)
    logits, ends = values.to(device), visible.to(device)
    starts = torch.zeros(8, dtype=torch.int32, device=device)
    indices = torch.empty_like(expected, device=device)
    operation = (
        ops.glm53_top_k_per_row_ordered
        if selector == "fused"
        else ops.glm53_top_k_per_row_prefill
    )
    with torch.cuda.device((device + 1) % 4):
        operation(logits, starts, ends, indices, 8, 1904, 1, 512)
        assert torch.cuda.current_device() == (device + 1) % 4
        if selector == "post-sort":
            # Test the native guard independently of the Triton post-sort.
            with torch.cuda.device(device):
                canonicalize(indices)
        assert torch.equal(indices.cpu(), expected)


@pytest.mark.parametrize(
    "bad", ["rows", "columns", "topk", "stride", "dtype", "output", "device"]
)
def test_native_host_guards(bad, selector):
    logits = torch.zeros(2, 600, device="cuda")
    starts = ends = torch.zeros(2, device="cuda", dtype=torch.int32)
    indices = torch.empty(2, 512, device="cuda", dtype=torch.int32)
    rows, stride0, stride1, topk = 2, 600, 1, 512
    if bad == "rows":
        rows = 0
    elif bad == "columns":
        logits = torch.zeros(2, 262145, device="cuda")
        stride0 = 262145
    elif bad == "topk":
        topk = 256
    elif bad == "stride":
        stride1 = 2
    elif bad == "dtype":
        logits = logits.bfloat16()
    elif bad == "output":
        indices = indices[:, :511]
    elif bad == "device":
        indices = indices.cpu()
    with pytest.raises(RuntimeError, match="GLM pool selector"):
        operation = (
            ops.glm53_top_k_per_row_ordered
            if selector == "fused"
            else ops.glm53_top_k_per_row_prefill
        )
        operation(logits, starts, ends, indices, rows, stride0, stride1, topk)


@pytest.mark.parametrize("start,end", [(1, 600), (0, -1), (0, 601)])
def test_invalid_device_ranges_fail_in_isolated_process(
    start, end, record_property, selector
):
    # Device assertions invalidate a CUDA context: exercise them in a child,
    # never in the context used by the positive correctness/graph tests.
    name = "ordered" if selector == "fused" else "prefill"
    code = f"""
import torch
from vllm import _custom_ops as ops
logits = torch.zeros(1, 600, device='cuda')
starts = torch.tensor([{start}], device='cuda', dtype=torch.int32)
ends = torch.tensor([{end}], device='cuda', dtype=torch.int32)
indices = torch.empty(1, 512, device='cuda', dtype=torch.int32)
try:
    ops.glm53_top_k_per_row_{name}(logits, starts, ends, indices, 1, 600, 1, 512)
    torch.cuda.synchronize()
except RuntimeError as error:
    assert 'device-side assert' in str(error), str(error)
else:
    raise AssertionError('invalid device range was accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, CUDA_LAUNCH_BLOCKING="1"),
        text=True,
        capture_output=True,
        timeout=60,
    )
    record_property("expected_device_assert_stderr", result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "starts[row] == 0" in result.stderr
