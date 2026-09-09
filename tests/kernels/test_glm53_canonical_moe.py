# SPDX-License-Identifier: Apache-2.0
"""Ordering-only intervention against independent CPU alignment construction."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from slimserve.canonical_moe import canonicalize, geometry
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.glm_route_align import (
    marlin_block_size_m,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def expected_alignment(ids, block_size):
    flat = ids.cpu().numpy().reshape(-1)
    capacity, max_blocks, _ = geometry(len(ids), 8, 288, block_size)
    sorted_ids = np.full(capacity, len(flat), dtype=np.int32)
    expert_ids = np.full(max_blocks, -1, dtype=np.int32)
    cursor = 0
    for expert in range(288):
        assignments = np.flatnonzero(flat == expert)
        padded = (len(assignments) + block_size - 1) // block_size * block_size
        sorted_ids[cursor : cursor + len(assignments)] = assignments
        expert_ids[cursor // block_size : (cursor + padded) // block_size] = expert
        cursor += padded
    return tuple(
        torch.from_numpy(x)
        for x in (sorted_ids, expert_ids, np.array([cursor], dtype=np.int32))
    )


@pytest.mark.parametrize(
    "tokens", [1, 2, 8, 16, 17, 31, 64, 65, 129, 640, 1024, 7616, 8192]
)
def test_exact_alignment_eager_and_changed_input_graph(tokens):
    block = marlin_block_size_m(tokens, 8, 288)
    k = torch.arange(8)
    inputs = [
        ((torch.arange(tokens)[:, None] * 37 + k) % 288).int(),
        k.expand(tokens, 8).contiguous().int(),
        ((torch.arange(tokens)[:, None] * 17 + k * 31 + 73) % 288).int(),
    ]
    ids = inputs[0].cuda()

    def call():
        result = moe_align_block_size(ids, block, 288)
        canonicalize(*result, tokens=tokens, block_size=block)
        return result

    def check(result, inp):
        for got, expected in zip(result, expected_alignment(inp, block)):
            assert torch.equal(got.cpu(), expected)
        assert torch.equal(ids.cpu(), inp)

    for inp in inputs:
        ids.copy_(inp)
        check(call(), inp)
        check(call(), inp)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=ids.device)
    stream.wait_stream(torch.cuda.current_stream(ids.device))
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    for inp in inputs + inputs[::-1]:
        ids.copy_(inp)
        graph.replay()
        check(output, inp)


def test_all_captured_layouts_canonicalize_identically():
    trace = Path("perf/results/2026-09-09/mhc-quality-moe-trace-diagnostic/trace")
    paths = sorted(trace.glob("model-*.jsonl"))
    if not paths:
        pytest.skip("requires bounded real-model MoE captures")
    assert len(paths) == 4
    canonical = None
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        files = {
            (r["match"], r["stage"]): Path(r["path"])
            for r in rows
            if r["kind"] == "moe_snapshot"
        }
        for match in (1, 2, 3):

            def load(stage, files=files, match=match):
                return torch.load(
                    files[match, "moe3." + stage], map_location="cpu", weights_only=True
                )

            ids = load("router.ids")
            count = load("up.padded_count")
            capacity, blocks, _ = geometry(640, 8, 288, 32)
            sorted_ids = torch.full((capacity,), 5120, dtype=torch.int32)
            sorted_ids[: int(count)] = load("up.sorted_ids")
            experts = torch.full((blocks,), -1, dtype=torch.int32)
            experts[: int(count) // 32] = load("up.expert_ids")
            arrays = [t.cuda() for t in (sorted_ids, experts, count)]
            canonicalize(*arrays, tokens=640, block_size=32)
            actual = [t.cpu() for t in arrays]
            for got, expected in zip(actual, expected_alignment(ids, 32)):
                assert torch.equal(got, expected)
            if canonical is None:
                canonical = actual
            assert all(torch.equal(a, b) for a, b in zip(actual, canonical))
