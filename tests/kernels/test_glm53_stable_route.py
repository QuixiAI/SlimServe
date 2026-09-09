# SPDX-License-Identifier: Apache-2.0
"""Stable origin alignment; exact IDs/weight bits vs installed native router.

Synthetic changed-input eager/graphs, not full-model or performance proof.
"""

import hashlib
import itertools
from pathlib import Path

import pytest
import torch

from benchmarks.kernels.glm53_stable_route_probe import build
from slimserve.canonical_moe import canonicalize
from tests.kernels.test_glm53_canonical_moe import expected_alignment
from vllm.model_executor.layers.fused_moe.router.glm_route_align import (
    alignment_geometry,
)
from vllm.quixicore.ops import quixicore_ops as qc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def probe():
    return build()


@pytest.fixture(scope="module")
def source_receipts(probe):
    import vllm._quixicore_C as native

    paths = [
        Path(__file__),
        Path(probe.__file__),
        Path(native.__file__),
        Path("csrc/quixicore/serving/glm_moe_routing.cuh"),
        Path("benchmarks/kernels/glm53_stable_route_probe.cu"),
        Path("benchmarks/kernels/glm53_stable_route_probe.py"),
        Path("benchmarks/kernels/benchmark_mhc_output_parallel.py"),
        Path("tests/kernels/test_glm53_canonical_moe.py"),
        Path("slimserve/canonical_moe.py"),
        Path("slimserve/canonical_moe_kernel.py"),
    ]

    def hashes():
        result = {}
        for path in paths:
            with path.open("rb") as stream:
                result[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result

    initial = hashes()
    yield initial
    assert hashes() == initial, "source/native changed during qualification"


@pytest.fixture(autouse=True)
def record_sources(record_property, source_receipts):
    for path, digest in source_receipts.items():
        record_property(path, digest)


def inputs(tokens, offset=0):
    generator = torch.Generator().manual_seed(53100 + tokens)
    base = torch.randn(tokens, 288, generator=generator) * 2
    bias = torch.randn(288, generator=generator) * 0.5
    tied = torch.zeros_like(base)
    mixed = base.clone()
    mixed[:, ::3] = float("nan")
    mixed[:, 1::7] = float("inf")
    mixed[:, 2::11] = float("-inf")
    biased = bias.clone()
    biased[::7] = float("nan")
    biased[1::11] = float("inf")
    biased[2::13] = float("-inf")
    cases = [
        (base, bias),
        (tied, torch.zeros_like(bias)),
        (base[:1].expand_as(base).clone(), bias),  # maximal expert skew
        (torch.full_like(base, float("nan")), bias),
        (torch.full_like(base, float("inf")), bias),
        (torch.full_like(base, float("-inf")), bias),
        (mixed, biased),
        (-base, -bias),
    ]
    # Adjacent red zones and nonzero contiguous storage offsets; all raw input
    # bytes, including NaN payloads, must remain unchanged.
    backing = [
        torch.full((tokens * 288 + offset + 1,), 123.0, device="cuda"),
        torch.full((288 + offset + 1,), 123.0, device="cuda"),
    ]
    views = [backing[0][offset:-1].view(tokens, 288), backing[1][offset:-1]]
    return cases, backing, views


def assert_bits(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(
        left.contiguous().view(torch.uint8).cpu(),
        right.contiguous().view(torch.uint8).cpu(),
    )


def check_result(result, native, tokens, block):
    for a, b in zip(result[:2], native[:2]):
        assert_bits(a, b)
    for actual, expected in zip(result[2:], expected_alignment(native[1], block)):
        assert_bits(actual, expected)
    ids = result[1].cpu()
    assert ((ids >= 0) & (ids < 288)).all()
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()
    assert_bits(result[3], native[3])
    assert_bits(result[4], native[4])


@pytest.mark.parametrize("tokens", range(1, 17))
@pytest.mark.parametrize(
    "scoring,renormalize", tuple(itertools.product((0, 1), (False, True)))
)
@pytest.mark.parametrize("block", (8, 16, 32, 48, 64))
def test_changed_inputs(probe, tokens, scoring, renormalize, block):
    cases, backing, (logits, bias) = inputs(tokens, offset=tokens % 2)
    capacity, blocks = alignment_geometry(tokens, 8, 288, block)
    scaling = 2.5 if renormalize else 0.75

    def control():
        return qc.glm_route_align(
            logits, bias, 8, scoring, renormalize, scaling, block, capacity, blocks
        )

    def set_inputs(case):
        logits.copy_(case[0])
        bias.copy_(case[1])
        return [t.clone() for t in backing]

    # Control probe must preserve installed native router arithmetic. Candidate
    # must additionally match the independent CPU layout and diagnostic sort.
    for case in cases:
        before = set_inputs(case)
        native = control()
        raw = probe.run(logits, bias, scoring, renormalize, scaling, block, False)
        for a, b in zip(raw[:2] + raw[3:], native[:2] + native[3:]):
            assert_bits(a, b)
        stable = probe.run(logits, bias, scoring, renormalize, scaling, block, True)
        check_result(stable, native, tokens, block)
        canonicalize(*native[2:], tokens=tokens, block_size=block)
        for a, b in zip(stable[2:], native[2:]):
            assert_bits(a, b)
        for a, b in zip(backing, before):
            assert_bits(a, b)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        atomic_output = probe.run(
            logits, bias, scoring, renormalize, scaling, block, False
        )
        output = probe.run(logits, bias, scoring, renormalize, scaling, block, True)
    for case in cases + cases[::-1]:
        before = set_inputs(case)
        # Poison outputs to check every slot is written on every replay.
        for t in atomic_output + output:
            t.fill_(-123)
        graph.replay()
        native = control()
        check_result(output, native, tokens, block)
        canonicalize(*atomic_output[2:], tokens=tokens, block_size=block)
        check_result(atomic_output, native, tokens, block)
        for a, b in zip(backing, before):
            assert_bits(a, b)


@pytest.mark.parametrize("device", range(4))
def test_foreign_current_device(probe, device):
    with torch.cuda.device(device):
        logits = torch.zeros(13, 288, device="cuda")
        bias = torch.zeros(288, device="cuda")
    with torch.cuda.device((device + 1) % 4):
        current = torch.cuda.current_device()
        result = probe.run(logits, bias, 0, True, 2.5, 8, True)
        assert torch.cuda.current_device() == current
        for actual, expected in zip(result[2:], expected_alignment(result[1], 8)):
            assert actual.device == logits.device
            assert_bits(actual, expected)
