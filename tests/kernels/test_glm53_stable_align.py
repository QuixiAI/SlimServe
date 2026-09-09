# SPDX-License-Identifier: Apache-2.0
"""Stable alignment candidate: independent CPU oracle, redzones and graphs."""

import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks.kernels.glm53_stable_align_probe import build
from slimserve.canonical_moe import canonicalize, geometry
from tests.kernels.test_glm53_canonical_moe import expected_alignment
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    Path(__file__),
    ROOT / "benchmarks/kernels/glm53_stable_align_probe.py",
    ROOT / "benchmarks/kernels/glm53_stable_align_probe.cu",
    ROOT / "csrc/quixicore/serving/glm_moe_stable_align.cuh",
    ROOT / "tests/kernels/test_glm53_canonical_moe.py",
    ROOT / "slimserve/canonical_moe.py",
    ROOT / "slimserve/canonical_moe_kernel.py",
    ROOT / "vllm/model_executor/layers/fused_moe/moe_align_block_size.py",
]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@pytest.fixture(
    scope="module",
    params=[
        pytest.param((256, True), id="256"),
        pytest.param((1024, True), id="1024"),
        pytest.param((1024, False), id="direct"),
    ],
)
def probe(request):
    module = build()
    threads, aggregate = request.param
    return SimpleNamespace(
        __file__=module.__file__,
        module=module,
        count_threads=threads,
        aggregate=aggregate,
        run=partial(module.run, parallel_count=threads == 1024, aggregate=aggregate),
        run_into=partial(
            module.run_into, parallel_count=threads == 1024, aggregate=aggregate
        ),
    )


@pytest.fixture(autouse=True)
def identities(probe, record_property):
    files = SOURCES + [Path(probe.__file__)]
    hashes = {str(p): sha(p) for p in files}
    record_property("source_sha256", json.dumps(hashes, sort_keys=True))
    record_property("count_threads", probe.count_threads)
    record_property("aggregate", int(probe.aggregate))
    yield
    assert hashes == {str(p): sha(p) for p in files}


def inputs(tokens):
    generator = torch.Generator().manual_seed(2309 + tokens)
    random = torch.rand(tokens, 288, generator=generator).topk(8, dim=1).indices.int()
    skew = torch.arange(8).expand(tokens, 8).contiguous().int()
    patterned = (
        (torch.arange(tokens)[:, None] * 37 + torch.arange(8) * 31) % 288
    ).int()
    invalid = patterned.clone()
    invalid[::3, 2] = -1
    invalid[::5, 7] = 288
    invalid[::7, 1] = 2**31 - 1
    invalid[::11, 4] = -(2**31)
    return [
        random,
        skew,
        patterned,
        invalid,
        torch.full_like(skew, -1),
        torch.full_like(skew, 13),
        random.flip(0).contiguous(),
    ]


def protected(shape, offset, device):
    size = int(np.prod(shape))
    backing = torch.full(
        (size + 64 + offset,), -1234567, dtype=torch.int32, device=device
    )
    view = backing[32 + offset : 32 + offset + size].view(shape)

    def check():
        assert (backing[: 32 + offset] == -1234567).all().item()
        assert (backing[32 + offset + size :] == -1234567).all().item()

    return view, check


def expected_offsets(ids, block):
    flat = ids.numpy().reshape(-1)
    counts = np.bincount(flat[(flat >= 0) & (flat < 288)], minlength=288)
    counts = (counts + block - 1) // block * block
    return torch.from_numpy(np.concatenate(([0], counts.cumsum())).astype(np.int32))


@pytest.mark.parametrize("tokens", [17, 31, 32, 33, 63, 64, 65, 129, 640, 7616, 8192])
@pytest.mark.parametrize("block", [8, 16, 32, 48, 64])
@pytest.mark.parametrize("offset", [0, 1])
def test_stable_eager_changed_input_graph_and_redzones(probe, tokens, block, offset):
    capacity, blocks, _ = geometry(tokens, 8, 288, block)
    ids, check_ids = protected((tokens, 8), offset, "cuda")
    allocations = [
        protected(shape, offset, "cuda")
        for shape in [(capacity,), (blocks,), (1,), (289,)]
    ]
    outputs = [x[0] for x in allocations]
    phases = inputs(tokens)
    expected = [
        (*expected_alignment(p, block), expected_offsets(p, block)) for p in phases
    ]

    def call():
        probe.run_into(ids, *outputs, block)

    def check(index):
        for got, ref in zip(outputs, expected[index]):
            assert torch.equal(got.cpu(), ref)
        assert torch.equal(ids.cpu(), phases[index])
        check_ids()
        for _, guard in allocations:
            guard()

    for index, inp in enumerate(phases):
        ids.copy_(inp)
        for output in outputs:
            output.fill_(-999)
        call()
        check(index)
        # Canonical post-sort assumes distinct top-k IDs per row. Duplicates
        # deliberately exceed that separate diagnostic's bound; test our
        # candidate against the independent CPU oracle for that phase only.
        if index != 5:
            native = moe_align_block_size(ids, block, 288)
            canonicalize(*native, tokens=tokens, block_size=block)
            assert all(torch.equal(a, b) for a, b in zip(native, outputs[:3]))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    for index in list(range(7)) + list(reversed(range(7))):
        ids.copy_(phases[index])
        for output in outputs:
            output.fill_(919191)
        graph.replay()
        check(index)


def test_all_actual_captured_routes(probe, record_property):
    directory = ROOT / "perf/results/2026-09-09/stable-route-quality-diagnostic/trace"
    paths = sorted(directory.glob("model-*.jsonl"))
    assert len(paths) == 4, "the prescribed four-rank capture is required"
    receipts = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        matches = [
            r
            for r in rows
            if r["kind"] == "moe_snapshot" and r["stage"] == "moe3.router.ids"
        ]
        assert len(matches) == 3
        for row in matches:
            archive = Path(row["path"])
            assert sha(archive) == row["file_sha256"]
            ids = torch.load(archive, weights_only=True, map_location="cpu")
            assert ids.shape == (640, 8)
            got = probe.run(ids.cuda(), 32)
            assert all(
                torch.equal(a.cpu(), b)
                for a, b in zip(got, expected_alignment(ids, 32))
            )
            receipts.append(dict(path=str(archive), sha256=sha(archive)))
    record_property("actual_routes", json.dumps(receipts, sort_keys=True))


@pytest.mark.parametrize("device", [0, 1, 2, 3])
def test_foreign_current_device_and_nondefault_stream(probe, device):
    if torch.cuda.device_count() != 4:
        pytest.skip("requires the four-GPU qualification box")
    ids_cpu = inputs(129)[0]
    with torch.cuda.device(device):
        ids = ids_cpu.cuda(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.cuda.device((device + 1) % 4):
            output = probe.run(ids, 48)
            assert torch.cuda.current_device() == (device + 1) % 4
        stream.synchronize()
        assert all(
            torch.equal(a.cpu(), b)
            for a, b in zip(output, expected_alignment(ids_cpu, 48))
        )


@pytest.mark.parametrize(
    "case",
    [
        "rows16",
        "rows8193",
        "topk7",
        "float",
        "cpu",
        "strided",
        "block0",
        "block128",
        "capacity",
        "expert_shape",
        "padded_dtype",
        "offset_size",
        "device",
        "direct_without_parallel",
    ],
)
def test_invalid_contracts(probe, case):
    ids = torch.zeros((17, 8), dtype=torch.int32, device="cuda")
    capacity, blocks, _ = geometry(17, 8, 288, 8)
    outputs = [
        torch.empty(n, dtype=torch.int32, device="cuda")
        for n in (capacity, blocks, 1, 289)
    ]
    block = 8
    if case in ("rows16", "rows8193", "topk7"):
        shape = {"rows16": (16, 8), "rows8193": (8193, 8), "topk7": (17, 7)}[case]
        ids = torch.zeros(shape, dtype=torch.int32, device="cuda")
    elif case == "float":
        ids = ids.float()
    elif case == "cpu":
        ids = ids.cpu()
    elif case == "strided":
        ids = torch.zeros((17, 16), dtype=torch.int32, device="cuda")[:, ::2]
    elif case.startswith("block"):
        block = int(case[5:])
    elif case == "capacity":
        outputs[0] = outputs[0][:-1]
    elif case == "expert_shape":
        outputs[1] = outputs[1].view(1, -1)
    elif case == "padded_dtype":
        outputs[2] = outputs[2].float()
    elif case == "offset_size":
        outputs[3] = outputs[3][:-1]
    elif case == "device":
        outputs[0] = outputs[0].cpu()
    elif case == "direct_without_parallel":
        with pytest.raises(RuntimeError, match="requires 1024 threads"):
            probe.module.run_into(ids, *outputs, block, False, False)
        return
    with pytest.raises(RuntimeError):
        probe.run_into(ids, *outputs, block)
