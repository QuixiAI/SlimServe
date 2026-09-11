# SPDX-License-Identifier: Apache-2.0
"""Order-only diagnostic: independent CPU oracle, changed graphs, real captures."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from slimserve.canonical_indexer import canonicalize

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)),
    reason="requires SM120",
)


def reference(indices):
    # CPU sort puts -1 first. Rotate each row's negative prefix to the end;
    # independent of the Triton INT_MAX-key implementation.
    ordered = indices.sort(dim=1).values
    padding = (indices == -1).sum(dim=1)
    offsets = (torch.arange(512)[None, :] + padding[:, None]) % 512
    return ordered.gather(1, offsets)


def fixture(rows, seed):
    g = torch.Generator().manual_seed(seed)
    # Duplicates are intentionally included: ordering must preserve the whole
    # multiset, not silently deduplicate or reselect. Real native IDs are unique.
    values = torch.randint(0, 262144, (rows, 512), dtype=torch.int32, generator=g)
    count = (torch.arange(rows) * 31 + seed) % 513
    count[0] = 0 if seed % 2 else 512
    values[torch.arange(512)[None, :] >= count[:, None]] = -1
    return values[:, torch.randperm(512, generator=g)]


@pytest.mark.parametrize("rows", [1, 2, 16, 17, 64, 65, 128, 640, 7616, 8192])
@pytest.mark.parametrize("offset", [0, 1])
def test_changed_eager_graph_inputs_padding_and_storage_bounds(rows, offset):
    storage = torch.full((rows * 512 + 5,), -777, device="cuda", dtype=torch.int32)
    indices = storage[offset : offset + rows * 512].view(rows, 512)
    indices.copy_(fixture(rows, 391))
    canonicalize(indices)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(device=indices.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        canonicalize(indices)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (392, 393, 394):
        value = fixture(rows, seed)
        want = reference(value)
        for callback in (lambda: canonicalize(indices), graph.replay):
            indices.copy_(value)
            callback()
            assert torch.equal(indices.cpu(), want)
            assert (storage[:offset] == -777).all()
            assert (storage[offset + rows * 512 :] == -777).all()


@pytest.mark.parametrize("device", [0, 1, 2, 3])
def test_all44_real_replay_outputs_canonicalize_without_changing_membership(device):
    root = Path(__file__).resolve().parents[2]
    path = root / "perf/results/2026-09-09/indexer-saved-input-replay/summary.json"
    if not path.exists():
        pytest.skip("requires locally retained full-chunk selector replay")
    doc = json.loads(path.read_text())
    assert doc["status"] == "complete"
    entry = next(d for d in doc["devices"] if d["device"] == device)
    archive = Path(entry["output_archive"]["path"])
    with archive.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == entry["output_archive"]["sha256"]
        )
    values = torch.load(archive, weights_only=True, map_location="cpu", mmap=True)
    assert values.shape == (11, 7616, 512)
    want = reference(values[0])
    with torch.cuda.device(device):
        indices = values[0].to(device)
        canonicalize(indices)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=stream):
            canonicalize(indices)
        torch.cuda.current_stream(device).wait_stream(stream)
        for index, value in enumerate(values):
            assert torch.equal(reference(value), want)
            indices.copy_(value)
            canonicalize(indices) if index % 2 == 0 else graph.replay()
            assert torch.equal(indices.cpu(), want)
