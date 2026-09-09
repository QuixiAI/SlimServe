# SPDX-License-Identifier: Apache-2.0
"""Exact serving entry: CPU layout, frozen probe parity, graphs and redzones."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.kernels import test_glm53_stable_align as reference
from vllm.model_executor.layers.fused_moe.router import glm_stable_align as custom

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def probe():
    import vllm._quixicore_C as native

    # Load the qualified binary directly. Do not rebuild an archived control
    # against the now-integrated header or silently change its source identity.
    directory = Path(os.environ["SLIMSERVE_GLM53_STABLE_ALIGN_PROBE"])
    path = directory / "glm53_stable_align_probe.so"
    assert (
        reference.sha(path)
        == "9b1ed1328d49b0f723cb2b33f9f06d2b18e91c271296eac5e8057e9e16116d5c"
    )
    spec = importlib.util.spec_from_file_location("glm53_stable_align_probe", path)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)

    def run_into(ids, *args):
        native.glm_stable_align(ids, *args)

    def run(ids, block):
        return custom.align(ids, block)

    return SimpleNamespace(
        __file__=native.__file__, frozen=frozen, run=run, run_into=run_into
    )


@pytest.fixture(autouse=True)
def identities(probe, record_property):
    files = reference.SOURCES + [
        Path(__file__),
        Path(probe.__file__),
        Path(probe.frozen.__file__),
        reference.ROOT / "csrc/quixicore/tm_cuda/tm_cuda_serving.cu",
        reference.ROOT / "vllm/quixicore/ops.py",
        Path(custom.__file__),
    ]
    hashes = {str(p): reference.sha(p) for p in files}
    record_property("source_sha256", json.dumps(hashes, sort_keys=True))
    yield
    assert hashes == {str(p): reference.sha(p) for p in files}


@pytest.mark.parametrize(
    "tokens", [17, 18, 30, 31, 32, 33, 34, 63, 64, 65, 129, 640, 7616, 8192]
)
@pytest.mark.parametrize("block", [8, 16, 32, 48, 64])
@pytest.mark.parametrize("offset", [0, 1])
def test_native_graphs_redzones_and_probe(probe, tokens, block, offset):
    reference.test_stable_eager_changed_input_graph_and_redzones(
        probe, tokens, block, offset
    )
    for cpu in reference.inputs(tokens):
        ids = cpu.cuda()
        candidate = probe.run(ids, block)
        small = tokens <= 32
        control = probe.frozen.run(ids, block, not small, small, not small)
        assert all(torch.equal(a, b) for a, b in zip(candidate, control))


def test_native_actual_captured_routes(probe, record_property):
    reference.test_all_actual_captured_routes(probe, record_property)


@pytest.mark.parametrize("device", [0, 1, 2, 3])
def test_native_device_and_stream(probe, device):
    reference.test_foreign_current_device_and_nondefault_stream(probe, device)


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
    ],
)
def test_native_invalid_contracts(probe, case):
    reference.test_invalid_contracts(probe, case)


@pytest.mark.parametrize("alias", ["ids-sorted", "sorted-experts", "padded-offsets"])
def test_native_rejects_aliases(probe, alias):
    capacity, blocks, _ = reference.geometry(17, 8, 288, 8)
    backing = torch.zeros(capacity + 289, dtype=torch.int32, device="cuda")
    ids = torch.zeros((17, 8), dtype=torch.int32, device="cuda")
    outputs = [
        torch.empty(n, dtype=torch.int32, device="cuda")
        for n in (capacity, blocks, 1, 289)
    ]
    if alias == "ids-sorted":
        ids, outputs[0] = backing[:136].view(17, 8), backing[:capacity]
    elif alias == "sorted-experts":
        outputs[0], outputs[1] = backing[:capacity], backing[:blocks]
    else:
        outputs[2], outputs[3] = backing[:1], backing[:289]
    with pytest.raises(RuntimeError, match="single memory location"):
        probe.run_into(ids, *outputs, 8)


@pytest.mark.parametrize("tokens", [17, 32, 33, 640, 8192])
@pytest.mark.parametrize("compiled", [False, True])
def test_custom_op_changed_input_graph(probe, tokens, compiled):
    call = custom.align
    if compiled:
        call = torch.compile(call, fullgraph=True, dynamic=False)
    ids = reference.inputs(tokens)[0].cuda()
    for _ in range(2):
        call(ids, 32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call(ids, 32)
    torch.cuda.current_stream().wait_stream(stream)
    for cpu in reference.inputs(tokens):
        ids.copy_(cpu)
        for output in outputs:
            output.fill_(-999)
        graph.replay()
        assert all(
            torch.equal(a.cpu(), b)
            for a, b in zip(outputs, reference.expected_alignment(cpu, 32))
        )
