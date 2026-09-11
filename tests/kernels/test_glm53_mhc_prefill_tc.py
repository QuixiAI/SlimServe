# SPDX-License-Identifier: Apache-2.0
"""Native SM120 opt-in dispatch, independent accuracy, graphs and fallback."""

import pytest
import torch

from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import installed
from benchmarks.kernels.benchmark_glm53_mhc_storage import exact
from benchmarks.kernels.benchmark_mhc_output_parallel import inputs
from benchmarks.kernels.mhc_fp64_oracle import accuracy_pair
from vllm.quixicore.ops import quixicore_ops as qc

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and qc.has_glm53_mhc_prefill_tc()
    ),
    reason="requires the SM120 native mHC tensor-core build",
)


@pytest.fixture(autouse=True)
def restore_switches():
    old = (
        qc.get_glm53_mhc_prefill_tc(),
        qc.get_dsv4_mhc_mode(),
        qc.get_dsv4_mhc_prefill_min_t(),
    )
    qc.set_dsv4_mhc_mode(2)
    qc.set_dsv4_mhc_prefill_min_t(64)
    yield
    qc.set_glm53_mhc_prefill_tc(old[0])
    qc.set_dsv4_mhc_mode(old[1])
    qc.set_dsv4_mhc_prefill_min_t(old[2])


def data_for(batch=65, seed=941):
    data = inputs(batch, seed)[:7]
    data[4] = data[4].bfloat16()
    return data


@pytest.mark.parametrize("batch", [64, 65, 129, 1000, 7616])
@pytest.mark.parametrize("fused", [False, True])
def test_native_both_arms_pass_independent_accuracy(batch, fused):
    data = data_for(batch)
    qc.set_glm53_mhc_prefill_tc(0)
    reference = installed(data, fused)
    qc.set_glm53_mhc_prefill_tc(1)
    candidate = installed(data, fused)
    exact(reference[:1], candidate[:1])
    result = accuracy_pair(reference[0], *data[4:], reference[1:], candidate[1:])
    assert result["passed"], result


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "fallback",
    [
        "small",
        "below_min",
        "large",
        "fp16",
        "fp32",
        "offset1",
        "offset2",
        "offset4",
        "rms_eps",
        "pre_eps",
        "sinkhorn_eps",
        "post_mult",
        "iterations",
        "norm",
    ],
)
def test_unqualified_cases_remain_bit_exact_to_control(fused, fallback):
    batch = {"small": 16, "below_min": 63, "large": 7617}.get(fallback, 65)
    data = data_for(batch)
    constants = [1e-5, 1e-6, 1e-6, 2.0, 20, None, 0.0]
    if fallback in ("fp16", "fp32"):
        data[4] = data[4].to(torch.float16 if fallback == "fp16" else torch.float32)
    if fallback.startswith("offset"):
        offset = int(fallback[-1])
        storage = torch.empty(
            data[4].numel() + offset, device="cuda", dtype=torch.bfloat16
        )
        unaligned = storage[offset:].view_as(data[4])
        unaligned.copy_(data[4])
        data[4] = unaligned
    changed = {
        "rms_eps": (0, 1e-6),
        "pre_eps": (1, 1e-2),
        "sinkhorn_eps": (2, 1e-5),
        "post_mult": (3, 0.5),
        "iterations": (4, 3),
    }
    if fallback in changed:
        index, value = changed[fallback]
        constants[index] = value
    if fallback == "norm":
        constants[5] = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        constants[6] = 1e-5

    def call():
        if fused:
            return qc.dsv4_mhc_fused_post_pre(*data, *constants)
        return [data[1], *qc.dsv4_mhc_pre(data[1], *data[4:], *constants)]

    qc.set_glm53_mhc_prefill_tc(0)
    a = call()
    qc.set_glm53_mhc_prefill_tc(1)
    b = call()
    torch.cuda.synchronize()
    exact(a, b)


@pytest.mark.parametrize("fused", [False, True])
def test_graphs_keep_their_captured_selection_with_changed_inputs(fused):
    data = data_for()
    graphs, output = [], []
    for enabled in (0, 1):
        qc.set_glm53_mhc_prefill_tc(enabled)
        installed(data, fused)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = installed(data, fused)
        graphs.append(graph)
        output.append(out)
    for seed in (943, 944, 945):
        fresh = data_for(seed=seed)
        for target, value in zip(data[:4], fresh[:4]):
            target.copy_(value)
        for enabled in (0, 1):
            # Flip the host selector before replay: it must NOT retroactively
            # change kernels already captured in either graph.
            qc.set_glm53_mhc_prefill_tc(1 - enabled)
            graphs[enabled].replay()
            qc.set_glm53_mhc_prefill_tc(enabled)
            exact(output[enabled], installed(data, fused))
        exact(output[0][:1], output[1][:1])
        assert any(
            not torch.equal(a, b) for a, b in zip(output[0][1:], output[1][1:])
        ), "candidate dispatch was not exercised"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two devices")
@pytest.mark.parametrize("fused", [False, True])
def test_candidate_setup_follows_device_switches(fused):
    qc.set_glm53_mhc_prefill_tc(1)
    reference = None
    for device in (0, 1, 0):
        with torch.cuda.device(device):
            out = installed(data_for(), fused)
            torch.cuda.synchronize()
            host = [v.cpu() for v in out]
        if reference is None:
            reference = host
        exact(reference, host)


@pytest.mark.parametrize("value", [-1, 2])
def test_switch_rejects_invalid_states(value):
    with pytest.raises(RuntimeError, match="0 or 1"):
        qc.set_glm53_mhc_prefill_tc(value)
