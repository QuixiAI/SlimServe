# SPDX-License-Identifier: Apache-2.0
"""Quarantined exact integer-histogram gate; not top-k or serving timing.

Compile the standalone CUDA source with nvcc -O3 -std=c++17 -arch=sm_80
-shared -Xcompiler=-fPIC csrc/quixicore/glm53f_histogram_benchmark.cu -o LIB.
Then run this module with --library LIB only when the GPUs are exclusive.
"""

import argparse
import ctypes
import json
import statistics

import torch


def histogram_oracle(logits, lengths):
    assert logits.device.type == lengths.device.type == "cpu"
    assert logits.ndim == 2 and lengths.shape == (logits.shape[0],)
    result = []
    for row, length in enumerate(lengths.tolist()):
        assert 0 <= length <= logits.shape[1]
        bits = logits[row, :length].half().view(torch.int16).to(torch.int32) & 0xFFFF
        keys = torch.where(bits & 0x8000 != 0, bits, ~bits & 0x7FFF) >> 5
        result.append(torch.bincount(keys.long(), minlength=2048).to(torch.int32))
    return torch.stack(result)


def edge_case_inputs():
    """Exercise float4 tails, partial warps and the 512-thread loop boundary."""
    lengths = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            2043,
            2044,
            2045,
            2046,
            2047,
            2048,
            2049,
            2050,
            4093,
            4094,
            4095,
            4096,
        ],
        dtype=torch.int32,
    )
    special = torch.tensor(
        [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            1.0,
            -1.0,
            65504.0,
            -65504.0,
            2**-24,
            -(2**-24),
            0.999,
            1.001,
        ],
    )
    logits = special.repeat((32 * 4096 + special.numel() - 1) // special.numel())
    logits = logits[: 32 * 4096].reshape(32, 4096).clone()
    for row, length in enumerate(lengths.tolist()):
        logits[row, length:] = float("nan")
    return logits, lengths


@torch.inference_mode()
def validate_edge_cases(launch):
    host_logits, host_lengths = edge_case_inputs()
    logits, lengths = host_logits.cuda(), host_lengths.cuda()
    guards, outputs, graphs = {}, {}, {}

    def run(arm):
        error = launch(
            logits.data_ptr(),
            lengths.data_ptr(),
            outputs[arm].data_ptr(),
            32,
            4096,
            arm,
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"CUDA edge-case launch error {error}")

    for arm in (0, 1):
        guards[arm] = torch.full(
            (32 * 2048 + 64,), -71, dtype=torch.int32, device="cuda"
        )
        outputs[arm] = guards[arm][32:-32].view(32, 2048)
        run(arm)  # Resolve lazy CUDA module loading before graph capture.
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs[arm] = graph
    for replay in range(2):
        if replay:
            host_lengths.zero_()
            lengths.zero_()
            logits.fill_(float("nan"))
        expected = histogram_oracle(host_logits, host_lengths)
        for arm, graph in graphs.items():
            graph.replay()
            torch.testing.assert_close(outputs[arm].cpu(), expected, atol=0, rtol=0)
            assert (guards[arm][:32] == -71).all()
            assert (guards[arm][-32:] == -71).all()
    return dict(
        scope="Native histogram edge-case correctness; not timing",
        rows=32,
        graph_replays=2,
        exact_histograms=True,
        output_guards_intact=True,
        padding_poisoned=True,
    )


@torch.inference_mode()
def measure(launch, rows, visible, distribution, layers, repeats, steps):
    capacity = 262144
    torch.manual_seed(51000 + rows + visible)
    logits = torch.empty(layers, rows, capacity, device="cuda")
    lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
    storage = {
        arm: [
            torch.full((rows * 2048 + 64,), -71, device="cuda", dtype=torch.int32)
            for _ in range(layers)
        ]
        for arm in (0, 1)
    }
    outputs = {
        arm: [s[32:-32].view(rows, 2048) for s in buffers]
        for arm, buffers in storage.items()
    }

    def run(arm):
        for layer in range(layers):
            error = launch(
                logits[layer].data_ptr(),
                lengths.data_ptr(),
                outputs[arm][layer].data_ptr(),
                rows,
                capacity,
                arm,
                torch.cuda.current_stream().cuda_stream,
            )
            if error:
                raise RuntimeError(f"CUDA histogram launch error {error}")

    def refill():
        logits.normal_()
        if distribution == "narrow":
            logits.mul_(1e-5).add_(1.0)
        elif distribution == "constant":
            logits.fill_(1.0)

    refill()
    lengths.fill_(visible)
    for arm in outputs:
        run(arm)
    torch.cuda.synchronize()
    graphs = {}
    for arm in outputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(arm)
        graphs[arm] = graph
    for replay in range(4):
        refill()
        host_lengths = torch.full((rows,), visible, dtype=torch.int32)
        if replay == 1:
            host_lengths[::2] = max(0, visible - 3)
        elif replay == 2:
            host_lengths[::2] = min(513, visible)
        elif replay == 3:
            host_lengths[-1] = 0
        lengths.copy_(host_lengths)
        for row, count in enumerate(host_lengths.tolist()):
            logits[:, row, count:].fill_(float("nan"))
        for graph in graphs.values():
            graph.replay()
        for layer in range(layers):
            expected = histogram_oracle(logits[layer].cpu(), host_lengths)
            for arm in outputs:
                torch.testing.assert_close(
                    outputs[arm][layer].cpu(), expected, atol=0, rtol=0
                )
                assert (storage[arm][layer][:32] == -71).all()
                assert (storage[arm][layer][-32:] == -71).all()
    lengths.fill_(visible)
    refill()
    timings = {"scalar_atomic": [], "warp_aggregated": []}
    for repeat in range(repeats):
        for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(steps):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            name = "warp_aggregated" if arm else "scalar_atomic"
            timings[name].append(begin.elapsed_time(end) * 1000 / steps / layers)
    return dict(
        rows=rows,
        visible_pools=visible,
        capacity_pools=capacity,
        distribution=distribution,
        layers=layers,
        repeats=repeats,
        steps=steps,
        changed_input_and_length_replays=4,
        exact_histograms=True,
        output_guards_intact=True,
        us=timings,
        median_us={k: statistics.median(v) for k, v in timings.items()},
        scope="First FP16-bin histogram only; not full top-k or serving",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if min(args.layers, args.repeats, args.steps) < 1:
        parser.error("positive sizes required")
    library = ctypes.CDLL(args.library)
    launch = library.glm53f_histogram_launch
    launch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    launch.restype = ctypes.c_int
    print(json.dumps(validate_edge_cases(launch)), flush=True)
    for distribution in ("normal", "narrow", "constant"):
        for visible in (256, 32768, 262144):
            for rows in (1, 8, 16, 32):
                print(
                    json.dumps(
                        measure(
                            launch,
                            rows,
                            visible,
                            distribution,
                            args.layers,
                            args.repeats,
                            args.steps,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
