#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 4.2: one full-head conv/KDA/norm fusion, complete-window comparison.

Reuses the completed prefetch experiment's actual-weight windows, without any
prefetch. 51 allocations exceed 3x L2. Fixed eight-warps/full-head ownership;
no tile sweep. FP32 recurrent state, BF16 conv state, original projections.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.benchmark_glm53_kda_prefetch import (
    load_weights,
    reset_states,
    window,
)
from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from benchmarks.kernels.glm53_kda_core import core
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)
from vllm.quixicore.ops import quixicore_ops


def fused_window(weights, inputs, state, indices, output):
    batch = inputs.shape[0]
    mixed, beta, fg_a = inputs.split([6144, 128, 256], dim=-1)
    fg_a = fg_a.view(batch, 2, 128).transpose(0, 1)
    g1, g2 = torch.bmm(fg_a, weights["fg"].transpose(1, 2)).unbind(0)
    normalized = torch.empty(batch, 2048, dtype=inputs.dtype, device=inputs.device)
    kernel = core(mixed, g1, g2, beta, weights, *state, indices, normalized)
    output.copy_(
        quixicore_ops.decode_gemm_fp8(
            normalized,
            weights["weight"],
            weights["scale"],
        )
    )
    return kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be NEW")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        parser.error(f"GPUs busy: {active}")
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(4201)
    cpu = [load_weights(args.model, layer) for layer in (0, 22, 44)]
    banks = [
        {name: value.cuda() for name, value in w.items()}
        for _ in range(17)
        for w in cpu
    ]
    result = {
        "status": "running",
        "roadmap": "4.2",
        "diagnostic_only": True,
        "method": __doc__,
        "recipe": "glm53-redhatai-nvfp4-fp8-kda-tp4-v1",
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_sha256": hashlib.sha256(
            Path(__file__).with_name("glm53_kda_core.py").read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for batch in (1, 8, 16):
            indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
            inputs = [
                torch.randn(batch, 6528, device="cuda", dtype=torch.bfloat16)
                for _ in banks
            ]
            initial = [
                (
                    torch.randn(
                        batch + 1, 3, 6144, device="cuda", dtype=torch.bfloat16
                    ).transpose(-1, -2),
                    torch.randn(batch + 1, 16, 128, 128, device="cuda") * 0.1,
                )
                for _ in banks
            ]
            states = [
                [tuple(s.clone() for s in pair) for pair in initial] for _ in range(2)
            ]
            outputs = [
                [
                    torch.empty(batch, 4096, device="cuda", dtype=torch.bfloat16)
                    for _ in banks
                ]
                for _ in range(2)
            ]
            graphs = []
            kernel = None
            for arm, fn in enumerate((window, fused_window)):
                for weights, x, state, out in zip(
                    banks, inputs, states[arm], outputs[arm]
                ):
                    found = fn(weights, x, state, indices, out)
                    if found is not None:
                        kernel = found
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for weights, x, state, out in zip(
                        banks, inputs, states[arm], outputs[arm]
                    ):
                        fn(weights, x, state, indices, out)
                graphs.append(graph)
            checks = []
            args.output.with_suffix(".core.ptx").write_text(kernel.asm["ptx"])
            for changed in (False, True):
                if changed:
                    for x in inputs:
                        x.mul_(0.5)
                    indices.copy_(indices.flip(0))
                    indices[0] = 0  # null slot: neither state may be written
                for arm in range(2):
                    reset_states(initial, states[arm])
                    graphs[arm].replay()
                torch.cuda.synchronize()
                for index, (a, b) in enumerate(zip(outputs[0], outputs[1])):
                    torch.testing.assert_close(b, a, rtol=0.01, atol=0.01)
                    conv_a, state_a = states[0][index]
                    conv_b, state_b = states[1][index]
                    torch.testing.assert_close(conv_b, conv_a, rtol=0, atol=0)
                    try:
                        torch.testing.assert_close(
                            state_b, state_a, rtol=1e-4, atol=1e-5
                        )
                    except AssertionError:
                        mixed, beta, fg_a = inputs[index].split(
                            [6144, 128, 256], dim=-1
                        )
                        g1, g2 = torch.bmm(
                            fg_a.view(batch, 2, 128).transpose(0, 1),
                            banks[index]["fg"].transpose(1, 2),
                        ).unbind(0)
                        conv_src, state_src = initial[index]
                        conv_ref = causal_conv1d_update(
                            mixed,
                            conv_src.clone(),
                            banks[index]["conv"],
                            activation="silu",
                            conv_state_indices=indices,
                            validate_data=True,
                            out=torch.empty_like(mixed),
                        )
                        conv_debug = torch.zeros_like(mixed)
                        debug_state = state_src.clone()
                        reference_state = state_src.clone()
                        fused_recurrent_kda_packed_decode(
                            conv_ref,
                            g1.view(1, batch, 16, 128),
                            beta[:, :16].unsqueeze(0),
                            banks[index]["a_log"],
                            banks[index]["dt_bias"],
                            -5.0,
                            reference_state,
                            indices,
                        )
                        debug_kernel = core(
                            mixed,
                            g1,
                            g2,
                            beta,
                            banks[index],
                            conv_src.clone(),
                            debug_state,
                            indices,
                            torch.empty(
                                batch, 2048, device="cuda", dtype=torch.bfloat16
                            ),
                            debug=conv_debug,
                        )
                        torch.cuda.synchronize()
                        result["failure_fixture"] = {
                            "batch": batch,
                            "index": index,
                            "changed": changed,
                            "conv_output_exact": torch.equal(conv_debug, conv_ref),
                            "conv_max_abs": (conv_debug.float() - conv_ref.float())
                            .abs()
                            .max()
                            .item(),
                            "debug_state_max_abs": (debug_state - reference_state)
                            .abs()
                            .max()
                            .item(),
                        }
                        torch.save(
                            {
                                "conv_reference": conv_ref.cpu(),
                                "conv_candidate": conv_debug.cpu(),
                                "state_reference": state_a.cpu(),
                                "state_candidate": state_b.cpu(),
                                "state_eager_reference": reference_state.cpu(),
                                "state_eager_debug": debug_state.cpu(),
                                "initial_state": state_src.cpu(),
                                "raw_gate": g1.cpu(),
                                "beta": beta.cpu(),
                                "a_log": banks[index]["a_log"].cpu(),
                                "dt_bias": banks[index]["dt_bias"].cpu(),
                            },
                            args.output.with_suffix(".failure.pt"),
                        )
                        args.output.with_suffix(".ptx").write_text(
                            debug_kernel.asm["ptx"]
                        )
                        raise
                    checks.append(
                        {
                            "fixture": index,
                            "changed_input_null": changed,
                            "output_max_abs": (a.float() - b.float())
                            .abs()
                            .max()
                            .item(),
                            "state_max_abs": (state_a - state_b).abs().max().item(),
                            "conv_exact": torch.equal(conv_a, conv_b),
                        }
                    )
            # Null-slot checks must not turn the timing workload into no-ops.
            indices.copy_(torch.arange(1, batch + 1, device="cuda", dtype=torch.int32))
            samples = []
            for _ in range(3):
                sample = {}
                for name, arm in (("before", 0), ("candidate", 1), ("after", 0)):
                    reset_states(initial, states[arm])
                    sample[name] = measure(graphs[arm], len(banks), 10)
                samples.append(sample)
            row = {
                "batch": batch,
                "checks": checks,
                "samples_us": samples,
                "median_us": {
                    k: statistics.median(s[k] for s in samples) for k in samples[0]
                },
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
            }
            result["cases"].append(row)
            print({k: v for k, v in row.items() if k != "checks"}, flush=True)
            save()
            del graphs, outputs, states, initial, inputs
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
