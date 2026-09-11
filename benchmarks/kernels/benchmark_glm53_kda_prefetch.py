#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 2.3: one fixed output-weight prefetch, complete KDA decode window.

Actual TP4 rank0 weights from layers 0/22/44, synthetic merged inputs/states.
51 independent 8 MiB output weights exceed three SM120 L2 capacities. Original
fg_b bmm -> conv -> state-updating KDA -> copy -> gated norm -> FP8 o_proj -> copy.
One side-stream fork per window; one join per graph, not per layer. The input
projection and TP output all-reduce are outside this component-window probe.
Three A/B/A rounds, ten graph replays, M1/8/16; no geometry or budget sweep.
No serving implementation, arithmetic change, installed binary or profile change.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors import safe_open

from benchmarks.kernels.benchmark_glm53_marlin_schedule import measure
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    get_conv_state_layout,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)
from vllm.quixicore.ops import quixicore_ops
from vllm.third_party.flash_linear_attention.ops.kda import rms_norm_gated


@triton.jit
def prefetch_output_weight(weight):
    # #576 mechanism: read-only cache hints, no value or synchronization output.
    # 16 CTAs x 128 lanes x 4 KiB covers exactly the contiguous 8 MiB FP8 weight.
    offset = (tl.program_id(0) * 128 + tl.arange(0, 128)) * 4096
    tl.inline_asm_elementwise(
        """{
        .reg .b64 policy;
        createpolicy.fractional.L2::evict_last.b64 policy, 1.0;
        cp.async.bulk.prefetch.L2.global.L2::cache_hint [$1], 4096, policy;
        mov.u32 $0, 0;
        }""",
        constraints="=r,l",
        args=[weight + offset],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


def load_weights(model, layer):
    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.self_attn."

    def get(name, artifact=None):
        key = prefix + name
        with safe_open(
            model / (artifact or index[key]), framework="pt", device="cpu"
        ) as f:
            return f.get_tensor(key)

    weight = get("o_proj.weight", "fp8-swapset.safetensors")[:, :2048].contiguous()
    scale = get("o_proj.weight_scale", "fp8-swapset.safetensors")[:, :16].contiguous()
    assert weight.shape == (4096, 2048) and weight.dtype == torch.float8_e4m3fn
    assert scale.shape == (32, 16) and scale.dtype == torch.float32
    return {
        "weight": weight,
        "scale": scale,
        "fg": torch.stack([get(f"{p}_b_proj.weight")[:2048] for p in ("f", "g")]),
        "conv": torch.cat(
            [get(f"{p}_conv1d.weight")[:2048].reshape(2048, 4) for p in ("q", "k", "v")]
        ),
        "norm": get("o_norm.weight"),
        "a_log": get("A_log", "f32-overrides.safetensors")[:16].contiguous(),
        "dt_bias": get("dt_bias", "f32-overrides.safetensors")[:2048].contiguous(),
    }


def window(weights, inputs, state, indices, output):
    batch = inputs.shape[0]
    mixed, beta, fg_a = inputs.split([6144, 128, 256], dim=-1)
    fg_a = fg_a.view(batch, 2, 128).transpose(0, 1)
    fg_b = torch.bmm(fg_a, weights["fg"].transpose(1, 2))
    g1, g2 = fg_b.unbind(0)
    conv_state, recurrent_state = state
    mixed = causal_conv1d_update(
        mixed,
        conv_state,
        weights["conv"],
        activation="silu",
        conv_state_indices=indices,
        validate_data=True,
        out=torch.empty_like(mixed),
    )
    core, _ = fused_recurrent_kda_packed_decode(
        mixed,
        g1.view(1, batch, 16, 128),
        beta[:, :16].unsqueeze(0),
        weights["a_log"],
        weights["dt_bias"],
        -5.0,
        recurrent_state,
        indices,
    )
    # Match the opaque attention op's placement into the caller-owned buffer.
    placed = torch.empty_like(core)
    placed.copy_(core)
    # FusedRMSNormGated defaults to 1e-5 in the actual KDA constructor.
    placed.copy_(
        rms_norm_gated(
            placed,
            g2.view(batch, 16, 128),
            weights["norm"],
            None,
            activation="sigmoid",
            eps=1e-5,
        )
    )
    output.copy_(
        quixicore_ops.decode_gemm_fp8(
            placed.view(batch, 2048),
            weights["weight"],
            weights["scale"],
        )
    )


def capture(banks, inputs, states, indices, outputs, prefetch):
    compute, side = torch.cuda.Stream(), torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())

    def run():
        for weights, x, state, out in zip(banks, inputs, states, outputs):
            if prefetch:
                side.wait_stream(compute)
                with torch.cuda.stream(side):
                    prefetch_output_weight[(16,)](weights["weight"], num_warps=4)
            window(weights, x, state, indices, out)
        if prefetch:
            compute.wait_stream(side)

    with torch.cuda.stream(compute):
        run()
    torch.cuda.current_stream().wait_stream(compute)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=compute):
        run()
    return graph


def reset_states(initial, states):
    for src, dst in zip(initial, states):
        for sv, dv in zip(src, dst):
            dv.copy_(sv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("a NEW output is required")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if active:
        raise RuntimeError(f"GPUs busy: {active}")
    assert torch.cuda.get_device_capability() == (12, 0)
    assert get_conv_state_layout() == "SD"
    assert MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto") == (
        torch.bfloat16,
        torch.float32,
    )
    torch.set_num_threads(4)
    torch.manual_seed(2301)
    cpu = [load_weights(args.model, layer) for layer in (0, 22, 44)]
    banks = [
        {name: value.cuda() for name, value in w.items()}
        for _ in range(17)
        for w in cpu
    ]
    assert len({w["weight"].data_ptr() for w in banks}) == 51
    result = {
        "status": "running",
        "roadmap": "2.3",
        "method": __doc__,
        "recipe": "glm53-redhatai-nvfp4-fp8-kda-tp4-v1",
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weight_sha256": [
            {
                k: hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest()
                for k, v in w.items()
            }
            for w in cpu
        ],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "state_dtype": "float32",
        "output_weight_bytes": 51 * 8 * 1024 * 1024,
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        kernel = prefetch_output_weight[(16,)](banks[0]["weight"], num_warps=4)
        assert "cp.async.bulk.prefetch.L2.global.L2::cache_hint" in kernel.asm["ptx"]
        args.output.with_suffix(".prefetch.ptx").write_text(kernel.asm["ptx"])
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
            graphs = [
                capture(banks, inputs, states[i], indices, outputs[i], bool(i))
                for i in range(2)
            ]

            # Both initial and changed-input replays must match every output/state bit.
            for check in range(2):
                if check:
                    for x in inputs:
                        x.mul_(0.5)
                    indices.copy_(indices.flip(0))
                for i in range(2):
                    reset_states(initial, states[i])
                    graphs[i].replay()
                for a, b in zip(outputs[0], outputs[1]):
                    assert torch.isfinite(a).all().item()
                    torch.testing.assert_close(b, a, rtol=0, atol=0)
                for a, b in zip(states[0], states[1]):
                    for av, bv in zip(a, b):
                        assert torch.isfinite(av).all().item()
                        torch.testing.assert_close(bv, av, rtol=0, atol=0)
            samples = []
            for _ in range(3):
                sample = {}
                for name, i in (("before_us", 0), ("candidate_us", 1), ("after_us", 0)):
                    reset_states(initial, states[i])
                    sample[name] = measure(graphs[i], len(banks), 10)
                samples.append(sample)
            row = {
                "batch": batch,
                "exact_output_state_checks": 102,
                "samples": samples,
                "median_us": {
                    k: statistics.median(s[k] for s in samples) for k in samples[0]
                },
                "conv_stride": list(initial[0][0].stride()),
            }
            result["cases"].append(row)
            print(json.dumps(row), flush=True)
            save()
            del graphs, outputs, states, initial, inputs
        result["status"] = "complete"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
