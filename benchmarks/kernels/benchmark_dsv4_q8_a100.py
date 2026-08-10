# SPDX-License-Identifier: Apache-2.0
"""Validate and time packed Q8_0 decode GEMV at DSV4 projection shapes."""

from __future__ import annotations

import argparse

import torch

from vllm.model_executor.layers.quantization.gguf import ops


def run_shape(
    rows: int,
    cols: int,
    tokens: int,
    repetitions: int,
    warmup: int,
    rotation: int,
    device: torch.device,
) -> None:
    if cols % 32:
        raise ValueError("Q8_0 columns must be divisible by 32")
    blocks_per_row = cols // 32
    packed = torch.empty(
        (rows, blocks_per_row, 34), dtype=torch.uint8, device=device
    )
    scales = (
        torch.rand((rows, blocks_per_row), dtype=torch.float32, device=device)
        .mul_(0.015)
        .add_(0.001)
        .to(torch.float16)
    )
    packed[:, :, :2].copy_(scales.view(torch.uint8).reshape(rows, blocks_per_row, 2))
    codes = torch.randint(
        -127,
        128,
        (rows, blocks_per_row, 32),
        dtype=torch.int8,
        device=device,
    )
    packed[:, :, 2:].copy_(codes.view(torch.uint8))
    weight = packed.reshape(rows, -1)
    x = torch.randn((tokens, cols), dtype=torch.bfloat16, device=device)

    # Per-token dispatch is the exact serving reference. The generic
    # multi-column path has historically hidden cross-token indexing bugs.
    reference = torch.cat(
        [ops.ggml_mul_mat_vec_a8(weight, x[i : i + 1], 8, rows) for i in range(tokens)]
    )
    generic = ops.ggml_mul_mat_vec_a8(weight, x, 8, rows)
    generic_exact = torch.equal(generic, reference)
    generic_max_error = (generic.float() - reference.float()).abs().max().item()
    aligned = ops.ggml_dsv4_repack_q8_0_aligned(weight)
    if not torch.equal(aligned[:1, : blocks_per_row * 2].view(torch.float16), scales[:1]):
        raise SystemExit("aligned Q8_0 scale repack failed")
    aligned_ring = [aligned]
    aligned_ring.extend(aligned.clone() for _ in range(rotation - 1))
    quant_x = ops.ggml_quantize_q8_1(x)

    strategies = (1, 2, 4)
    outputs = {
        rows_per_cta: ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
            aligned_ring[0], x, quant_x, rows, rows_per_cta
        )
        for rows_per_cta in strategies
    }
    exact = {key: torch.equal(value, reference) for key, value in outputs.items()}
    max_error = {
        key: (value.float() - reference.float()).abs().max().item()
        for key, value in outputs.items()
    }

    def time_us(callable_):
        for _ in range(warmup):
            callable_()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            callable_()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / repetitions

    generic_us = time_us(lambda: ops.ggml_mul_mat_vec_a8(weight, x, 8, rows))
    aligned_us = {
        key: time_us(
            lambda key=key: ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
                aligned, x, None, rows, key
            )
        )
        for key in strategies
    }
    aligned_prequant_us = {}
    for key in strategies:
        ring_index = 0

        def rotating_call(key=key):
            nonlocal ring_index
            result = ops.ggml_dsv4_mul_mat_vec_aligned_q8_0(
                aligned_ring[ring_index], x, quant_x, rows, key
            )
            ring_index = (ring_index + 1) % rotation
            return result

        aligned_prequant_us[key] = time_us(rotating_call)
    packed_gb = weight.numel() / 1e9
    print(
        f"rows={rows:>6} cols={cols:>5} tokens={tokens:>2} "
        f"generic_us={generic_us:8.3f} "
        + " ".join(
            f"aligned_r{key}_us={aligned_us[key]:8.3f} "
            f"prequant_rot_r{key}_us={aligned_prequant_us[key]:8.3f} "
            f"speedup_r{key}={generic_us / aligned_us[key]:6.3f} "
            f"exact_r{key}={exact[key]} max_error_r{key}={max_error[key]:.6f}"
            for key in strategies
        )
        + f" rotation={rotation} generic_exact={generic_exact} "
        f"generic_max_error={generic_max_error:.6f} "
        f"weight_gbps={packed_gb / (min(aligned_us.values()) / 1e6):8.1f}"
    )
    if not all(exact.values()):
        raise SystemExit("aligned Q8_0 GEMV is not exact")


def run_shared_shape(
    intermediate: int,
    cols: int,
    tokens: int,
    repetitions: int,
    warmup: int,
    device: torch.device,
) -> None:
    rows = 2 * intermediate
    blocks_per_row = cols // 32
    packed = torch.empty(
        (rows, blocks_per_row, 34), dtype=torch.uint8, device=device
    )
    scales = (
        torch.rand((rows, blocks_per_row), dtype=torch.float32, device=device)
        .mul_(0.015)
        .add_(0.001)
        .to(torch.float16)
    )
    packed[:, :, :2].copy_(
        scales.view(torch.uint8).reshape(rows, blocks_per_row, 2)
    )
    codes = torch.randint(
        -127,
        128,
        (rows, blocks_per_row, 32),
        dtype=torch.int8,
        device=device,
    )
    packed[:, :, 2:].copy_(codes.view(torch.uint8))
    weight = packed.reshape(rows, -1)
    x = torch.randn((tokens, cols), dtype=torch.bfloat16, device=device)
    def reference_call() -> torch.Tensor:
        gate_up = ops.ggml_mul_mat_vec_a8(weight, x, 8, rows)
        activated = torch.empty(
            (tokens, intermediate), dtype=x.dtype, device=device
        )
        torch.ops._C.silu_and_mul_with_clamp(
            activated, gate_up, 7.0, 1.0, 0.0
        )
        return activated

    reference = reference_call()
    output = ops.ggml_dsv4_shared_gate_up_swiglu(weight, x, 7.0)
    error = (output.float() - reference.float()).abs()
    quant_reference = ops.ggml_quantize_q8_1(output)
    packed_output, packed_quant = ops.ggml_dsv4_shared_gate_up_swiglu_q8_1(
        weight, x, 7.0
    )
    packed_output_exact = torch.equal(packed_output, output)
    packed_quant_exact = torch.equal(packed_quant, quant_reference)

    down_rows = 4096
    down_blocks = intermediate // 32
    down_packed = torch.empty(
        (down_rows, down_blocks, 34), dtype=torch.uint8, device=device
    )
    down_scales = (
        torch.rand((down_rows, down_blocks), dtype=torch.float32, device=device)
        .mul_(0.015)
        .add_(0.001)
        .to(torch.float16)
    )
    down_packed[:, :, :2].copy_(
        down_scales.view(torch.uint8).reshape(down_rows, down_blocks, 2)
    )
    down_packed[:, :, 2:].copy_(
        torch.randint(
            -127,
            128,
            (down_rows, down_blocks, 32),
            dtype=torch.int8,
            device=device,
        ).view(torch.uint8)
    )
    down_weight = down_packed.reshape(down_rows, -1)

    def shared_down_reference() -> torch.Tensor:
        activated = ops.ggml_dsv4_shared_gate_up_swiglu(weight, x, 7.0)
        return ops.ggml_mul_mat_vec_a8(down_weight, activated, 8, down_rows)

    def shared_down_packed() -> torch.Tensor:
        activated, quant_activated = (
            ops.ggml_dsv4_shared_gate_up_swiglu_q8_1(weight, x, 7.0)
        )
        return ops.ggml_mul_mat_vec_prequant_a8(
            down_weight, activated, quant_activated, 8, down_rows
        )

    down_reference = shared_down_reference()
    down_output = shared_down_packed()
    down_exact = torch.equal(down_output, down_reference)

    def time_us(callable_):
        for _ in range(warmup):
            callable_()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repetitions):
            callable_()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0 / repetitions

    reference_us = time_us(reference_call)
    fused_us = time_us(
        lambda: ops.ggml_dsv4_shared_gate_up_swiglu(weight, x, 7.0)
    )
    fused_then_quant_us = time_us(
        lambda: ops.ggml_quantize_q8_1(
            ops.ggml_dsv4_shared_gate_up_swiglu(weight, x, 7.0)
        )
    )
    fused_packed_us = time_us(
        lambda: ops.ggml_dsv4_shared_gate_up_swiglu_q8_1(weight, x, 7.0)
    )
    down_reference_us = time_us(shared_down_reference)
    down_packed_us = time_us(shared_down_packed)
    print(
        f"shared_i={intermediate:>4} cols={cols:>5} tokens={tokens:>2} "
        f"reference_us={reference_us:8.3f} fused_us={fused_us:8.3f} "
        f"fused_quant_us={fused_then_quant_us:8.3f} "
        f"packed_us={fused_packed_us:8.3f} "
        f"packed_speedup={fused_then_quant_us / fused_packed_us:6.3f} "
        f"down_ref_us={down_reference_us:8.3f} "
        f"down_packed_us={down_packed_us:8.3f} "
        f"down_speedup={down_reference_us / down_packed_us:6.3f} "
        f"max_error={error.max().item():.6f} "
        f"output_exact={packed_output_exact} quant_exact={packed_quant_exact} "
        f"down_exact={down_exact}"
    )
    if error.max().item() > 0.25:
        raise SystemExit("fused Q8_0 shared-expert correctness failed")
    if not packed_output_exact or not packed_quant_exact or not down_exact:
        raise SystemExit("packed shared-expert output is not exact")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument(
        "--rotation",
        type=int,
        default=8,
        help="Number of byte-identical aligned weights cycled while timing",
    )
    parser.add_argument(
        "--shape",
        nargs=2,
        type=int,
        action="append",
        metavar=("ROWS", "COLS"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rotation < 1:
        raise ValueError("--rotation must be at least 1")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    if args.shared:
        for intermediate in (1024, 512):
            run_shared_shape(
                intermediate,
                4096,
                args.tokens,
                args.repetitions,
                args.warmup,
                device,
            )
        return
    shapes = args.shape or [
        (1536, 4096),
        (8192, 1024),
        (2048, 4096),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
    ]
    for rows, cols in shapes:
        run_shape(
            rows,
            cols,
            args.tokens,
            args.repetitions,
            args.warmup,
            args.rotation,
            device,
        )


if __name__ == "__main__":
    main()
