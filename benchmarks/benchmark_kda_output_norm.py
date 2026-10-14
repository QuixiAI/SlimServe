#!/usr/bin/env python3
"""Isolated CUDA-graph timing; includes an identical input reset in both arms."""

import json
import sys
from contextlib import redirect_stdout

import torch
from triton.testing import do_bench_cudagraph

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _apply_kda_output_norm,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


@torch.no_grad()
def main():
    torch.manual_seed(42)
    results = []
    for heads in (8, 16):
        for tokens in (1, 8, 16, 256):
            norm = FusedRMSNormGated(
                128, activation="sigmoid", device="cuda", dtype=torch.bfloat16
            )
            norm.weight.fill_(1)
            original = torch.randn(
                1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16
            )
            x = torch.empty_like(original)
            gate = torch.randn_like(original)

            def native(x=x, original=original, norm=norm, gate=gate):
                x.copy_(original)
                x.copy_(norm.forward_native(x, gate))

            def fused(x=x, original=original, norm=norm, gate=gate):
                x.copy_(original)
                _apply_kda_output_norm(norm, x, gate)

            native()
            expected = x.clone()
            fused()
            torch.testing.assert_close(x, expected, atol=0.008, rtol=0.008)
            torch.cuda.synchronize()
            results.append(
                dict(
                    heads=heads,
                    tokens=tokens,
                    native_us=1000 * do_bench_cudagraph(native, rep=20),
                    fused_us=1000 * do_bench_cudagraph(fused, rep=20),
                )
            )
    return results


if __name__ == "__main__":
    # VllmConfig logs initialization on stdout by default. Keep JSON clean.
    with redirect_stdout(sys.stderr), set_current_vllm_config(VllmConfig()):
        results = main()
    print(json.dumps(results, indent=2))
