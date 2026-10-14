# SPDX-License-Identifier: Apache-2.0
"""Offline compiler driver for the owned adaptive pool-score candidate."""

from vllm.model_executor.layers.glm5_next_pool_score import (
    _adaptive_pool_logits,
    adaptive_pool_logits,  # noqa: F401 (old benchmark import compatibility)
)


def compile_only():
    import json

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(
        Q="*bf16",
        W="*fp32",
        CACHE="*bf16",
        BT="*i32",
        ROW_REQ="*i32",
        VISIBLE="*i32",
        OUT="*fp32",
        MAX_POOLS="i32",
        BT_STRIDE="i32",
        PAGE_STRIDE="i32",
        SCALE="fp32",
    )
    for rows in (1, 8, 16, 32):
        constants = dict(
            BS=4608,
            H=32,
            SHORT_BP=16 if rows < 16 else 32 if rows < 32 else 64,
            SHORT_PROGRAMS=128 if rows == 1 else 32,
            LONG_PROGRAMS=64 if rows >= 32 else 128,
        )
        kernel = triton.compile(
            ASTSource(_adaptive_pool_logits, signature, constexprs=constants),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4},
        )
        print(
            json.dumps(
                dict(rows=rows, shared=kernel.metadata.shared, hash=kernel.hash)
            ),
            flush=True,
        )


if __name__ == "__main__":
    compile_only()
