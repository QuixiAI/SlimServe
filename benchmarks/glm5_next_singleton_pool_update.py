# SPDX-License-Identifier: Apache-2.0
"""Offline compiler driver for the owned singleton pool-update candidate."""

from vllm.model_executor.layers.glm5_next_pool_cache import (
    _singleton_pool_update as _singleton_update,
)
from vllm.model_executor.layers.glm5_next_pool_cache import (
    update_pool_cache,
)


def singleton_update(packed, slots, ape, cache):
    update_pool_cache(packed, slots, ape, cache, singleton_fused=True)


def compile_only():
    import json

    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(
        SRC="*bf16", CACHE="*bf16", SLOTS="*i64", APE="*fp32", PAGE_STRIDE="i32"
    )
    for bs in (64, 4608):
        kernel = triton.compile(
            ASTSource(_singleton_update, signature, constexprs=dict(BS=bs)),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 4},
        )
        print(
            json.dumps(
                dict(block_size=bs, shared=kernel.metadata.shared, hash=kernel.hash)
            ),
            flush=True,
        )


if __name__ == "__main__":
    compile_only()
