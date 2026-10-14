# SPDX-License-Identifier: Apache-2.0
"""Selection-only score elision; benchmark-only, never a raw-logits API.

Native prefill top-k returns 0..count-1 followed by -1 for count <= 512,
without reading logits (sampler.cu topKPerRowJob, multipleBlocksPerRow=false).
Retain native selection and expansion; skip only provably unused scores.
"""

import torch

from benchmarks.glm5_next_pool_query_tile_candidate import (
    _independent_row,
    validate_query_tile_inputs,
)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.glm5_next_indexer import _expand_topk_kernel
from vllm.triton_utils import tl, triton


@triton.jit
def _score_nontrivial_rows(
    Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, MAX_POOLS,
    BT_STRIDE, PAGE_STRIDE, SCALE, BS: tl.constexpr,
):
    r = tl.program_id(0)
    if tl.load(VISIBLE + r) // 4 > 512:
        _independent_row(Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, r,
                         MAX_POOLS, BT_STRIDE, PAGE_STRIDE, SCALE, BS)


def select_without_trivial_scores(q, weights, cache, block_table, row_req, visible,
                                 logits, selected, expanded, programs=128):
    """Only selected/expanded are outputs; trivial-row logits stay untouched."""
    validate_query_tile_inputs(q, weights, cache, block_table, row_req, visible,
                               logits, 2, programs)
    rows = q.shape[0]
    assert selected.shape == (rows, 512) and selected.dtype == torch.int32
    assert expanded.shape == (rows, 2080) and expanded.dtype == torch.int32
    assert selected.is_contiguous() and expanded.is_contiguous()
    if not rows:
        return
    _score_nontrivial_rows[(rows, min(programs, triton.cdiv(logits.shape[1], 16)))](
        q, weights, cache, block_table, row_req, visible, logits, logits.shape[1],
        block_table.stride(0), cache.stride(0), 128**-0.5, cache.shape[1],
        num_warps=4,
    )
    counts = torch.div(visible, 4, rounding_mode="floor").to(torch.int32)
    zeros = torch.zeros_like(counts)
    ops.top_k_per_row_prefill(logits, zeros, counts, selected, rows,
                             logits.stride(0), logits.stride(1), 512)
    _expand_topk_kernel[(rows,)](selected, visible, expanded, selected.stride(0),
                                KP=4, KSEL=512, OUT_W=2080, BLOCK_S=64)


def compile_only():
    import json

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(Q="*bf16", W="*fp32", CACHE="*bf16", BT="*i32",
                     ROW_REQ="*i32", VISIBLE="*i32", OUT="*fp32", MAX_POOLS="i32",
                     BT_STRIDE="i32", PAGE_STRIDE="i32", SCALE="fp32")
    for bs in (64, 576, 4608):
        kernel = triton.compile(
            ASTSource(_score_nontrivial_rows, signature, constexprs=dict(BS=bs)),
            target=GPUTarget("cuda", 80, 32), options=dict(num_warps=4),
        )
        print(json.dumps(dict(block_size=bs, shared_bytes=kernel.metadata.shared,
                              scope="Offline SM80 compile only; no GPU parity")),
              flush=True)


if __name__ == "__main__":
    compile_only()
