# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only split launches for wide-query and independent-row scoring.

The predicates are complementary and are evaluated from device row mappings
on every graph replay. No CPU readback, persistent row list, or serving edits.
"""

from benchmarks.glm5_next_pool_query_tile_candidate import (
    _independent_row,
    _query_tile,
    validate_query_tile_inputs,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _mixed_rows(
    Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, ROWS, MAX_POOLS,
    BT_STRIDE, PAGE_STRIDE, SCALE, BS: tl.constexpr, BQ: tl.constexpr,
):
    r = tl.program_id(0)
    row0 = r // BQ * BQ
    rows = row0 + tl.arange(0, BQ)
    live = rows < ROWS
    requests = tl.load(ROW_REQ + rows, live, -1)
    req0 = tl.load(ROW_REQ + row0)
    same = tl.sum(((requests == req0) | ~live).to(tl.int32)) == BQ
    if not same:
        _independent_row(Q, W, CACHE, BT, ROW_REQ, VISIBLE, OUT, r,
                         MAX_POOLS, BT_STRIDE, PAGE_STRIDE, SCALE, BS)


def split_query_pool_logits(q, weights, cache, block_table, row_req, visible, out,
                            query_tile=4, programs=128):
    validate_query_tile_inputs(q, weights, cache, block_table, row_req, visible, out,
                               query_tile, programs)
    rows = q.shape[0]
    if rows == 0:
        return
    active_programs = min(programs, triton.cdiv(out.shape[1], 16))
    args = (q, weights, cache, block_table, row_req, visible, out, rows,
            out.shape[1], block_table.stride(0), cache.stride(0), 128**-0.5,
            cache.shape[1], query_tile)
    _query_tile[(triton.cdiv(rows, query_tile), active_programs)](
        *args, FALLBACK=False, num_warps=4,
    )
    # One row per CTA: mixed requests do not inherit the wide tile's shared
    # memory footprint or serialize BQ independent rows in one CTA.
    _mixed_rows[(rows, active_programs)](*args, num_warps=4)


def compile_only(block_size=4608):
    import json

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    signature = dict(Q="*bf16", W="*fp32", CACHE="*bf16", BT="*i32",
                     ROW_REQ="*i32", VISIBLE="*i32", OUT="*fp32", ROWS="i32",
                     MAX_POOLS="i32", BT_STRIDE="i32", PAGE_STRIDE="i32",
                     SCALE="fp32")
    for tile in (2, 4):
        for name, function in (("wide", _query_tile), ("mixed", _mixed_rows)):
            constants = dict(BS=block_size, BQ=tile)
            if name == "wide":
                constants["FALLBACK"] = False
            kernel = triton.compile(
                ASTSource(function, signature, constexprs=constants),
                target=GPUTarget("cuda", 80, 32), options=dict(num_warps=4),
            )
            print(json.dumps(dict(query_tile=tile, phase=name, block_size=block_size,
                                  shared_bytes=kernel.metadata.shared,
                                  scope="Offline SM80 compile only")), flush=True)


if __name__ == "__main__":
    compile_only()
