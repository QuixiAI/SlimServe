# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only reuse of the serving TP communicator for row-shard gather.

No production dispatch changes. Compare against the ProcessGroupNCCL serving
candidate, and measure memory in separate fresh processes before retention.
"""

import torch

from vllm.distributed import get_tp_group
from vllm.model_executor.layers.glm5_next_indexer import (
    POOL_CACHE_HEAD_DIM,
    _expand_topk_kernel,
    _pooled_select,
    _pooled_topk,
)


def _pooled_select_with_gather(
    q, weights, ape, cache, block_table, row_req, visible, logits,
    max_pools, block_size, softmax_scale, ksel, topk_out, kp,
    adaptive_score=False, row_shard=False, existing_tp=True,
):
    if not row_shard:
        return _pooled_select(
            q, weights, ape, cache, block_table, row_req, visible, logits,
            max_pools, block_size, softmax_scale, ksel, topk_out, kp,
            adaptive_score=adaptive_score,
        )
    rows = q.shape[0]
    group = get_tp_group()
    assert group.world_size == 8 and 0 <= group.rank_in_group < 8
    assert rows in (16, 32) and q.shape[1:] == (32, 128)
    assert cache.shape[-1] == POOL_CACHE_HEAD_DIM and kp == 4 and ksel == 512
    assert not adaptive_score
    # Require a working owned communicator: a silent fallback to the original
    # torch.distributed path would invalidate the intended experiment.
    pynccl = None
    if existing_tp:
        communicator = group.device_communicator
        assert communicator is not None
        communicator.wait_for_comm_init()
        pynccl = communicator.pynccl_comm
        assert pynccl is not None and not pynccl.disabled
    local_rows = rows // group.world_size
    lo = group.rank_in_group * local_rows
    hi = lo + local_rows
    local_sel = _pooled_topk(
        q[lo:hi], weights[lo:hi], ape, cache, block_table,
        row_req[lo:hi], visible[lo:hi], logits[:local_rows],
        max_pools, block_size, softmax_scale, ksel, kp,
    )
    sel = torch.empty((rows, ksel), dtype=torch.int32, device=q.device)
    # Live runtime lookup, original caller stream, existing communicator.
    # Never put its native handle into the AOT graph or a persistent cache.
    if existing_tp:
        pynccl.all_gather(sel, local_sel)
    else:
        torch.distributed.all_gather_into_tensor(
            sel, local_sel, group=group.device_group
        )
    _expand_topk_kernel[(rows,)](
        sel, visible, topk_out, sel.stride(0),
        KP=kp, KSEL=ksel, OUT_W=topk_out.shape[1], BLOCK_S=64,
    )


def pooled_select_existing_tp(*args, **kwargs):
    return _pooled_select_with_gather(*args, **kwargs, existing_tp=True)


def pooled_select_process_group(*args, **kwargs):
    """Historical separate-communicator control, independent of serving edits."""
    return _pooled_select_with_gather(*args, **kwargs, existing_tp=False)
