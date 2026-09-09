# SPDX-License-Identifier: Apache-2.0
"""Quarantined ordering-only diagnostic; no score access or selection changes."""

import triton
import triton.language as tl


@triton.jit
def _sort_selected_pools(Indices):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    selected = tl.load(Indices + row * 512 + col)
    # Pool IDs are nonnegative and bounded by context, far below INT_MAX.
    # Only the native -1 padding sentinel is moved behind the valid IDs.
    keys = tl.where(selected == -1, 2147483647, selected)
    ordered = tl.sort(keys, descending=False)
    result = tl.where(ordered == 2147483647, -1, ordered)
    tl.store(Indices + row * 512 + col, result)


def sort_selected_pools(indices):
    _sort_selected_pools[(indices.shape[0],)](indices, num_warps=4)
