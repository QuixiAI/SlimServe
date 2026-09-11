# SPDX-License-Identifier: Apache-2.0
"""Warm the align-cache copy path without touching any model/cache storage."""

import torch


def warm_mamba_copy_kernel(device: torch.device) -> None:
    from vllm.v1.worker.mamba_utils import batch_memcpy

    # The copy length and batch count are runtime values, not JIT constants.
    # Use the production metadata dtypes so this compiles its actual signature.
    # A non-block-multiple length also exercises the final masked iteration.
    source = torch.zeros(1025, dtype=torch.uint8, device=device)
    destination = torch.empty_like(source)
    src_ptrs = torch.tensor([source.data_ptr()], dtype=torch.uint64, device=device)
    dst_ptrs = torch.tensor([destination.data_ptr()], dtype=torch.uint64, device=device)
    sizes = torch.tensor([source.numel()], dtype=torch.int32, device=device)
    batch_memcpy(src_ptrs, dst_ptrs, sizes)
