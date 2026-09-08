# SPDX-License-Identifier: Apache-2.0
import ctypes

import pytest
import torch

from vllm.model_executor.warmup.mamba_copy_warmup import warm_mamba_copy_kernel
from vllm.v1.worker import mamba_utils


def test_warmup_uses_production_copy_api_and_metadata_dtypes(monkeypatch):
    calls = []

    def copy(src, dst, sizes):
        assert src.dtype == dst.dtype == torch.uint64
        assert sizes.dtype == torch.int32
        assert src.shape == dst.shape == sizes.shape == (1,)
        assert src.item() != dst.item()
        assert sizes.item() == 1025
        ctypes.memmove(dst.item(), src.item(), sizes.item())
        calls.append(sizes.item())

    monkeypatch.setattr(mamba_utils, "batch_memcpy", copy)
    warm_mamba_copy_kernel(torch.device("cpu"))
    assert calls == [1025]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_warmed_copy_handles_real_lengths_and_left_overlap(monkeypatch):
    from triton import knobs

    device = torch.device("cuda")
    warm_mamba_copy_kernel(device)

    def unexpected_compile(**kwargs):
        raise AssertionError("state copy recompiled after startup warmup")

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", unexpected_compile)
    for size in (0, 1, 1023, 1024, 1025, 8192):
        source = torch.arange(size + 8, device=device, dtype=torch.int32).to(
            torch.uint8
        )
        reference = source.clone()
        # Left-shift an overlapping state as used by convolution caches.
        source_ptr = source.data_ptr() + 3
        destination_ptr = source.data_ptr()
        mamba_utils.batch_memcpy(
            torch.tensor([source_ptr], dtype=torch.uint64, device=device),
            torch.tensor([destination_ptr], dtype=torch.uint64, device=device),
            torch.tensor([size], dtype=torch.int32, device=device),
        )
        assert torch.equal(source[:size], reference[3 : 3 + size])
        assert torch.equal(source[size:], reference[size:])
