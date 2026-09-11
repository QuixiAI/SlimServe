# SPDX-License-Identifier: Apache-2.0
"""Zero-copy tier checksums preserve the exact existing byte-level contract."""

import hashlib

import pytest
import torch

from vllm.v1.worker.gpu.kv_tier_dma import _digest
from vllm.v1.worker.gpu.kv_tier_nvme import _row_digest


@pytest.mark.parametrize("size", [0, 1, 4096, 6488064])
def test_contiguous_dma_and_disk_digest_match_old_bytes(size):
    tensor = torch.arange(size, dtype=torch.int64).to(torch.uint8)
    expected = hashlib.sha1(tensor.numpy().tobytes()).hexdigest()[:12]
    assert _digest(tensor) == expected
    assert _row_digest(memoryview(tensor.numpy())) == expected


@pytest.mark.parametrize("view", [lambda t: t[::2], lambda t: t.reshape(16, 16).T])
def test_strided_views_preserve_logical_c_order(view):
    tensor = view(torch.arange(256, dtype=torch.int32))
    expected = hashlib.sha1(tensor.numpy().tobytes()).hexdigest()[:12]
    assert _digest(tensor) == expected
    assert _row_digest(memoryview(tensor.numpy())) == expected


def test_contiguous_path_passes_original_storage_to_hashlib(monkeypatch):
    import numpy as np

    tensor = torch.arange(4096, dtype=torch.int32)
    original = hashlib.sha1
    seen = []

    def checked(data):
        assert isinstance(data, memoryview)
        assert np.shares_memory(np.asarray(data), tensor.numpy())
        seen.append(True)
        return original(data)

    monkeypatch.setattr(hashlib, "sha1", checked)
    before = _digest(tensor)
    tensor[3] += 1
    after = _digest(tensor)
    assert before != after
    assert len(seen) == 2
