# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.worker.workspace import WorkspaceManager


def test_named_workspace_is_isolated_and_reused() -> None:
    manager = WorkspaceManager(torch.device("cpu"))
    (transient,) = manager.get_simultaneous(((1024,), torch.uint8))
    first, second = manager.get_named_simultaneous(
        "indexer",
        ((257,), torch.float32),
        ((513,), torch.uint8),
    )

    assert transient.untyped_storage().data_ptr() != first.untyped_storage().data_ptr()
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    assert second.data_ptr() - first.data_ptr() == 1280  # 257 fp32, 256B aligned

    again, _ = manager.get_named_simultaneous(
        "indexer",
        ((257,), torch.float32),
        ((513,), torch.uint8),
    )
    assert again.data_ptr() == first.data_ptr()


def test_named_workspace_respects_lock() -> None:
    manager = WorkspaceManager(torch.device("cpu"))
    manager.get_named_simultaneous("indexer", ((1024,), torch.uint8))
    manager.lock()

    manager.get_named_simultaneous("indexer", ((512,), torch.uint8))
    with pytest.raises(AssertionError, match="Named workspace 'indexer' is locked"):
        manager.get_named_simultaneous("indexer", ((2048,), torch.uint8))
