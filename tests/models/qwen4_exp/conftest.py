# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.fixture()
def default_vllm_config():
    """Set a default VllmConfig for tests that exercise pathways using
    get_current_vllm_config() outside of a full engine context."""
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


@pytest.fixture
def workspace_init():
    """Initialize the workspace manager for tests that need it."""
    import torch

    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    if torch.accelerator.is_available():
        init_workspace_manager(torch.device(0))
    yield
    reset_workspace_manager()
