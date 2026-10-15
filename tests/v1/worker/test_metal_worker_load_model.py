# SPDX-License-Identifier: Apache-2.0
"""The Metal worker's load_model override must keep the base post-load step."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.worker import gpu_worker, metal_worker


def _worker(weight_transfer_config):
    worker = object.__new__(metal_worker.MetalWorker)
    worker.rank = 0
    worker.device = "mps"
    worker.vllm_config = SimpleNamespace(weight_transfer_config=weight_transfer_config)
    worker.weight_transfer_engine = None
    model = object()
    worker.model_runner = MagicMock()
    worker.model_runner.get_model.return_value = model
    return worker, model


@pytest.fixture
def no_distributed():
    tp = SimpleNamespace(device_communicator=None)
    with (
        patch.object(metal_worker, "get_tp_group", return_value=tp),
        patch.object(metal_worker, "set_current_vllm_config", MagicMock()),
        patch.object(metal_worker.gc, "freeze"),
    ):
        yield


@pytest.mark.usefixtures("no_distributed")
def test_metal_load_model_creates_the_weight_transfer_engine():
    config = SimpleNamespace(backend="nccl")
    worker, model = _worker(config)
    engine = object()
    with (
        patch.object(
            gpu_worker.WeightTransferEngineFactory,
            "create_engine",
            return_value=engine,
        ) as create,
        patch.object(gpu_worker, "bootstamp") as stamp,
    ):
        worker.load_model(load_dummy_weights=True)
    worker.model_runner.load_model.assert_called_once_with(load_dummy_weights=True)
    create.assert_called_once_with(config, worker.vllm_config, "mps", model)
    assert worker.weight_transfer_engine is engine
    assert any(
        "load_model done" in call.args[0] for call in stamp.call_args_list
    )


@pytest.mark.usefixtures("no_distributed")
def test_metal_load_model_without_weight_transfer_config():
    worker, _ = _worker(None)
    with (
        patch.object(
            gpu_worker.WeightTransferEngineFactory, "create_engine"
        ) as create,
        patch.object(gpu_worker, "bootstamp"),
    ):
        worker.load_model(load_dummy_weights=True)
    create.assert_not_called()
    assert worker.weight_transfer_engine is None
