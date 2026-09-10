# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of actual DMA completion bookkeeping after disk failures."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.worker.gpu.kv_tier_dma import KVTierDMA


class CompletedDisk:
    slot_bytes = 4096

    def __init__(self):
        self.done = []

    def poll_done(self):
        done, self.done = self.done, []
        return done


def make_dma():
    # Exercise the real constructor without allocating CUDA or pinned memory.
    backing = SimpleNamespace(
        dtype=torch.int8, is_cuda=True, numel=lambda: 4096, view=lambda *shape: None
    )
    disk = CompletedDisk()
    with (
        patch(
            "vllm.v1.worker.gpu.kv_tier_dma._register_host_arena",
            return_value=(None, None),
        ),
        patch("torch.cuda.Stream"),
    ):
        dma = KVTierDMA(backing, 4096, 1, torch.device("cpu"), disk=disk)
    return dma, disk


@pytest.mark.parametrize("failure_position", [0, 1, 2])
@pytest.mark.parametrize("separate_polls", [False, True])
def test_any_failed_write_prevents_batch_ack(failure_position, separate_polls):
    dma, disk = make_dma()
    dma._disk_ops = {i: ("w", 101) for i in range(3)}
    dma._write_remaining = {101: 3}
    completions = [
        (i, "injected write failure" if i == failure_position else None)
        for i in range(3)
    ]
    if separate_polls:
        for completion in completions:
            disk.done = [completion]
            dma.pump()
            assert dma.take_disk_done() == []
    else:
        disk.done = completions
        dma.pump()
        assert dma.take_disk_done() == []
    assert dma._write_remaining == {}
    assert dma._disk_ops == {}


def test_interleaved_failure_does_not_poison_other_batch():
    dma, disk = make_dma()
    dma._disk_ops = {0: ("w", 101), 1: ("w", 102), 2: ("w", 101), 3: ("w", 102)}
    dma._write_remaining = {101: 2, 102: 2}
    disk.done = [(0, "injected failure"), (1, None)]
    dma.pump()
    assert dma.take_disk_done() == []
    disk.done = [(3, None), (2, None)]
    dma.pump()
    assert dma.take_disk_done() == [102]
    dma.pump()
    assert dma.take_disk_done() == []


def test_success_requires_every_write_and_reports_once():
    dma, disk = make_dma()
    dma._disk_ops = {i: ("w", 101) for i in range(3)}
    dma._write_remaining = {101: 3}
    disk.done = [(2, None), (0, None)]
    dma.pump()
    assert dma.take_disk_done() == []
    disk.done = [(1, None)]
    dma.pump()
    assert dma.take_disk_done() == [101]
    assert dma.take_disk_done() == []
