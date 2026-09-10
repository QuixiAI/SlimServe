# SPDX-License-Identifier: Apache-2.0
"""A completed offload must never acknowledge an unfinished restore."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.v1.core.test_host_tier_connector import STRIDE, _base_init, make_groups
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector import (
    HostTierConnector,
    HostTierMeta,
)


class DelayedDMA:
    def __init__(self):
        self.issued = []
        self.done = []

    def pump(self):
        pass

    def fence_restores(self):
        pass

    def issue(self, batch):
        self.issued.append(batch)

    def poll_done(self):
        done, self.done = self.done, []
        return done


def make_worker():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=8),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"host_tier_gb_per_rank": 1.0}
        ),
    )
    kv_config = SimpleNamespace(kv_cache_groups=make_groups(), kv_cache_tensors=[])
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.host_tier_connector."
            "_get_packed_kv_cache_layout",
            return_value=(STRIDE, {}),
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.base."
            "KVConnectorBase_V1.__init__",
            _base_init,
        ),
    ):
        worker = HostTierConnector(config, KVConnectorRole.WORKER, kv_config)
    worker._dma = DelayedDMA()
    return worker


@pytest.mark.parametrize("offload_seq", [1, (1 << 20) + 1, 1 << 40])
def test_delayed_restore_not_completed_by_offload(offload_seq):
    worker = make_worker()
    worker._connector_metadata = HostTierMeta(
        offloads={offload_seq: [(10, 0, 0)]},
        restores={"still-copying": [(3, 20, 0)]},
    )
    worker.start_load_kv(None)
    offload, restore = worker._dma.issued
    worker._dma.done = [offload.seq]
    assert worker.get_finished(set()) == (None, None)
    assert worker._pending_restore_reqs == {restore.seq: "still-copying"}
    assert restore.seq < 0 < offload.seq
    worker._dma.done = [restore.seq]
    assert worker.get_finished(set()) == (None, {"still-copying"})
    assert not worker._pending_restore_reqs
    worker._dma.done = [restore.seq]
    assert worker.get_finished(set()) == (None, None)


def test_multiple_steps_keep_unique_negative_restore_ids():
    worker = make_worker()
    restore_ids = []
    for step in range(12):
        worker._connector_metadata = HostTierMeta(
            offloads={step + 1: [(10, 0, 0)]},
            restores={f"req-{step}": [(3, 20, 0)]},
        )
        worker.start_load_kv(None)
        restore_ids.append(worker._dma.issued[-1].seq)
    assert len(set(restore_ids)) == 12
    assert all(seq < 0 for seq in restore_ids)
    worker._dma.done = list(range(1, 13))
    assert worker.get_finished(set()) == (None, None)
    assert len(worker._pending_restore_reqs) == 12
    for step in reversed(range(12)):
        worker._dma.done = [restore_ids[step]]
        assert worker.get_finished(set()) == (None, {f"req-{step}"})
    assert not worker._pending_restore_reqs
