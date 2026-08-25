# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal KV pool sizing (vllm/v1/worker/metal_worker.py).

An oversized pool on unified memory does not raise: Metal grants the
allocation and then evicts pages, so decode reads non-resident memory and
returns garbage with a successful HTTP status. These tests pin the sizing
that keeps the pool inside `recommendedMaxWorkingSetSize`, using the
figures measured while serving DeepSeek-V4-Flash IQ2_XXS on an M5 Max.
"""

import pytest

from vllm.v1.worker.metal_worker import (
    _KV_POOL_FLOOR_BYTES,
    _KV_RESERVE_MIN_BYTES,
    MetalWorker,
)

GIB = 1 << 30
# M5 Max: recommendedMaxWorkingSetSize, and what DSV4 IQ2_XXS + its DSpark
# drafter actually hold once loaded and pinned resident.
WORKING_SET = 107.52 * GIB
DSV4_RESIDENT = 96.70 * GIB


def _worker(total_bytes: float, resident_bytes: float) -> MetalWorker:
    """A MetalWorker stub whose device reports the given residency."""
    worker = MetalWorker.__new__(MetalWorker)
    worker.device = "mps"
    return worker


def _fit(monkeypatch, requested: float, total: float, resident: float) -> int:
    import vllm.v1.worker.metal_worker as mw

    monkeypatch.setattr(
        mw.torch.accelerator,
        "get_memory_info",
        lambda device: (int(total - resident), int(total)),
    )
    return _worker(total, resident)._fit_kv_pool_to_working_set(int(requested))


def test_pool_that_fits_is_granted_exactly(monkeypatch):
    """Qwen3.8 and Muse ask for 12 GiB against ~22-43 GiB of weights."""
    granted = _fit(monkeypatch, 12 * GIB, WORKING_SET, 22.43 * GIB)
    assert granted == 12 * GIB


def test_oversized_pool_is_reduced_to_stay_resident(monkeypatch):
    """The DSV4 profile's 16 GiB ceiling on a 128 GiB Mac.

    16 GiB on top of 96.70 GiB resident overruns the 107.52 GiB working
    set; the granted pool must leave the activation reserve intact.
    """
    granted = _fit(monkeypatch, 16 * GIB, WORKING_SET, DSV4_RESIDENT)
    reserve = max(_KV_RESERVE_MIN_BYTES, int(WORKING_SET * 0.04))
    assert granted == int(WORKING_SET - DSV4_RESIDENT) - reserve
    assert granted + DSV4_RESIDENT + reserve <= WORKING_SET
    # Measured-good band on this machine: 6-9 GiB decoded at 23-31 tok/s.
    assert 6 * GIB <= granted <= 9 * GIB


def test_refuses_when_nothing_useful_fits(monkeypatch):
    """A model that leaves no room must not boot into a doomed geometry."""
    with pytest.raises(RuntimeError, match="No unified memory left"):
        _fit(monkeypatch, 16 * GIB, WORKING_SET, WORKING_SET - 1 * GIB)


def test_floor_is_the_refusal_boundary(monkeypatch):
    """Just above the floor is granted; just below refuses."""
    reserve = max(_KV_RESERVE_MIN_BYTES, int(WORKING_SET * 0.04))
    resident_ok = WORKING_SET - reserve - _KV_POOL_FLOOR_BYTES
    assert _fit(monkeypatch, 16 * GIB, WORKING_SET, resident_ok) == _KV_POOL_FLOOR_BYTES

    with pytest.raises(RuntimeError, match="No unified memory left"):
        _fit(monkeypatch, 16 * GIB, WORKING_SET, resident_ok + 1 * GIB)


def test_reserve_scales_with_the_working_set(monkeypatch):
    """A 512 GiB Mac keeps 4% back, not the 4 GiB floor."""
    total = 460 * GIB
    granted = _fit(monkeypatch, 400 * GIB, total, 100 * GIB)
    assert granted == int(total) - 100 * GIB - int(total * 0.04)


@pytest.mark.parametrize("requested_gib", [1, 4, 6.5])
def test_smaller_requests_are_never_inflated(monkeypatch, requested_gib):
    """The fit only ever shrinks: a modest request is honoured as asked."""
    requested = int(requested_gib * GIB)
    assert _fit(monkeypatch, requested, WORKING_SET, DSV4_RESIDENT) == requested
