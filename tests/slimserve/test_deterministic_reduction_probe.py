# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from benchmarks.kernels.check_glm53_deterministic_reductions import (
    forbid_benchmark,
    policy_metadata,
    recorded_choice,
)


def test_only_deterministic_metadata_changes_without_aliasing():
    old = dict(
        deterministic=False,
        dynamic_scale_rblock=True,
        batch_invariant=False,
        nested={"source": [1, 2]},
    )
    new = policy_metadata(old)
    assert new == {**old, "deterministic": True}
    new["nested"]["source"].append(3)
    assert old["nested"]["source"] == [1, 2]
    assert old["deterministic"] is False


@pytest.mark.parametrize(
    "metadata", [{}, {"deterministic": True}, {"deterministic": 0}]
)
def test_unexpected_source_policy_rejected(metadata):
    with pytest.raises(ValueError, match="recorded nondeterministic"):
        policy_metadata(metadata)


def test_benchmarking_is_a_failure():
    with pytest.raises(ValueError, match="attempted GPU benchmarking"):
        forbid_benchmark(None, stream=1)


def test_selection_requires_exact_recorded_config():
    config = SimpleNamespace(
        kwargs={"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=8, num_stages=1
    )
    saved = dict(
        XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1, triton_cache_hash="binary"
    )
    assert recorded_choice(config, [saved]) is saved
    for altered in ([{**saved, "num_warps": 16}], [saved, saved]):
        with pytest.raises(ValueError, match="unqualified"):
            recorded_choice(config, altered)
