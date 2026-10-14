# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of actual-token accounting in the live tier harness."""

import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / "benchmarks/kv_tier_acceptance.py"
_spec = importlib.util.spec_from_file_location("tier_acceptance", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_churn = _module._churn


@pytest.mark.parametrize("estimate", [1, 10, 1000])
def test_actual_usage_not_estimate_controls_completion(estimate):
    records, total = _churn(lambda j: {"id": j, "prompt_tokens": 7}, 100, estimate, 4)
    assert total >= 100
    assert total == sum(r["prompt_tokens"] for r in records)
    assert [r["id"] for r in records] == list(range(len(records)))
    assert len(records) >= 15
    assert total < 100 + 4 * 7


@pytest.mark.parametrize("tokens", [None, 0, -1, 1.5, True])
def test_missing_or_invalid_usage_fails_closed(tokens):
    with pytest.raises(ValueError, match="positive prompt_tokens"):
        _churn(lambda j: {"prompt_tokens": tokens}, 10, 10, 2)


@pytest.mark.parametrize("args", [(0, 10, 2), (10, 0, 2), (10, 10, 0)])
def test_invalid_churn_shape(args):
    with pytest.raises(ValueError, match="must be positive"):
        _churn(lambda j: {}, *args)


def test_failed_request_is_not_counted_as_success():
    def fill(j):
        raise RuntimeError("request failed")

    with pytest.raises(RuntimeError, match="request failed"):
        _churn(fill, 10, 10, 2)
