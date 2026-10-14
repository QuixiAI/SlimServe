# SPDX-License-Identifier: Apache-2.0
"""The compact page format must never silently accept unsupported state."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.layers.glm5_next_indexer import _cache_row_dim


@pytest.mark.parametrize(
    "enabled,sm80,kp,spec,expected",
    [
        (False, False, 4, 20, 256),
        (True, True, 4, 0, 64),
        (True, True, 4, 5, 64),
        (True, True, 4, 6, None),
        (True, True, 2, 0, None),
        (True, False, 4, 0, None),
        ("false", True, 4, 0, None),
    ],
)
def test_compact_cache_config(enabled, sm80, kp, spec, expected):
    config = SimpleNamespace(
        additional_config={"glm5_next_compact_indexer_cache": enabled},
        speculative_config=SimpleNamespace(num_speculative_tokens=spec),
    )
    with patch(
        "vllm.model_executor.layers.glm5_next_indexer.current_platform"
    ) as platform:
        platform.is_cuda.return_value = sm80
        platform.is_device_capability.return_value = sm80
        if expected is None:
            with pytest.raises(ValueError):
                _cache_row_dim(config, kp)
        else:
            assert _cache_row_dim(config, kp) == expected
