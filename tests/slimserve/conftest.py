# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep the operator's private profiles.local.json out of the registry tests.

The tests describe the checked-in registry; records a machine adds through the
git-ignored local file must not change what they see.
"""

import pytest

from slimserve import registry


@pytest.fixture(autouse=True)
def _checked_in_registry_only(monkeypatch, tmp_path):
    monkeypatch.setenv("SLIMSERVE_LOCAL_PROFILES", str(tmp_path / "absent.json"))
    registry._registry.cache_clear()
    yield
    registry._registry.cache_clear()
