# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest

from slimserve import registry


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _use(monkeypatch, path):
    monkeypatch.setenv("SLIMSERVE_LOCAL_PROFILES", str(path))
    registry._registry.cache_clear()


def test_local_file_adds_private_records(monkeypatch, tmp_path):
    base = registry._registry()
    profile_id = next(iter(base["profiles"]))
    source_id = base["profiles"][profile_id]["source"]
    local = _write(
        tmp_path / "local.json",
        {
            "sources": {"private-src": base["sources"][source_id]},
            "profiles": {
                "private-1": {**base["profiles"][profile_id], "source": "private-src"}
            },
        },
    )
    _use(monkeypatch, local)
    assert "private-1" in registry.profile_ids()
    assert registry.describe("private-1")["source"] == "private-src"


def test_local_file_cannot_redefine_a_checked_in_record(monkeypatch, tmp_path):
    profile_id = registry.profile_ids()[0]
    local = _write(tmp_path / "local.json", {"profiles": {profile_id: {}}})
    _use(monkeypatch, local)
    with pytest.raises(ValueError, match="already defined"):
        registry.profile_ids()


def test_local_file_rejects_unknown_sections(monkeypatch, tmp_path):
    local = _write(tmp_path / "local.json", {"schema": 2})
    _use(monkeypatch, local)
    with pytest.raises(ValueError, match="unsupported sections"):
        registry.profile_ids()


def test_missing_local_file_leaves_registry_unchanged(monkeypatch, tmp_path):
    before = registry.profile_ids()
    _use(monkeypatch, tmp_path / "absent.json")
    assert registry.profile_ids() == before
