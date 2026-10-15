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


def test_local_records_are_classified_by_kind(monkeypatch, tmp_path):
    """Overlay sources go through the same kind allowlist and LLM/non-LLM split
    as checked-in ones."""
    base = registry._registry()
    llm_id = registry.language_model_profile_ids()[0]
    video_id = next(
        pid for pid in registry.profile_ids() if registry.profile_kind(pid) == "video"
    )
    records = {}
    for new, old in (("private-llm", llm_id), ("private-video", video_id)):
        records[new] = {**base["profiles"][old], "source": f"{new}-src"}
    sources = {
        f"{new}-src": base["sources"][p["source"]]
        for new, p in (
            ("private-llm", base["profiles"][llm_id]),
            ("private-video", base["profiles"][video_id]),
        )
    }
    _use(
        monkeypatch,
        _write(tmp_path / "local.json", {"sources": sources, "profiles": records}),
    )
    assert "private-llm" in registry.language_model_profile_ids()
    assert "private-video" not in registry.language_model_profile_ids()
    assert registry.profile_kind("private-video") == "video"
    bad = {
        "sources": {
            "odd": {
                **base["sources"][base["profiles"][llm_id]["source"]],
                "kind": "speech",
            }
        }
    }
    _use(monkeypatch, _write(tmp_path / "bad.json", bad))
    with pytest.raises(ValueError, match="kind 'speech'"):
        registry.profile_ids()
