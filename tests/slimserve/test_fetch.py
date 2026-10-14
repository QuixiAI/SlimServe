import json
from pathlib import Path

import pytest

from slimserve import fetch
from slimserve.fetch import _request_headers
from slimserve.registry import ProfileError


def test_huggingface_auth_preserves_resume_range(monkeypatch):
    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "local-token")

    assert _request_headers("https://huggingface.co/org/model/resolve/main/a", 42) == {
        "Authorization": "Bearer local-token",
        "Range": "bytes=42-",
    }


def test_huggingface_auth_is_not_sent_to_other_hosts(monkeypatch):
    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "local-token")

    assert _request_headers("https://cdn-lfs.huggingface.co/a", 0) == {}
    assert _request_headers("https://huggingface.co.example/a", 0) == {}
    assert _request_headers("http://huggingface.co/a", 0) == {}


def test_huggingface_auth_is_optional(monkeypatch):
    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: None)

    assert _request_headers("https://huggingface.co/org/model/resolve/main/a", 0) == {}


def test_explicit_download_headers_are_preserved_without_mutation(monkeypatch):
    def unexpected_token_read():
        raise AssertionError("explicit auth must take precedence")

    monkeypatch.setattr("slimserve.fetch.hf_token", unexpected_token_read)
    headers = {"Authorization": "Bearer explicit", "X-Request": "fixture"}
    actual = _request_headers("https://huggingface.co/org/model/a", 42, headers)
    assert actual == {**headers, "Range": "bytes=42-"}
    assert "Range" not in headers


def _registered_huggingface_url() -> str:
    profiles = json.loads(
        (Path(fetch.__file__).parent / "profiles.json").read_text()
    )

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)
        elif isinstance(node, str) and node.startswith("https://huggingface.co/"):
            yield node

    return next(walk(profiles))


def test_huggingface_token_env_alone_authenticates_registered_url(
    monkeypatch, tmp_path
):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no stored login token
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "env-only-token")

    url = _registered_huggingface_url() + "/model.gguf"
    assert _request_headers(url, 0) == {"Authorization": "Bearer env-only-token"}


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        raise AssertionError("the 401 must be reported before raise_for_status")


@pytest.mark.parametrize("status", [401, 403])
def test_gated_registry_download_explains_how_to_get_access(
    monkeypatch, tmp_path, status
):
    import requests

    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(status))
    entry = {
        "path": "a.gguf",
        "bytes": 10,
        "url": "https://huggingface.co/org/model/resolve/main/a.gguf",
    }
    with pytest.raises(ProfileError, match=r"org/model .*Accept.*HF_TOKEN"):
        fetch._download(entry, tmp_path / "a.gguf")
