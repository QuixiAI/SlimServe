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


class _StreamResponse(_Response):
    def __init__(self, status_code, body=b""):
        super().__init__(status_code)
        self.body = body
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")

    def iter_content(self, _size):
        yield self.body


def _hf_entry(body: bytes) -> dict:
    return {
        "path": "a.gguf",
        "bytes": len(body),
        "url": "https://huggingface.co/org/model/resolve/main/a.gguf",
    }


def test_stale_token_does_not_break_a_public_download(monkeypatch, tmp_path):
    """A revoked token gets 401 even on a public repo: retry anonymously."""
    import requests

    body = b"public-bytes"
    calls = []

    def get(url, headers, **_):
        calls.append(dict(headers))
        if "Authorization" in headers:
            return _StreamResponse(401)
        return _StreamResponse(200, body)

    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "stale-token")
    monkeypatch.setattr(requests, "get", get)
    fetch._download(_hf_entry(body), tmp_path / "a.gguf")

    assert (tmp_path / "a.gguf").read_bytes() == body
    assert calls == [{"Authorization": "Bearer stale-token"}, {}]


def test_gated_repo_hint_after_the_anonymous_retry_also_fails(
    monkeypatch, tmp_path
):
    import requests

    calls = []

    def get(url, headers, **_):
        calls.append(dict(headers))
        return _StreamResponse(401)

    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "no-access-token")
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(ProfileError, match=r"org/model .*gated.*HF_TOKEN"):
        fetch._download(_hf_entry(b"x"), tmp_path / "a.gguf")
    assert len(calls) == 2 and calls[1] == {}


def _clear_token_env(monkeypatch, home):
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN_PATH",
        "HF_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(home))


def test_hf_token_reads_the_hf_home_token(monkeypatch, tmp_path):
    _clear_token_env(monkeypatch, tmp_path / "home")
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "token").write_text("home-token\n")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert fetch.hf_token() == "home-token"


def test_hf_token_path_wins_over_hf_home_and_env_wins_over_both(
    monkeypatch, tmp_path
):
    _clear_token_env(monkeypatch, tmp_path / "home")
    (tmp_path / "hf").mkdir()
    (tmp_path / "hf" / "token").write_text("home-token")
    (tmp_path / "explicit").write_text("path-token")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "explicit"))
    assert fetch.hf_token() == "path-token"
    monkeypatch.setenv("HF_TOKEN", "env-token")
    assert fetch.hf_token() == "env-token"


def _hub_error(status):
    from types import SimpleNamespace

    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError.__new__(HfHubHTTPError)
    Exception.__init__(exc, f"HTTP {status}")
    exc.response = SimpleNamespace(status_code=status)
    return exc


def _hub_entry():
    return {
        "path": "a.gguf",
        "bytes": 1,
        "hf_repo": "org/model",
        "hf_revision": "main",
        "local_dir": "m",
    }


def test_hub_download_gated_repo_gets_the_hint(monkeypatch, tmp_path):
    import huggingface_hub

    tokens = []

    def hub(**kwargs):
        tokens.append(kwargs["token"])
        raise _hub_error(401)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "no-access-token")
    monkeypatch.setattr("slimserve.fetch.cache_root", lambda: tmp_path)
    with pytest.raises(ProfileError, match=r"org/model .*gated.*HF_TOKEN"):
        fetch._download(_hub_entry(), tmp_path / "m" / "a.gguf")
    assert tokens == ["no-access-token", False]


def test_hub_download_stale_token_retries_anonymously(monkeypatch, tmp_path):
    import huggingface_hub

    dest = tmp_path / "m" / "a.gguf"
    tokens = []

    def hub(**kwargs):
        tokens.append(kwargs["token"])
        if kwargs["token"]:
            raise _hub_error(401)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hub)
    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "stale-token")
    monkeypatch.setattr("slimserve.fetch.cache_root", lambda: tmp_path)
    fetch._download(_hub_entry(), dest)
    assert tokens == ["stale-token", False]


def test_override_listing_with_a_stale_token_falls_back_to_anonymous(
    monkeypatch, tmp_path
):
    import requests

    from slimserve.registry import ModelOverride

    class _Api:
        def __init__(self, status):
            self.status_code = status

        def raise_for_status(self):
            pass

        def json(self):
            return {"siblings": [{"rfilename": "w.safetensors", "size": 3}]}

    calls = []

    def get(url, headers, **_):
        calls.append(dict(headers))
        return _Api(401 if "Authorization" in headers else 200)

    monkeypatch.setattr("slimserve.fetch.hf_token", lambda: "stale-token")
    monkeypatch.setattr(requests, "get", get)
    entries = fetch.override_entries(
        ModelOverride("org/ft", "org/ft", tmp_path / "ft")
    )
    assert calls == [
        {"Authorization": "Bearer stale-token"},
        {},
        {"Range": "bytes=0-0"},
    ]
    assert [e["headers"] for e in entries] == [{}]
