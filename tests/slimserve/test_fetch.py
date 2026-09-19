from slimserve.fetch import _request_headers


def test_huggingface_auth_preserves_resume_range(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "local-token")

    assert _request_headers("https://huggingface.co/org/model/resolve/main/a", 42) == {
        "Authorization": "Bearer local-token",
        "Range": "bytes=42-",
    }


def test_huggingface_auth_is_not_sent_to_other_hosts(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "local-token")

    assert _request_headers("https://cdn-lfs.huggingface.co/a", 0) == {}
    assert _request_headers("https://huggingface.co.example/a", 0) == {}
    assert _request_headers("http://huggingface.co/a", 0) == {}


def test_huggingface_auth_is_optional(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)

    assert _request_headers("https://huggingface.co/org/model/resolve/main/a", 0) == {}


def test_explicit_download_headers_are_preserved_without_mutation(monkeypatch):
    def unexpected_token_read():
        raise AssertionError("explicit auth must take precedence")

    monkeypatch.setattr("huggingface_hub.get_token", unexpected_token_read)
    headers = {"Authorization": "Bearer explicit", "X-Request": "fixture"}
    actual = _request_headers("https://huggingface.co/org/model/a", 42, headers)
    assert actual == {**headers, "Range": "bytes=42-"}
    assert "Range" not in headers
