# SPDX-License-Identifier: Apache-2.0
import json

import requests

from slimserve.stream import chat_completion


def test_chat_observer_preserves_raw_fields_without_changing_request_or_text(
    monkeypatch,
):
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning_content": "Red", "reasoning": "Red"}}]},
        {"choices": [{"delta": {"content": "red"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 3}},
    ]
    bodies = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_lines(self, decode_unicode):
            assert decode_unicode
            yield ""
            yield "data: not-json"
            for chunk in chunks:
                yield "data: " + json.dumps(chunk)
            yield "data: [DONE]"

    def post(url, json, stream, timeout):
        assert url == "http://localhost/v1/chat/completions"
        assert stream and timeout == 600
        bodies.append(json)
        return Response()

    monkeypatch.setattr(requests, "post", post)
    events = []
    kwargs = dict(max_tokens=256, seed=42)
    messages = [{"role": "user", "content": "test"}]
    plain = list(chat_completion("http://localhost", "model", messages, **kwargs))
    observed = list(
        chat_completion(
            "http://localhost", "model", messages, on_event=events.append, **kwargs
        )
    )
    assert plain == observed == ["Red", "Red", "red"]
    assert events == chunks
    assert (
        bodies[0]
        == bodies[1]
        == {
            "model": "model",
            "messages": messages,
            "max_tokens": 256,
            "stream": True,
            "seed": 42,
        }
    )
