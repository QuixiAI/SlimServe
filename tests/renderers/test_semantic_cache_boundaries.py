from copy import deepcopy
from unittest.mock import patch

import pytest

from vllm.renderers.online_renderer import (
    _common_token_prefix_len,
    _semantic_message_prefixes,
)


def test_message_and_text_part_prefixes_are_preserved():
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "input_text", "text": "second"},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]

    prefixes = _semantic_message_prefixes(messages)
    # The system-only prefix is not renderable by every chat template. Keep
    # the three later whole-message boundaries and the textual part boundary.
    assert len(prefixes) == 4
    assert prefixes[0][-1]["content"] == [{"type": "input_text", "text": "first"}]
    assert prefixes[-1] == messages

    # Candidate construction is isolated from the request objects.
    prefixes[0][-1]["content"][0]["text"] = "mutated"
    assert messages[1]["content"][0]["text"] == "first"


def test_multimodal_parts_do_not_create_partial_message_replays():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe"},
                {"type": "input_image", "image_url": "data:image/png;base64,..."},
            ],
        }
    ]
    assert _semantic_message_prefixes(messages) == []


def test_common_token_prefix_stops_at_first_template_divergence():
    assert _common_token_prefix_len([1, 2, 3, 9], [1, 2, 3, 4, 5]) == 3
    assert _common_token_prefix_len([1, 2], [1, 2, 3]) == 2


def test_system_only_prefix_is_not_a_semantic_candidate():
    messages = [{"role": "system", "content": "system"}]
    assert _semantic_message_prefixes(messages) == []


@pytest.mark.parametrize(
    "part_type", ["image_url", "input_image", "input_audio", "video"]
)
def test_media_anywhere_in_history_skips_all_semantic_replays(part_type):
    messages = [
        {"role": "user", "content": "earlier text"},
        {"role": "user", "content": [{"type": part_type}]},
        {"role": "assistant", "content": "description"},
        {"role": "user", "content": "later text"},
    ]
    with patch("vllm.renderers.online_renderer.deepcopy") as copy:
        assert _semantic_message_prefixes(messages) == []
    copy.assert_not_called()


@pytest.mark.parametrize("many_parts", [False, True])
def test_only_selected_recent_candidates_are_copied(many_parts):
    if many_parts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(index)} for index in range(200)
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": str(index)} for index in range(200)]
    with patch("vllm.renderers.online_renderer.deepcopy", wraps=deepcopy) as copy:
        prefixes = _semantic_message_prefixes(messages)
    assert copy.call_count == 16
    assert len(prefixes) == 16
    assert prefixes[-1] == messages
    if many_parts:
        assert [len(prefix[-1]["content"]) for prefix in prefixes] == list(
            range(185, 201)
        )
    else:
        assert [len(prefix) for prefix in prefixes] == list(range(185, 201))
