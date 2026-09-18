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
    # Four whole-message boundaries plus one boundary between the user parts.
    assert len(prefixes) == 5
    assert prefixes[1][-1]["content"] == [
        {"type": "input_text", "text": "first"}
    ]
    assert prefixes[-1] == messages

    # Candidate construction is isolated from the request objects.
    prefixes[1][-1]["content"][0]["text"] = "mutated"
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
    assert _semantic_message_prefixes(messages) == [messages]


def test_common_token_prefix_stops_at_first_template_divergence():
    assert _common_token_prefix_len([1, 2, 3, 9], [1, 2, 3, 4, 5]) == 3
    assert _common_token_prefix_len([1, 2], [1, 2, 3]) == 2
