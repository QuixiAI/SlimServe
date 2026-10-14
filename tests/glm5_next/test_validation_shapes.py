# SPDX-License-Identifier: Apache-2.0
"""Long-context options preserve the default serving comparison protocol."""

import io
import json
import sys

import pytest

from benchmarks import validate_glm5_next as validate


@pytest.mark.parametrize("long_context", [False, True])
def test_exact_shape_options_and_default_warmup(monkeypatch, tmp_path, long_context):
    args = [
        "validate",
        "--tokenizer",
        "/unused/model",
        "--out",
        str(tmp_path),
        "--concurrency",
        "1",
        "--repeats",
        "1",
    ]
    if long_context:
        args += [
            "--input-tokens",
            "16384",
            "--output-tokens",
            "1000",
            "--repeat-source",
        ]
    monkeypatch.setattr(sys, "argv", args)
    messages = iter(
        [
            {"content": "Paris", "reasoning": "A capital."},
            {"content": "red square"},
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"}),
                        }
                    }
                ]
            },
        ]
    )

    def urlopen(request, **kwargs):
        if isinstance(request, str):
            return io.BytesIO(b"")  # health
        return io.BytesIO(
            json.dumps({"choices": [{"message": next(messages)}]}).encode()
        )

    monkeypatch.setattr(validate.urllib.request, "urlopen", urlopen)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        kwargs["stdout"].write(json.dumps({"aggregate_output_tps": 1.0}))

    monkeypatch.setattr(validate.subprocess, "run", run)
    validate.main()
    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--input-tokens") + 1] == (
            "16384" if long_context else "1000"
        )
        assert ("--repeat-source" in command) == long_context
    assert commands[0][commands[0].index("--output-tokens") + 1] == "32"
    assert commands[1][commands[1].index("--output-tokens") + 1] == (
        "1000" if long_context else "300"
    )
