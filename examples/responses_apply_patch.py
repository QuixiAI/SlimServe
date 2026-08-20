#!/usr/bin/env python3
"""Exercise a grammar-constrained Responses custom tool with genai-pyo3."""

from __future__ import annotations

import argparse
import asyncio
import sys

from genai_pyo3 import (
    ChatMessage,
    ChatOptions,
    ChatRequest,
    Client,
    Tool,
    ToolCall,
)

APPLY_PATCH_LARK = r"""start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""

PROMPT = (
    "The existing file greeting.txt currently contains exactly one line: "
    "`Hello, world!`. Change that line to `Hello, SlimServe!`. Use apply_patch, "
    "and pass exactly the following patch as its free-form input—do not add, "
    "delete, move, rename, or edit any other file or line:\n\n"
    "*** Begin Patch\n"
    "*** Update File: greeting.txt\n"
    "-Hello, world!\n"
    "+Hello, SlimServe!\n"
    "*** End Patch\n"
)


def validate_patch(patch: str) -> None:
    expected = (
        "*** Begin Patch\n"
        "*** Update File: greeting.txt\n"
        "-Hello, world!\n"
        "+Hello, SlimServe!\n"
        "*** End Patch\n"
    )
    if patch != expected:
        raise RuntimeError(
            "apply_patch input does not match the requested edit\n"
            f"expected: {expected!r}\n"
            f"received: {patch!r}"
        )


def custom_tool_input(tool_call: ToolCall) -> str:
    # The Responses provider serializes a custom tool's free-form input as a
    # JSON string for genai-pyo3. fn_arguments is that decoded string;
    # fn_arguments_json retains the quoted/escaped wire representation.
    value = tool_call.fn_arguments
    if isinstance(value, str):
        return value
    raise RuntimeError(
        "custom tool input was not a string: "
        f"fn_arguments_json={tool_call.fn_arguments_json!r}"
    )


def build_request() -> ChatRequest:
    tool = Tool(
        "apply_patch",
        description=(
            "Use the `apply_patch` tool to edit files. This is a FREEFORM "
            "tool, so do not wrap the patch in JSON."
        ),
        custom_format={
            "type": "grammar",
            "syntax": "lark",
            "definition": APPLY_PATCH_LARK,
        },
    )
    return ChatRequest(
        system="You are a coding assistant. Edit files with the apply_patch tool.",
        messages=[ChatMessage("user", PROMPT)],
        tools=[tool],
    )


async def stream_apply_patch(
    client: Client,
    model: str,
    request: ChatRequest,
    max_tokens: int,
) -> ToolCall:
    options = ChatOptions(
        temperature=0.0,
        max_tokens=max_tokens,
        capture_content=True,
        capture_reasoning_content=True,
        capture_tool_calls=True,
        capture_usage=True,
        reasoning_effort="none",
        extra_body={
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "chat_template_kwargs": {
                "thinking": False,
                "enable_thinking": False,
            },
        },
    )

    final_call: ToolCall | None = None
    final_usage: dict[str, object] | None = None
    final_reasoning = ""
    final_content = ""
    stream = await client.astream_chat(model, request, options)
    async for event in stream:
        if event.kind == "tool_call_chunk" and event.tool_call is not None:
            final_call = event.tool_call
        elif event.kind == "end" and event.end is not None:
            calls = event.end.captured_tool_calls
            if calls:
                final_call = calls[0]
            if event.end.captured_usage is not None:
                final_usage = event.end.captured_usage.to_dict()
            final_reasoning = event.end.captured_reasoning_content or ""
            final_content = event.end.captured_text or ""

    if final_call is None:
        raise RuntimeError(
            "model did not emit an apply_patch custom-tool call\n"
            f"content={final_content!r}\nreasoning={final_reasoning!r}\n"
            f"usage={final_usage!r}"
        )
    patch = custom_tool_input(final_call)
    sys.stdout.write(patch)
    if not patch.endswith("\n"):
        print()
    try:
        validate_patch(patch)
    except RuntimeError:
        print(
            f"content={final_content!r}\nreasoning={final_reasoning!r}\n"
            f"usage={final_usage!r}",
            file=sys.stderr,
        )
        raise
    return final_call


async def stream_continuation(
    client: Client,
    model: str,
    request: ChatRequest,
    tool_call: ToolCall,
    max_tokens: int,
) -> None:
    request.add_message(ChatMessage("assistant", "", tool_calls=[tool_call]))
    request.add_message(
        ChatMessage(
            "tool",
            "Patch applied successfully.",
            tool_response_call_id=tool_call.call_id,
        )
    )
    options = ChatOptions(
        temperature=0.0,
        max_tokens=max_tokens,
        capture_content=True,
        capture_tool_calls=True,
        capture_usage=True,
        reasoning_effort="none",
        extra_body={"tool_choice": "none"},
    )

    print("\nContinuation:")
    content = ""
    stream = await client.astream_chat(model, request, options)
    async for event in stream:
        if event.kind == "chunk" and event.content:
            sys.stdout.write(event.content)
            sys.stdout.flush()
            content += event.content
        elif event.kind == "end" and event.end is not None:
            content = event.end.captured_text or content
    if not content:
        raise RuntimeError("continuation did not produce assistant text")
    leaked_markers = (
        "<|start|>",
        "<|message|>",
        "<|eom|>",
        "<|eot|>",
        "<atem:",
    )
    if any(marker in content for marker in leaked_markers):
        raise RuntimeError(f"continuation leaked Muse protocol framing: {content!r}")
    if not content.endswith("\n"):
        print()


async def run(args: argparse.Namespace) -> None:
    # rust-genai resolves the Responses path relative to this URL. Keep a
    # trailing slash so `/v1` is retained instead of treated as a file name.
    base_url = f"{args.base_url.rstrip('/')}/"
    client = Client.with_api_key_and_base_url(
        "openai_resp",
        "not-needed",
        base_url,
        read_timeout_seconds=None,
        timeout_seconds=None,
    )
    request = build_request()
    tool_call = await stream_apply_patch(
        client,
        args.model,
        request,
        args.max_output_tokens,
    )
    if not args.skip_continuation:
        await stream_continuation(
            client,
            args.model,
            request,
            tool_call,
            args.continuation_tokens,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise a grammar-constrained Responses custom tool."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="SlimServe OpenAI-compatible API base URL",
    )
    parser.add_argument("--model", default="Muse-Glimmer-30B")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--continuation-tokens", type=int, default=256)
    parser.add_argument(
        "--skip-continuation",
        action="store_true",
        help="Stop after validating the custom-tool call",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
