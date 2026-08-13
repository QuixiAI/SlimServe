#!/usr/bin/env python3
"""Call SlimServe's Responses API with Codex's apply_patch custom tool."""

from __future__ import annotations

import argparse
import sys

from openai import OpenAI

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise a grammar-constrained apply_patch custom tool."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="SlimServe OpenAI-compatible API base URL",
    )
    parser.add_argument("--model", default="Muse-Glimmer-30B")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")
    response = client.responses.create(
        model=args.model,
        input=(
            "greeting.txt currently contains exactly `Hello, world!\\n`. "
            "Use apply_patch to make it contain exactly "
            "`Hello, SlimServe!\\n`. Edit the relative path greeting.txt. "
            "The deletion line must be `-Hello, world!` and the addition "
            "line must be `+Hello, SlimServe!`. Call apply_patch now."
        ),
        tools=[
            {
                "type": "custom",
                "name": "apply_patch",
                "description": (
                    "Use the `apply_patch` tool to edit files. This is a "
                    "FREEFORM tool, so do not wrap the patch in JSON."
                ),
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": APPLY_PATCH_LARK,
                },
            }
        ],
        tool_choice={"type": "custom", "name": "apply_patch"},
        reasoning={"effort": "none"},
        max_output_tokens=args.max_output_tokens,
    )

    for item in response.output:
        if item.type == "custom_tool_call":
            patch = item.input
            print(patch, end="" if patch.endswith("\n") else "\n")
            required_fragments = (
                "*** Begin Patch\n",
                "*** Update File: greeting.txt",
                "\n-Hello, world!\n",
                "\n+Hello, SlimServe!\n",
                "*** End Patch",
            )
            if not all(fragment in patch for fragment in required_fragments):
                print("patch does not perform the requested edit", file=sys.stderr)
                raise SystemExit(1)
            return
    raise RuntimeError(f"Model did not emit apply_patch: {response.output!r}")


if __name__ == "__main__":
    main()
