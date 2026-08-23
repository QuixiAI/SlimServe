# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from xgrammar import Grammar, GrammarCompiler, GrammarMatcher, TokenizerInfo

from vllm.v1.structured_output.utils import convert_lark_to_ebnf

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


def test_openai_custom_tool_lark_example_converts_to_ebnf():
    grammar = """
start: expr
expr: term (SP ADD SP term)* -> add
| term
term: factor (SP MUL SP factor)* -> mul
| factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
%import common.INT
"""

    ebnf = convert_lark_to_ebnf(grammar)

    assert "root ::= start" in ebnf
    assert "expr ::= term (SP ADD SP term)* | term" in ebnf
    assert "term ::= factor (SP MUL SP factor)* | factor" in ebnf
    assert "INT ::= [0-9]+" in ebnf
    assert "->" not in ebnf
    assert "%import" not in ebnf


def test_lark_builtin_import_alias_converts_to_ebnf():
    grammar = "start: value\nvalue: integer\n%import common.INT -> integer"

    ebnf = convert_lark_to_ebnf(grammar)

    assert "integer ::= [0-9]+" in ebnf


def test_lark_builtin_lf_converts_to_ebnf():
    grammar = 'start: "line" LF\n%import common.LF'

    ebnf = convert_lark_to_ebnf(grammar)

    assert 'start ::= "line" LF' in ebnf
    assert 'LF ::= "\\n"' in ebnf


def test_lark_inline_regex_terminals_convert_to_ebnf():
    grammar = 'start: "+" /(.*)/ LF\n%import common.LF'

    ebnf = convert_lark_to_ebnf(grammar)

    assert 'start ::= "+" __lark_regex_0 LF' in ebnf
    assert "__lark_regex_0 ::= (([^\\n\\r]*))" in ebnf


def test_codex_apply_patch_lark_compiles_and_matches_real_patch():
    ebnf = convert_lark_to_ebnf(APPLY_PATCH_LARK)
    grammar = Grammar.from_ebnf(ebnf)
    tokenizer = TokenizerInfo([chr(index) for index in range(128)])
    matcher = GrammarMatcher(
        GrammarCompiler(tokenizer).compile_grammar(grammar),
        terminate_without_stop_token=True,
    )

    patch = (
        "*** Begin Patch\n"
        "*** Update File: greeting.txt\n"
        "@@\n"
        "-Hello, world!\n"
        "+Hello, SlimServe!\n"
        "*** End Patch\n"
    )
    assert matcher.accept_string(patch)
    assert matcher.is_completed()

    matcher.reset()
    invalid = "*** Begin Patch\n*** Add File: hello\n.txt\n+hello\n*** End Patch\n"
    assert not matcher.accept_string(invalid)


def test_lark_unknown_import_is_rejected():
    with pytest.raises(ValueError, match="Unsupported common terminal"):
        convert_lark_to_ebnf("start: WORD\n%import common.WORD")
