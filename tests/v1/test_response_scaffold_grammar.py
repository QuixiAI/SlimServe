# SPDX-License-Identifier: Apache-2.0
"""The chat template's post-reasoning scaffold is tolerated by JSON grammars.

Qwen3-family templates render assistant turns as ``'</think>\n\n' + content``,
so the model (and its MTP drafter) naturally emits the ``\n\n`` scaffold
before a constrained JSON payload. xgrammar's JSON root accepts no leading
whitespace, so without the wrap the first constrained token fights the
model's natural continuation (rejected drafts, off-distribution forcing).
The reasoning parser declares the scaffold; the xgrammar backend accepts it
at most once, exactly as the template renders it.
"""

from types import SimpleNamespace

import pytest

xgr = pytest.importorskip("xgrammar")

from vllm.parser.engine.registered_adapters import Qwen3ParserReasoningAdapter
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

SCHEMA = (
    '{"type": "object", "properties": {"city": {"type": "string"}},'
    ' "required": ["city"]}'
)


@pytest.fixture(scope="module")
def tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture(scope="module")
def compiler(tokenizer):
    info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=len(tokenizer.get_vocab())
    )
    return xgr.GrammarCompiler(info)


def _compile(compiler, scaffold: str):
    backend = SimpleNamespace(
        compiler=compiler,
        disable_any_whitespace=False,
        response_scaffold=scaffold,
    )
    return XgrammarBackend._compile_json_schema(backend, SCHEMA)


def _matcher(compiler, scaffold: str):
    return xgr.GrammarMatcher(_compile(compiler, scaffold))


def test_qwen3_parser_declares_template_scaffold(tokenizer):
    adapter = Qwen3ParserReasoningAdapter(tokenizer=tokenizer)
    assert adapter.response_scaffold == "\n\n"


def test_scaffold_prefix_accepted(tokenizer, compiler):
    m = _matcher(compiler, "\n\n")
    for token in tokenizer.encode('\n\n{"city": "Tokyo"}', add_special_tokens=False):
        assert m.accept_token(token), token
    assert m.is_terminated() or m.accept_token(tokenizer.eos_token_id)


def test_scaffold_split_across_tokens_accepted(tokenizer, compiler):
    m = _matcher(compiler, "\n\n")
    (nl,) = tokenizer.encode("\n", add_special_tokens=False)
    (brace,) = tokenizer.encode("{", add_special_tokens=False)
    assert m.accept_token(nl)
    assert m.accept_token(nl)
    assert m.accept_token(brace)


def test_scaffold_is_optional(tokenizer, compiler):
    m = _matcher(compiler, "\n\n")
    (brace,) = tokenizer.encode("{", add_special_tokens=False)
    assert m.accept_token(brace)


def test_scaffold_not_repeatable(tokenizer, compiler):
    m = _matcher(compiler, "\n\n")
    (nn,) = tokenizer.encode("\n\n", add_special_tokens=False)
    (nl,) = tokenizer.encode("\n", add_special_tokens=False)
    assert m.accept_token(nn)
    assert not m.accept_token(nl)


def test_empty_scaffold_keeps_strict_root(tokenizer, compiler):
    m = _matcher(compiler, "")
    (nn,) = tokenizer.encode("\n\n", add_special_tokens=False)
    (brace,) = tokenizer.encode("{", add_special_tokens=False)
    assert not m.accept_token(nn)
    assert m.accept_token(brace)
