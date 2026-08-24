# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import xgrammar as xgr

from vllm.v1.structured_output.backend_xgrammar import _mark_added_special_tokens


def _token_allowed(bitmask: torch.Tensor, token_id: int) -> bool:
    return bool((int(bitmask[0, token_id // 32]) >> (token_id % 32)) & 1)


def test_added_special_tokens_are_excluded_from_permissive_regex() -> None:
    tokenizer_info = xgr.TokenizerInfo(
        [b"<bos>", b"a", b"/", b"."],
        vocab_type=xgr.VocabType.RAW,
    )
    tokenizer = SimpleNamespace(
        added_tokens_decoder={
            0: SimpleNamespace(special=True),
            1: SimpleNamespace(special=False),
            # IDs outside the target model vocabulary must be ignored. This
            # mirrors wrappers that report an inflated added-token range.
            10: SimpleNamespace(special=True),
        }
    )

    fixed_info = _mark_added_special_tokens(tokenizer_info, tokenizer)
    compiler = xgr.GrammarCompiler(fixed_info)
    matcher = xgr.GrammarMatcher(compiler.compile_regex(r".+"))
    bitmask = xgr.allocate_token_bitmask(1, fixed_info.vocab_size)
    matcher.fill_next_token_bitmask(bitmask, 0)

    assert not _token_allowed(bitmask, 0)
    assert _token_allowed(bitmask, 1)
    assert _token_allowed(bitmask, 2)
    assert fixed_info.decoded_vocab[0] == b""


def test_correct_special_token_metadata_is_reused() -> None:
    tokenizer_info = xgr.TokenizerInfo(
        [b"", b"a"],
        vocab_type=xgr.VocabType.RAW,
    )
    tokenizer = SimpleNamespace(added_tokens_decoder={0: SimpleNamespace(special=True)})

    assert _mark_added_special_tokens(tokenizer_info, tokenizer) is tokenizer_info
