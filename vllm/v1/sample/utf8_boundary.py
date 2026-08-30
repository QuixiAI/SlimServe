# SPDX-License-Identifier: Apache-2.0
"""UTF-8 boundary classification for thinking-budget enforcement.

When a thinking budget expires mid-multi-byte character (CJK, emoji), forcing
the reasoning-end marker at that exact step emits a broken codepoint. The
reference implementation (llama.cpp ``common/reasoning-budget.cpp``,
``REASONING_BUDGET_WAITING_UTF8``) holds forcing until the codepoint closes.

This module classifies which vocabulary tokens END on a UTF-8 codepoint
boundary. Byte-level BPE vocabularies (the GPT-2 lineage, which includes the
Qwen family) map token strings to raw bytes via the ``bytes_to_unicode``
table, so completeness is decidable statically per token id.
"""

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


def utf8_tail_complete(bs: bytes) -> bool:
    """Whether ``bs`` ends on a UTF-8 codepoint boundary.

    A token whose bytes end mid-codepoint (a dangling lead byte, or fewer
    continuation bytes than its last lead byte requires) is incomplete: text
    ending with it cannot be decoded without a replacement character.
    """
    if not bs:
        return True
    # Count trailing continuation bytes (0b10xxxxxx), at most 3.
    n_cont = 0
    i = len(bs) - 1
    while i >= 0 and n_cont < 3 and (bs[i] & 0xC0) == 0x80:
        n_cont += 1
        i -= 1
    if i < 0:
        # Nothing but continuation bytes: the codepoint's lead byte is in a
        # previous token, and this token may or may not close it. Treat as
        # incomplete only when it could still be awaiting more bytes; without
        # the lead byte the safe classification is complete (forcing after a
        # pure-continuation token cannot be *known* to split a codepoint).
        return True
    lead = bs[i]
    if (lead & 0x80) == 0:
        # ASCII lead followed by continuation bytes is invalid UTF-8 anyway;
        # complete when there are no trailing continuations.
        return n_cont == 0
    if (lead & 0xE0) == 0xC0:
        need = 1
    elif (lead & 0xF0) == 0xE0:
        need = 2
    elif (lead & 0xF8) == 0xF0:
        need = 3
    else:
        # Invalid lead byte: nothing sensible to hold for.
        return True
    return n_cont >= need


def _bytes_to_unicode() -> dict[int, str]:
    """The fixed GPT-2 byte<->unicode table used by byte-level BPE
    vocabularies (inlined: its import path moves between transformers
    versions)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def incomplete_utf8_token_ids(tokenizer) -> list[int]:
    """Token ids whose byte content ends mid-codepoint, or [] when the
    vocabulary's byte mapping cannot be recovered (feature inert, matching
    llama.cpp's ``vocab == nullptr`` behavior)."""
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return []
    u2b = {u: b for b, u in _bytes_to_unicode().items()}
    incomplete: list[int] = []
    unmapped = 0
    for tok_s, tid in vocab.items():
        try:
            bs = bytes(u2b[c] for c in tok_s)
        except KeyError:
            # Special/added tokens are not byte-mapped; they are always
            # codepoint-complete.
            unmapped += 1
            continue
        if not utf8_tail_complete(bs):
            incomplete.append(tid)
    if incomplete:
        logger.debug(
            "utf8 boundary table: %d incomplete-tail tokens, %d unmapped",
            len(incomplete),
            unmapped,
        )
    return incomplete
