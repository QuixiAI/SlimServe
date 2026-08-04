# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.engine.registered_adapters import KimiK3ParserReasoningAdapter

KimiK3ReasoningParser = KimiK3ParserReasoningAdapter

__all__ = ["KimiK3ReasoningParser", "KimiK3ParserReasoningAdapter"]
