# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field

from vllm.config.model import ModelConfig
from vllm.config.utils import config
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers import cached_tokenizer_from_config


@config
class ReasoningConfig:
    """Configuration for reasoning models.

    Set `reasoning_start_str` and `reasoning_end_str` to the strings that delimit
    the reasoning block (e.g. `"<think>"` and `"</think>"`).  The
    corresponding token IDs are derived automatically via
    `initialize_token_ids` and are not intended to be set directly.
    """

    reasoning_parser: str = ""
    """The name of the ReasoningParser to use for this model."""
    reasoning_start_str: str = ""
    """String that indicates the start of reasoning."""
    reasoning_end_str: str = ""
    """String that indicates the end of reasoning content."""
    thinking_budget_message: str = (
        "\n\nConsidering the limited time by the user, I have to give the "
        "solution based on the thinking directly now.\n"
    )
    """Wrap-up text injected (forced token-by-token, uncharged) inside the
    reasoning block once a request's thinking_token_budget consumption crosses
    `thinking_budget_nudge_fraction`. Injected mid-block rather than at
    exhaustion so the model can still react to it with its remaining budget;
    the budget itself stays a hard cutoff (the end marker is force-closed at
    100%). Empty string disables the nudge."""
    thinking_budget_nudge_fraction: float = 0.85
    """Fraction of the thinking budget at which the wrap-up message is
    injected (operator directive 2026-08-30: ~85-90%). Values outside (0, 1)
    disable the nudge."""

    _reasoning_start_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_start_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""
    _reasoning_end_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `reasoning_end_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _thinking_budget_message_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `thinking_budget_message_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _incomplete_utf8_token_ids: list[int] | None = field(
        default=None, init=False, repr=False
    )
    """Private backing field for `incomplete_utf8_token_ids`. Set by
    `initialize_token_ids`. Not intended to be configured directly."""

    _enabled: bool = field(default=False, init=False, repr=False)
    """Private field indicating whether reasoning token IDs have been initialized.
    Set to True by `initialize_token_ids` once token IDs are initialized."""

    @property
    def enabled(self) -> bool:
        """Returns True if reasoning is enabled (i.e. if token IDs have been
        initialized), False otherwise."""
        return self._enabled

    @property
    def reasoning_start_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_start_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_start_token_ids

    @property
    def reasoning_end_token_ids(self) -> list[int] | None:
        """Token IDs derived from `reasoning_end_str`. Set automatically by
        `initialize_token_ids`. Not intended to be configured directly."""
        return self._reasoning_end_token_ids

    @property
    def thinking_budget_message_token_ids(self) -> list[int]:
        """Token IDs derived from `thinking_budget_message` (empty when no
        message is configured). Set automatically by `initialize_token_ids`."""
        return self._thinking_budget_message_token_ids or []

    @property
    def forced_close_token_ids(self) -> list[int]:
        """The sequence forced when a thinking budget expires: the budget
        message (if any) followed by the reasoning-end marker."""
        return self.thinking_budget_message_token_ids + (
            self._reasoning_end_token_ids or []
        )

    @property
    def incomplete_utf8_token_ids(self) -> list[int]:
        """Token ids whose byte content ends mid-UTF-8-codepoint. Budget
        enforcement must not force the end marker right after one of these
        (it would sever a multi-byte character). Set by
        `initialize_token_ids`; empty when the vocabulary's byte mapping is
        unavailable."""
        return self._incomplete_utf8_token_ids or []

    def initialize_token_ids(self, model_config: ModelConfig) -> None:
        """Initialize reasoning token IDs from strings using the tokenizer."""
        if (
            self._reasoning_start_token_ids is not None
            and self._reasoning_end_token_ids is not None
        ):
            self._enabled = True
            return  # Already initialized

        tokenizer = cached_tokenizer_from_config(model_config=model_config)
        reasoning_start_str = self.reasoning_start_str
        reasoning_end_str = self.reasoning_end_str
        if self.reasoning_parser is not None and (
            not reasoning_start_str or not reasoning_end_str
        ):
            parser_cls = ReasoningParserManager.get_reasoning_parser(
                self.reasoning_parser
            )
            reasoning_parser = parser_cls(tokenizer)
            start_token = reasoning_parser.reasoning_start_str
            if start_token and not reasoning_start_str:
                reasoning_start_str = start_token

            end_token = reasoning_parser.reasoning_end_str
            if end_token and not reasoning_end_str:
                reasoning_end_str = end_token

        if not reasoning_start_str or not reasoning_end_str:
            # If we don't have valid strings to tokenize,
            # we can't initialize the token IDs.
            return
        self._reasoning_start_token_ids = tokenizer.encode(
            reasoning_start_str, add_special_tokens=False
        )
        self._reasoning_end_token_ids = tokenizer.encode(
            reasoning_end_str, add_special_tokens=False
        )
        if self.thinking_budget_message:
            self._thinking_budget_message_token_ids = tokenizer.encode(
                self.thinking_budget_message, add_special_tokens=False
            )
        else:
            self._thinking_budget_message_token_ids = []

        from vllm.v1.sample.utf8_boundary import incomplete_utf8_token_ids

        self._incomplete_utf8_token_ids = incomplete_utf8_token_ids(tokenizer)

        if not self._reasoning_start_token_ids or not self._reasoning_end_token_ids:
            raise ValueError(
                f"ReasoningConfig: failed to tokenize reasoning strings: "
                f"reasoning_start_str='{self.reasoning_start_str}', "
                f"reasoning_end_str='{self.reasoning_end_str}'. "
                "Ensure the strings are valid tokens in the model's vocabulary."
            )
        self._enabled = True
