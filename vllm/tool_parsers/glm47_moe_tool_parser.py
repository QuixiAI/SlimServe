# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from functools import lru_cache

from openai.types.responses import CustomTool
from xgrammar import (
    Grammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTag,
    TokenizerInfo,
)
from xgrammar.structural_tag import ConstStringFormat, SequenceFormat, TagFormat

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_thinking_compat import (
    glm53_tool_compat_enabled,
)
from vllm.parser.engine.registered_adapters import Glm47MoeParserToolAdapter
from vllm.tool_parsers.structural_tag_registry import (
    custom_tool_as_function_schema,
    get_custom_tool_input_format,
    register_custom_tool_format,
    replace_custom_tool_payloads,
)
from vllm.tool_parsers.utils import (
    allowed_tool_choice_names,
    find_tool_name,
    named_tool_choice_name,
    tool_choice_mode,
)


@lru_cache(maxsize=128)
def _compile_custom_payload_grammar(syntax: str, definition: str):
    """Compile a custom-tool grammar against a tokenizer-neutral byte vocab."""
    compiler = GrammarCompiler(TokenizerInfo([bytes([i]) for i in range(256)]))
    if syntax == "lark":
        from vllm.v1.structured_output.utils import convert_lark_to_ebnf

        grammar = Grammar.from_ebnf(convert_lark_to_ebnf(definition))
        return compiler.compile_grammar(grammar)
    if syntax == "regex":
        return compiler.compile_regex(definition)
    raise ValueError(f"Unsupported custom-tool grammar syntax: {syntax!r}")


def _matcher_is_complete(matcher: GrammarMatcher) -> bool:
    """Support both the old and new XGrammar completion method names."""
    is_completed = getattr(matcher, "is_completed", None)
    return is_completed() if is_completed is not None else matcher.is_terminated()


def _validate_custom_payload(
    tool: CustomTool,
    payload: str,
) -> tuple[bool, bool]:
    """Return ``(valid_prefix, complete)`` for a grammar custom payload."""
    input_format = tool.format
    if input_format is None or input_format.type != "grammar":
        return False, False
    try:
        compiled = _compile_custom_payload_grammar(
            input_format.syntax,
            input_format.definition,
        )
        matcher = GrammarMatcher(compiled, terminate_without_stop_token=True)
        valid_prefix = matcher.accept_string(payload)
        return valid_prefix, valid_prefix and _matcher_is_complete(matcher)
    except (RuntimeError, ValueError):
        # Request validation normally rejects invalid grammars before parsing.
        # The recovery path must still fail closed if one reaches it directly.
        return False, False


@register_custom_tool_format(
    "glm_4_7",
    as_function_tool=custom_tool_as_function_schema,
)
def _apply_glm47_custom_tool_format(
    structural_tag: StructuralTag,
    custom_tools: dict[str, CustomTool],
) -> None:
    """Constrain raw custom input inside GLM's native XML argument tags."""
    tool_prefix = "<tool_call>"

    def match_tool_name(tag: TagFormat) -> str | None:
        begin = tag.begin
        if isinstance(begin, str) and begin.startswith(tool_prefix):
            return begin[len(tool_prefix) :]
        return None

    def make_content(custom_tool: CustomTool) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value="<arg_key>input</arg_key><arg_value>"),
                get_custom_tool_input_format(custom_tool),
                ConstStringFormat(value="</arg_value>"),
            ]
        )

    replace_custom_tool_payloads(
        structural_tag,
        custom_tools,
        match_tool_name=match_tool_name,
        make_content=make_content,
    )


class Glm47MoeModelToolParser(Glm47MoeParserToolAdapter):  # type: ignore[valid-type, misc]
    supports_required_and_named = False
    structural_tag_model = "glm_4_7"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glm53_json_streaming = False
        self._glm53_json_streaming_emitted = False
        self._glm53_json_streaming_text = ""
        self._glm53_custom_streaming = False
        self._glm53_custom_streaming_passthrough = False
        self._glm53_custom_streaming_text = ""
        self._glm53_custom_streaming_tool: CustomTool | None = None

    def get_structural_tag(self, request, *, reasoning=False):
        """Use GLM-5.3's JSON-array path for unbounded parallel requirements.

        XGrammar's native GLM ``required`` format is an unbounded sequence of
        XML tool tags. GLM-5.3 models can greedily repeat a correct parallel call
        set in that format instead of ending the turn. In that case, fall back
        to the existing whole-array JSON schema and GLM-5.3-only array parser.
        Responses ``max_tool_calls`` cannot make this finite: the API field
        limits processed built-in tools, not generated function/custom calls.
        """
        allowed_names = allowed_tool_choice_names(request.tool_choice)
        eligible_tool_count = (
            len(allowed_names)
            if allowed_names is not None
            else len(request.tools or ())
        )
        has_custom_tools = bool(request.tools) and any(
            isinstance(tool, CustomTool) for tool in request.tools
        )
        if (
            glm53_tool_compat_enabled()
            and tool_choice_mode(request.tool_choice) == "required"
            and request.parallel_tool_calls is not False
            and eligible_tool_count >= 1
            and not has_custom_tools
        ):
            return None
        return super().get_structural_tag(request, reasoning=reasoning)

    def _extract_glm53_json_array_calls(self, model_output, request):
        """Parse GLM-5.3's alternate native parallel-call representation.

        With unconstrained parallel generation GLM-5.3 can emit the
        tool schema's JSON array directly instead of its documented XML
        envelope. Keep this fallback GLM-5.3-only and accept only a complete,
        all-tool array whose names were declared by the request.
        """
        if (
            not glm53_tool_compat_enabled()
            or request.tool_choice == "none"
            or not request.tools
            or any(isinstance(tool, CustomTool) for tool in request.tools)
        ):
            return None

        candidate = model_output.strip()
        if not (candidate.startswith("[") and candidate.endswith("]")):
            return None
        try:
            raw_calls = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(raw_calls, list) or not raw_calls:
            return None

        forced_name = named_tool_choice_name(request.tool_choice)
        tool_calls = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                return None
            name = raw_call.get("name")
            parameters = raw_call.get("parameters")
            if (
                not isinstance(name, str)
                or not find_tool_name(request.tools, name)
                or (forced_name is not None and name != forced_name)
                or not isinstance(parameters, dict)
            ):
                return None
            try:
                arguments = json.dumps(
                    parameters,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                return None
            tool_calls.append(
                ToolCall(function=FunctionCall(name=name, arguments=arguments))
            )

        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=None,
        )

    @staticmethod
    def _eligible_raw_custom_tool(request) -> CustomTool | None:
        """Return the sole eligible grammar custom tool, if unambiguous."""
        if (
            not glm53_tool_compat_enabled()
            or tool_choice_mode(request.tool_choice) == "none"
            or not request.tools
        ):
            return None

        selected_names = allowed_tool_choice_names(request.tool_choice)
        forced_name = named_tool_choice_name(request.tool_choice)
        if forced_name is not None:
            selected_names = frozenset((forced_name,))

        eligible_tools = [
            tool
            for tool in request.tools
            if selected_names is None or getattr(tool, "name", None) in selected_names
        ]
        if len(eligible_tools) != 1 or not isinstance(eligible_tools[0], CustomTool):
            return None

        tool = eligible_tools[0]
        if tool.format is None or tool.format.type != "grammar":
            return None
        return tool

    def _extract_glm53_raw_custom_call(self, model_output, request):
        """Recover a grammar-valid raw payload emitted without GLM's envelope."""
        tool = self._eligible_raw_custom_tool(request)
        if tool is None:
            return None
        _, complete = _validate_custom_payload(tool, model_output)
        if not complete:
            return None
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name=tool.name,
                        arguments=model_output,
                    )
                )
            ],
            content=None,
        )

    def extract_tool_calls(self, model_output, request):
        result = super().extract_tool_calls(model_output, request)
        if result.tools_called:
            return result
        return (
            self._extract_glm53_json_array_calls(model_output, request)
            or self._extract_glm53_raw_custom_call(model_output, request)
            or result
        )

    def _stream_raw_custom_payload(
        self,
        *,
        current_text,
        delta_text,
        current_token_ids,
        request,
    ):
        """Buffer a possible bare custom payload without leaking it as text."""
        tool = self._eligible_raw_custom_tool(request)
        if tool is None:
            return False, None

        if self._glm53_custom_streaming_passthrough:
            # GLM-5.3 can spell its native XML delimiters with ordinary
            # tokens even though the same strings also exist as special
            # tokens in the vocabulary.  Once token IDs have been observed,
            # the generic parser engine intentionally refuses to recognize a
            # textual spelling of a special-token terminal (to avoid treating
            # a delimiter mentioned in prose as structure).  A sole eligible
            # custom tool is unambiguous here, and guided decoding owns the
            # entire native envelope, so keep this stream in text-terminal
            # mode through the closing tag.
            return True, super().extract_tool_calls_streaming(
                previous_text="",
                current_text=delta_text,
                delta_text=delta_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )

        # Engine-backed DelegatingParser streams one delta at a time rather
        # than an accumulated ``current_text``. Keep our own candidate buffer
        # so validation and any replay into the native parser see the complete
        # response. Direct tool-parser callers may provide cumulative text;
        # once buffering begins, appending only ``delta_text`` handles both.
        candidate_text = (
            self._glm53_custom_streaming_text + delta_text
            if self._glm53_custom_streaming
            else current_text
        )

        # Hold a partial native opening marker until it is distinguishable
        # from a raw grammar payload. Once the full marker is present, replay
        # the buffered text through the native parser in one feed.
        native_open = "<tool_call>"
        undecided_native_prefix = native_open.startswith(candidate_text)
        native_envelope = candidate_text.startswith(native_open)
        valid_prefix = False
        if not native_envelope and not undecided_native_prefix:
            valid_prefix, _ = _validate_custom_payload(tool, candidate_text)

        if undecided_native_prefix or valid_prefix:
            self._glm53_custom_streaming = True
            self._glm53_custom_streaming_text = candidate_text
            self._glm53_custom_streaming_tool = tool
            return True, None

        # This can happen after one or more chunks were held as a possible
        # raw payload. Feed the complete accumulated text to the native parser
        # once, then continue it normally for subsequent chunks.
        if self._glm53_custom_streaming or native_envelope:
            self._glm53_custom_streaming = False
            self._glm53_custom_streaming_text = ""
            self._glm53_custom_streaming_tool = None
            self._glm53_custom_streaming_passthrough = True
            return True, super().extract_tool_calls_streaming(
                previous_text="",
                current_text=candidate_text,
                delta_text=candidate_text,
                previous_token_ids=[],
                current_token_ids=[],
                delta_token_ids=[],
                request=request,
            )

        self._glm53_custom_streaming_passthrough = True
        return False, None

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        json_candidate = (
            self._glm53_json_streaming_text + delta_text
            if self._glm53_json_streaming
            else current_text
        )
        is_candidate = (
            glm53_tool_compat_enabled()
            and request.tool_choice != "none"
            and bool(request.tools)
            and not any(isinstance(tool, CustomTool) for tool in request.tools)
            and json_candidate.lstrip().startswith("[")
        )
        if self._glm53_json_streaming or is_candidate:
            self._glm53_json_streaming = True
            self._glm53_json_streaming_text = json_candidate
            # A complete constrained array is commonly delivered one chunk
            # before the engine's empty finish notification.  Re-parsing the
            # unchanged buffer on that final notification used to emit every
            # call twice.  Once the array has produced its semantic delta,
            # later chunks cannot add another element (the closing bracket
            # has already been consumed), so emission is strictly one-shot.
            if self._glm53_json_streaming_emitted:
                return None
            result = self._extract_glm53_json_array_calls(json_candidate, request)
            if result is None:
                # Buffer a possible array so raw JSON is never leaked as text
                # before the closing bracket lets us validate the calls.
                return None
            self._glm53_json_streaming_emitted = True
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=index,
                        id=tool_call.id,
                        type="function",
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                    for index, tool_call in enumerate(result.tool_calls)
                ]
            )
        handled, custom_delta = self._stream_raw_custom_payload(
            current_text=current_text,
            delta_text=delta_text,
            current_token_ids=current_token_ids,
            request=request,
        )
        if handled:
            return custom_delta
        return super().extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )

    def finish_streaming(self):
        if self._glm53_json_streaming:
            emitted = self._glm53_json_streaming_emitted
            text = self._glm53_json_streaming_text
            self._glm53_json_streaming = False
            self._glm53_json_streaming_emitted = False
            self._glm53_json_streaming_text = ""
            if emitted:
                return None
            # Invalid/incomplete arrays remain ordinary content instead of
            # disappearing because they were buffered as fallback candidates.
            return DeltaMessage(content=text or None)
        if self._glm53_custom_streaming:
            text = self._glm53_custom_streaming_text
            tool = self._glm53_custom_streaming_tool
            self._glm53_custom_streaming = False
            self._glm53_custom_streaming_passthrough = False
            self._glm53_custom_streaming_text = ""
            self._glm53_custom_streaming_tool = None
            if tool is not None:
                _, complete = _validate_custom_payload(tool, text)
                if complete:
                    tool_call = ToolCall(
                        function=FunctionCall(name=tool.name, arguments=text)
                    )
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id=tool_call.id,
                                type="function",
                                function=DeltaFunctionCall(
                                    name=tool.name,
                                    arguments=text,
                                ),
                            )
                        ]
                    )
            return DeltaMessage(content=text or None)
        self._glm53_custom_streaming_passthrough = False
        return super().finish_streaming()
