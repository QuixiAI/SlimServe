# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import dataclasses
import functools
import os
import sys
from argparse import Namespace
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from logging import Logger
from string import Template
from typing import Any, TypeVar

import anyio
import regex as re
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask, BackgroundTasks
from starlette.types import Receive, Scope, Send

from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.engine.protocol import StreamOptions
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm.logger import current_formatter_type, init_logger
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

logger = init_logger(__name__)

_SSEItem = TypeVar("_SSEItem")
SSE_KEEPALIVE_INTERVAL_SECONDS = 30.0
SSE_CLEANUP_TIMEOUT_SECONDS = 5.0

VLLM_SUBCMD_PARSER_EPILOG = (
    "For full list:            vllm {subcmd} --help=all\n"
    "For a section:            vllm {subcmd} --help=ModelConfig    (case-insensitive)\n"  # noqa: E501
    "For a flag:               vllm {subcmd} --help=max-model-len  (_ or - accepted)\n"  # noqa: E501
    "Documentation:            https://docs.vllm.ai\n"
)


async def listen_for_disconnect(request: Request) -> None:
    """Returns if a disconnect message is received"""
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            # If load tracking is enabled *and* the counter exists, decrement
            # it. Combines the previous nested checks into a single condition
            # to satisfy the linter rule.
            if getattr(
                request.app.state, "enable_server_load_tracking", False
            ) and hasattr(request.app.state, "server_load_metrics"):
                request.app.state.server_load_metrics -= 1
            break


def with_cancellation(handler_func):
    """Decorator that allows a route handler to be cancelled by client
    disconnections.

    This does _not_ use request.is_disconnected, which does not work with
    middleware. Instead this follows the pattern from
    starlette.StreamingResponse, which simultaneously awaits on two tasks- one
    to wait for an http disconnect message, and the other to do the work that we
    want done. When the first task finishes, the other is cancelled.

    A core assumption of this method is that the body of the request has already
    been read. This is a safe assumption to make for fastapi handlers that have
    already parsed the body of the request into a pydantic model for us.
    This decorator is unsafe to use elsewhere, as it will consume and throw away
    all incoming messages for the request while it looks for a disconnect
    message.

    In the case where a `StreamingResponse` is returned by the handler, this
    wrapper will stop listening for disconnects and instead the response object
    will start listening for disconnects.
    """

    # Functools.wraps is required for this wrapper to appear to fastapi as a
    # normal route handler, with the correct request type hinting.
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        # The request is either the second positional arg or `raw_request`
        request = args[1] if len(args) > 1 else kwargs["raw_request"]

        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))

        done, pending = await asyncio.wait(
            [handler_task, cancellation_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()

        if handler_task in done:
            return handler_task.result()
        return None

    return wrapper


def decrement_server_load(request: Request):
    request.app.state.server_load_metrics -= 1


async def sse_with_keepalive(
    content: AsyncIterable[_SSEItem],
    interval_seconds: float = SSE_KEEPALIVE_INTERVAL_SECONDS,
) -> AsyncIterator[_SSEItem | str]:
    """Emit an SSE comment when an upstream stream is temporarily silent.

    The pending ``__anext__`` call must survive a heartbeat timeout. Using
    ``asyncio.wait_for`` here would cancel it and can close the model-output
    generator on the first quiet interval. SSE clients ignore comment lines,
    while proxies still observe bytes on the connection.
    """
    if interval_seconds <= 0:
        raise ValueError("SSE keepalive interval must be positive")

    iterator = aiter(content)
    pending: asyncio.Future[_SSEItem] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))

            done, _ = await asyncio.wait({pending}, timeout=interval_seconds)
            if not done:
                yield ": keepalive\n\n"
                continue

            completed = pending
            pending = None
            try:
                item = completed.result()
            except StopAsyncIteration:
                return
            yield item
    finally:
        await _close_sse_stream(iterator, pending)


async def _close_sse_stream(iterator, pending=None) -> None:
    """Finish cancellation outside the response's cancelled AnyIO scope.

    A second cancellation while awaiting ``pending`` can interrupt the engine's
    abort send. Cancel it once, and shield its cleanup until the deadline;
    only a timeout escalates to cancellation of the abort cleanup itself.
    Cleanup errors must not replace a stream/send exception already in flight.
    """

    original_exception = sys.exception()

    async def close() -> None:
        try:
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await asyncio.shield(pending)
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()

    task = asyncio.create_task(close())
    cancelled = None

    async def wait_for_cleanup(tasks, timeout: float) -> None:
        nonlocal cancelled
        deadline = asyncio.get_running_loop().time() + timeout
        with anyio.CancelScope(shield=True):
            while any(not task.done() for task in tasks):
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    # Do not forward repeated cancellation into the source's
                    # in-progress abort cleanup.
                    await asyncio.wait(tasks, timeout=remaining)
                except asyncio.CancelledError as exc:
                    cancelled = exc

    await wait_for_cleanup({task}, SSE_CLEANUP_TIMEOUT_SECONDS)

    def consume_result(done: asyncio.Task) -> None:
        try:
            done.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("SSE stream cleanup failed")

    if task.done():
        consume_result(task)
    else:
        # The graceful abort deadline has expired. Cancel and briefly drain
        # cooperative tasks rather than leaving a shielded anext detached.
        # A source that ignores cancellation cannot be forcibly terminated.
        remaining_tasks = {task}
        if pending is not None and not pending.done():
            pending.cancel()
            remaining_tasks.add(pending)
        task.cancel()
        await wait_for_cleanup(remaining_tasks, min(1.0, SSE_CLEANUP_TIMEOUT_SECONDS))
        for remaining_task in remaining_tasks:
            if remaining_task.done():
                consume_result(remaining_task)
            else:
                remaining_task.add_done_callback(consume_result)
        logger.warning("SSE stream cleanup exceeded %.1fs", SSE_CLEANUP_TIMEOUT_SECONDS)
    if cancelled is not None and original_exception is None:
        raise cancelled


class _SSEKeepaliveResponse(StreamingResponse):
    """Own iterator cleanup even when Starlette stops during ASGI send."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await _close_sse_stream(self.body_iterator)


def load_aware_call(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        raw_request = kwargs.get("raw_request", args[1] if len(args) > 1 else None)

        if raw_request is None:
            raise ValueError(
                "raw_request required when server load tracking is enabled"
            )

        load_tracking = getattr(
            raw_request.app.state, "enable_server_load_tracking", False
        )

        if load_tracking:
            # ensure the counter exists
            if not hasattr(raw_request.app.state, "server_load_metrics"):
                raw_request.app.state.server_load_metrics = 0

            raw_request.app.state.server_load_metrics += 1
        try:
            response = await func(*args, **kwargs)
        except Exception:
            if load_tracking:
                raw_request.app.state.server_load_metrics -= 1
            raise

        if (
            isinstance(response, StreamingResponse)
            and response.media_type == "text/event-stream"
        ):
            original_response = response
            response = _SSEKeepaliveResponse(
                content=sse_with_keepalive(original_response.body_iterator),
                status_code=original_response.status_code,
                media_type=original_response.media_type,
                background=original_response.background,
            )
            # Preserve duplicate headers (e.g. Set-Cookie) and exact wire values.
            response.raw_headers = original_response.raw_headers

        if not load_tracking:
            return response

        if isinstance(response, (JSONResponse, StreamingResponse)):
            if response.background is None:
                response.background = BackgroundTask(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTasks):
                response.background.add_task(decrement_server_load, raw_request)
            elif isinstance(response.background, BackgroundTask):
                # Convert the single BackgroundTask to BackgroundTasks
                # and chain the decrement_server_load task to it
                tasks = BackgroundTasks()
                tasks.add_task(
                    response.background.func,
                    *response.background.args,
                    **response.background.kwargs,
                )
                tasks.add_task(decrement_server_load, raw_request)
                response.background = tasks
        else:
            raw_request.app.state.server_load_metrics -= 1

        return response

    return wrapper


def cli_env_setup():
    # The safest multiprocessing method is `spawn`, as the default `fork` method
    # is not compatible with some accelerators. The default method will be
    # changing in future versions of Python, so we should use it explicitly when
    # possible.
    #
    # We only set it here in the CLI entrypoint, because changing to `spawn`
    # could break some existing code using vLLM as a library. `spawn` will cause
    # unexpected behavior if the code is not protected by
    # `if __name__ == "__main__":`.
    #
    # References:
    # - https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    # - https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing
    # - https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    # - https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html?highlight=multiprocessing#torch-multiprocessing-for-dataloaders
    # This tree serves one model on one known-good ROCm box; fork is safe
    # here and saves ~10 s of re-imports per child process generation.
    # `_maybe_force_spawn` still upgrades to spawn when CUDA is already
    # initialized (or Ray/WSL/NUMA demand it), so leave the env untouched
    # and let it decide at process-creation time.
    if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
        logger.debug("Leaving VLLM_WORKER_MULTIPROC_METHOD unset (fork default)")


def get_max_tokens(
    max_model_len: int,
    max_tokens: int | None,
    input_length: int,
    default_sampling_params: dict,
    override_max_tokens: int | None = None,
    truncate_prompt_tokens: int | None = None,
) -> int:
    if truncate_prompt_tokens is not None:
        limit = truncate_prompt_tokens
        input_length = min(
            input_length,
            max_model_len if limit == -1 else limit,
        )
    if max_model_len < input_length:
        raise ValueError(
            f"Input length ({input_length}) exceeds model's maximum "
            f"context length ({max_model_len})."
        )
    model_max_tokens = max_model_len - input_length
    platform_max_tokens = current_platform.get_max_output_tokens(input_length)
    fallback_max_tokens = (
        max_tokens
        if max_tokens is not None
        else default_sampling_params.get("max_tokens")
    )

    return min(
        val
        for val in (
            model_max_tokens,
            fallback_max_tokens,
            override_max_tokens,
            platform_max_tokens,
        )
        if val is not None
    )


def get_non_default_args(args: Namespace | EngineArgs) -> dict[str, Any]:
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    non_default_args = {}

    # Handle Namespace
    if isinstance(args, Namespace):
        parser = make_arg_parser(FlexibleArgumentParser())
        for arg, default in vars(parser.parse_args([])).items():
            if default != getattr(args, arg):
                non_default_args[arg] = getattr(args, arg)

    # Handle EngineArgs instance
    elif isinstance(args, EngineArgs):
        default_args = EngineArgs(model=args.model)  # Create default instance
        for field in dataclasses.fields(args):
            current_val = getattr(args, field.name)
            default_val = getattr(default_args, field.name)
            if current_val != default_val:
                non_default_args[field.name] = current_val
        if default_args.model != EngineArgs.model:
            non_default_args["model"] = default_args.model
    else:
        raise TypeError(
            "Unsupported argument type. Must be Namespace or EngineArgs instance."
        )

    return non_default_args


def _jsonify_arg_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            key: _jsonify_arg_value(val)
            for key, val in dataclasses.asdict(value).items()
        }
    if isinstance(value, dict):
        return {str(key): _jsonify_arg_value(val) for key, val in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify_arg_value(item) for item in value]
    if (model_dump := getattr(value, "model_dump", None)) is not None:
        return _jsonify_arg_value(model_dump(mode="json"))
    if (to_dict := getattr(value, "dict", None)) is not None:
        return _jsonify_arg_value(to_dict())
    return repr(value)


def jsonify_non_default_args(
    args: Namespace | EngineArgs,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    non_default_args = get_non_default_args(args)
    if exclude is not None:
        for key in exclude:
            non_default_args.pop(key, None)

    return {key: _jsonify_arg_value(value) for key, value in non_default_args.items()}


def log_non_default_args(args: Namespace | EngineArgs):
    non_default_args = get_non_default_args(args)
    logger.info("non-default args: %s", non_default_args)


def should_include_usage(
    stream_options: StreamOptions | None, enable_force_include_usage: bool
) -> tuple[bool, bool]:
    if enable_force_include_usage:
        return True, True
    if stream_options:
        include_usage = bool(stream_options.include_usage)
        include_continuous_usage = include_usage and bool(
            stream_options.continuous_usage_stats
        )
    else:
        include_usage, include_continuous_usage = False, False
    return include_usage, include_continuous_usage


def process_lora_modules(
    args_lora_modules: list[LoRAModulePath], default_mm_loras: dict[str, str] | None
) -> list[LoRAModulePath]:
    from vllm.entrypoints.openai.models.serving import LoRAModulePath

    lora_modules = args_lora_modules
    if default_mm_loras:
        default_mm_lora_paths = [
            LoRAModulePath(
                name=modality,
                path=lora_path,
            )
            for modality, lora_path in default_mm_loras.items()
        ]
        if args_lora_modules is None:
            lora_modules = default_mm_lora_paths
        else:
            lora_modules += default_mm_lora_paths
    return lora_modules


def sanitize_message(message: str) -> str:
    """Strip memory addresses, tracebacks, and file paths from error messages."""
    message = re.sub(r" at 0x[0-9a-f]+>", ">", message)
    message = re.sub(r'\n?\s*File "[^"]+", line \d+, in \S+(\n\s+.*)?', "", message)
    message = re.sub(
        r"/(?:home|usr|opt|var|tmp|root|lib|mnt|srv)(?:/[\w.\-]+)+", "<path>", message
    )
    message = re.sub(r"(?:/[\w\-]+)+/[\w\-]+\.\w+", "<path>", message)
    return message.strip()


def log_version_and_model(lgr: Logger, version: str, model_name: str) -> None:
    if envs.VLLM_DISABLE_LOG_LOGO or (formatter := current_formatter_type(lgr)) is None:
        message = "vLLM server version %s, serving model %s"
    else:
        logo_template = Template(
            "\n       ${w}█     █     █▄   ▄█${r}\n"
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}  version ${w}%s${r}\n"
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}  model   ${w}%s${r}\n"
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}\n"
        )
        colors = {
            "w": "\033[1m",  # bold, default foreground
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
        }
        if formatter != "color":
            # monochrome logo (no ansi escape codes)
            colors = dict.fromkeys(colors, "")

        message = logo_template.substitute(colors)

    lgr.info(message, version, model_name)


async def validate_json_request(raw_request: Request):
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=["Unsupported Media Type: Only 'application/json' is allowed"]
        )
