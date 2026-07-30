# SPDX-License-Identifier: Apache-2.0
"""Deferred `torch.compile` for module-level helpers.

A module-level `@torch.compile` fires at import, long before the engine's
`CompilationConfig` exists, so it compiles unconditionally -- emitting inductor
Triton even when the engine is configured with `CompilationMode.NONE`. That
makes the config a lie for these call sites and, on architectures where the
Triton dependency is unwanted, leaves kernels nothing asked for.

`CustomOp.maybe_compile` already applies the right rule; this is the same rule
for plain functions. The decision is deferred to first call, by which point the
config is available.
"""

import functools
from collections.abc import Callable

import torch


def lazy_compile(**compile_kwargs) -> Callable:
    """Compile on first call iff compilation is enabled.

    Falls back to the eager function when the engine sets
    `CompilationMode.NONE` or an `eager` backend, matching
    `CustomOp.maybe_compile`.

    Args:
        **compile_kwargs: Forwarded to `torch.compile`.

    Returns:
        A decorator producing a function that resolves on first invocation.
    """

    def decorator(fn: Callable) -> Callable:
        resolved: list[Callable] = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not resolved:
                target = _resolve(fn, compile_kwargs)
                if target is None:
                    # Config not readable yet; stay eager for this call and try
                    # again rather than caching a guess.
                    return fn(*args, **kwargs)
                resolved.append(target)
            return resolved[0](*args, **kwargs)

        return wrapper

    return decorator


def _resolve(fn: Callable, compile_kwargs: dict) -> Callable | None:
    """Returns the callable to use, or None if the config cannot be read yet.

    `get_current_vllm_config()` raises outside a `set_current_vllm_config()`
    context, and that context is exited before forward runs -- so for a
    module-level helper the config is usually unavailable at first call. Do not
    compile in that case: compilation is the surprising, expensive default to
    fall into, and silently doing it is what this decorator exists to prevent.
    """
    from vllm.config import get_cached_compilation_config
    from vllm.config.compilation import CompilationMode

    try:
        config = get_cached_compilation_config()
    except AssertionError:
        return None

    if config.mode == CompilationMode.NONE or config.backend == "eager":
        return fn
    return torch.compile(fn, **compile_kwargs)
