# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Accessors for the optional ``b12x`` package (SM12x CuTe DSL kernels)."""

import importlib
import importlib.util
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType

import torch


@dataclass(frozen=True)
class B12xWarmupUnit:
    """One JIT-compile unit collected from the model before graph capture."""

    name: str
    key: Hashable
    compile: Callable[[], None]


_HAS_B12X = importlib.util.find_spec("b12x") is not None


def _import_submodule(module_name: str) -> ModuleType | None:
    if not _HAS_B12X:
        return None
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


_B12X_SUBMODULES = {
    module_name: _import_submodule(module_name)
    for module_name in ("b12x.moe.fused_moe",)
}


def has_b12x() -> bool:
    """Return whether the b12x package is installed."""
    return _HAS_B12X


def get_b12x_fused_moe() -> ModuleType | None:
    return _B12X_SUBMODULES.get("b12x.moe.fused_moe")


def b12x_warmup_token_counts(
    *,
    max_tokens: int,
    sizes: Iterable[int] = (),
) -> tuple[int, ...]:
    # b12x deduplicates shapes that select the same internal kernel policy,
    # so hand it the complete serving shape set (graph capture sizes, compile
    # sizes, the scheduler ceilings) rather than duplicating its policy
    # heuristics here.
    counts = {1}
    counts.update(int(size) for size in sizes if int(size) > 0)
    if int(max_tokens) > 0:
        counts.add(int(max_tokens))
    return tuple(sorted(counts))


def _same_packed_layout(current, replacement) -> bool:
    if type(current) is not type(replacement):
        return False
    if isinstance(current, torch.Tensor):
        return (
            current.shape == replacement.shape
            and current.stride() == replacement.stride()
            and current.dtype == replacement.dtype
            and current.device == replacement.device
        )
    if is_dataclass(current):
        return all(
            _same_packed_layout(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )
            for field in fields(current)
        )
    return bool(current == replacement)


def _copy_packed_tensors(current, replacement) -> None:
    if isinstance(current, torch.Tensor):
        current.copy_(replacement)
    elif is_dataclass(current):
        for field in fields(current):
            _copy_packed_tensors(
                getattr(current, field.name),
                getattr(replacement, field.name),
            )


@torch.no_grad()
def reuse_packed_weight_storage(current, replacement):
    """Keep the prepared tensors' addresses when compatible weights are
    reloaded, so captured graphs stay valid across a weight reload."""
    if current is None or not _same_packed_layout(current, replacement):
        return replacement
    _copy_packed_tensors(current, replacement)
    return current
